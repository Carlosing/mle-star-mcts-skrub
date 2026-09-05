"""Tests for spec_resolver: LLM dotted-path spec -> seeded instance-spec (lazy)."""

import json
import os

import pandas as pd
import pytest
import skrub
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler

from machine_learning_engineering import skrub_ops
from machine_learning_engineering import spec_resolver
from machine_learning_engineering.spec_resolver import (
    parse_spec_json,
    resolve_spec,
    unknown_operators,
)

RAW = {
    "cleaner": {"params": {"drop_if_constant": {"choice": [False, True]}}},
    "vectorizer": {
        "slots": {
            "high_cardinality": ["skrub.GapEncoder", "skrub.MinHashEncoder"]
        }
    },
    "stages": [
        {
            "name": "scale",
            "options": ["skip", "sklearn.preprocessing.StandardScaler"],
        }
    ],
    "model": [
        "sklearn.ensemble.HistGradientBoostingRegressor",
        "sklearn.ensemble.RandomForestRegressor",
    ],
}


def _california() -> pd.DataFrame:
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "machine_learning_engineering",
        "tasks",
        "california-housing-prices",
        "train.csv",
    )
    return pd.read_csv(path)


def test_parse_json_fenced_and_prose():
    fenced = (
        '```json\n{"model": ["sklearn.ensemble.RandomForestRegressor"]}\n```'
    )
    assert parse_spec_json(fenced) == {
        "model": ["sklearn.ensemble.RandomForestRegressor"]
    }
    prose = 'Here you go:\n{"model": ["sklearn.ensemble.RandomForestRegressor"]}\nThanks.'
    assert parse_spec_json(prose) == {
        "model": ["sklearn.ensemble.RandomForestRegressor"]
    }


def test_parse_json_ignores_braces_in_the_models_commentary():
    """plan_author narrates its HP ranges before the plan; braces there must
    not poison the parse (a raise silently downgrades to _fallback_spec)."""
    chatty = (
        'The n_estimators range {"int": [100, 1000]} is standard, and\n'
        '{"float": [0.01, 0.3]} suits the learning rate.\n'
        '```json\n{"model": ["sklearn.ensemble.RandomForestRegressor"]}\n```'
    )
    assert parse_spec_json(chatty) == {
        "model": ["sklearn.ensemble.RandomForestRegressor"]
    }

    unfenced = (
        'Ranges like {"int": [100, 1000]} are typical.\n'
        '{"model": ["sklearn.ensemble.RandomForestRegressor"]}'
    )
    assert parse_spec_json(unfenced) == {
        "model": ["sklearn.ensemble.RandomForestRegressor"]
    }


def test_resolve_paths_to_instances():
    spec = resolve_spec(RAW, task_type="regression")

    # always-on backbones resolve to instances (knobs -> choose_* nodes)
    assert isinstance(spec["cleaner"], skrub.Cleaner)
    assert isinstance(spec["vectorizer"], skrub.TableVectorizer)

    assert spec["stages"][0]["name"] == "scale"
    assert None in spec["stages"][0]["options"]
    assert any(
        isinstance(o, StandardScaler) for o in spec["stages"][0]["options"]
    )

    # model path-list -> {class_name: instance} dict
    assert set(spec["model"]) == {
        "HistGradientBoostingRegressor",
        "RandomForestRegressor",
    }
    assert isinstance(
        spec["model"]["HistGradientBoostingRegressor"],
        HistGradientBoostingRegressor,
    )


def test_classifier_path_imports_classifier():
    spec = resolve_spec(
        {"model": ["sklearn.ensemble.HistGradientBoostingClassifier"]},
        task_type="classification",
    )
    assert isinstance(
        spec["model"]["HistGradientBoostingClassifier"],
        HistGradientBoostingClassifier,
    )


def test_open_vocab_path_not_in_registry_uses_defaults():
    # MinMaxScaler is importable + allowlisted but NOT in the curated REGISTRY.
    from sklearn.preprocessing import MinMaxScaler

    spec = resolve_spec(
        {
            "stages": [
                {
                    "name": "scale",
                    "options": ["sklearn.preprocessing.MinMaxScaler"],
                }
            ],
            "model": ["sklearn.ensemble.RandomForestRegressor"],
        }
    )
    assert any(
        isinstance(o, MinMaxScaler) for o in spec["stages"][0]["options"]
    )


def test_non_allowlisted_root_is_rejected():
    assert unknown_operators({"model": ["os.system"]}) == ["os.system"]
    with pytest.raises(ValueError):
        resolve_spec({"model": ["os.system"]})


def test_unimportable_paths_dropped_and_reported():
    raw = {
        "stages": [
            {
                "name": "fe",
                "options": [
                    "skrub.GapEncoder",
                    "sklearn.preprocessing.NotARealThing",
                ],
            }
        ],
        "model": [
            "sklearn.ensemble.RandomForestRegressor",
            "sklearn.bogus.Nope",
        ],
    }
    reported = unknown_operators(raw)
    assert "sklearn.preprocessing.NotARealThing" in reported
    assert "sklearn.bogus.Nope" in reported

    spec = resolve_spec(raw)
    import skrub

    # the bad operator is dropped; GapEncoder survives in the stage's options
    opts = spec["stages"][0]["options"]
    assert any(isinstance(o, skrub.GapEncoder) for o in opts)
    assert not any(type(o).__name__ == "NotARealThing" for o in opts)
    assert set(spec["model"]) == {"RandomForestRegressor"}  # bad model dropped


def test_operator_with_missing_optional_dep_is_dropped(monkeypatch):
    # an operator that imports fine but needs an OPTIONAL runtime dep that
    # isn't installed must be DROPPED at resolve, not crash the build later.
    # Simulate by pointing GapEncoder's guard at a nonexistent module.
    monkeypatch.setitem(
        spec_resolver._OPTIONAL_DEPS,
        "skrub.GapEncoder",
        "nonexistent_pkg_xyz_123",
    )
    assert spec_resolver._load_class("skrub.GapEncoder") is None
    # in a full plan it's dropped like any unusable operator; the run survives
    spec = resolve_spec(
        {
            "stages": [
                {
                    "name": "fe",
                    "options": ["skrub.GapEncoder", "skrub.MinHashEncoder"],
                }
            ],
            "model": ["sklearn.ensemble.RandomForestRegressor"],
        }
    )
    opts = spec["stages"][0]["options"]
    assert not any(type(o).__name__ == "GapEncoder" for o in opts)
    assert any(isinstance(o, skrub.MinHashEncoder) for o in opts)


def test_text_encoder_routed_to_shim():
    # skrub.TextEncoder resolves to the caching subclass, not the stock class —
    # same class name (so action-space labels/gating are unchanged) but a
    # genuine skrub.TextEncoder subclass. No model is loaded here: _make only
    # constructs the operator; the backbone loads lazily at fit.
    spec = resolve_spec(
        {
            "stages": [{"name": "text", "options": ["skrub.TextEncoder"]}],
            "model": ["sklearn.ensemble.RandomForestRegressor"],
        }
    )
    (enc,) = spec["stages"][0]["options"]
    assert isinstance(enc, skrub.TextEncoder)
    assert type(enc).__name__ == "TextEncoder"
    assert type(enc) is not skrub.TextEncoder  # the shim subclass


def test_text_encoder_shim_shares_one_backbone_per_model(monkeypatch):
    # The shim memoizes the loaded backbone in a MODULE-level dict keyed by the
    # load-determining params, so two distinct model_names coexist (each loaded
    # once) and a repeat config reuses the resident model. Sentinels stand in
    # for real SentenceTransformers, so nothing is downloaded.
    cls = spec_resolver._text_encoder_shim()
    cache = {}
    monkeypatch.setattr(spec_resolver, "_TEXT_BACKBONE_CACHE", cache)
    sentinel_a, sentinel_b = object(), object()
    cache[("model-a", None, None, None)] = sentinel_a
    cache[("model-b", None, None, None)] = sentinel_b

    a1 = cls(model_name="model-a")
    a2 = cls(model_name="model-a")  # a "clone" — different instance, same key
    b1 = cls(model_name="model-b")
    assert a1._estimator is sentinel_a
    assert a2._estimator is sentinel_a  # reused, not reloaded per instance
    assert b1._estimator is sentinel_b  # a different model coexists


def test_plan_has_text_encoder_detection(monkeypatch):
    enc = spec_resolver._text_encoder_shim()(model_name="model-a")

    class _FakeChoice:
        outcomes = [None, enc]

    class _PlainChoice:
        outcomes = ["GBM", "RF"]

    monkeypatch.setattr(
        skrub_ops._ev, "choices", lambda plan: {0: _PlainChoice()}
    )
    assert skrub_ops.plan_has_text_encoder(object()) is False
    monkeypatch.setattr(
        skrub_ops._ev,
        "choices",
        lambda plan: {0: _PlainChoice(), 1: _FakeChoice()},
    )
    assert skrub_ops.plan_has_text_encoder(object()) is True


def test_strict_raises_on_unknown():
    with pytest.raises(ValueError):
        resolve_spec(
            {
                "model": ["sklearn.ensemble.RandomForestRegressor"],
                "stages": [{"name": "fe", "options": ["sklearn.bogus.Nope"]}],
            },
            strict=True,
        )


def test_no_usable_model_raises():
    with pytest.raises(ValueError):
        resolve_spec({"model": ["sklearn.bogus.Nope"]})


def test_seeding_is_applied():
    spec = resolve_spec({"model": ["sklearn.ensemble.RandomForestRegressor"]})
    assert (
        spec["model"]["RandomForestRegressor"].get_params()["random_state"]
        == 42
    )


def test_resolved_spec_builds_a_plan_on_california():
    df = _california()
    spec = resolve_spec(RAW, task_type="regression")
    plan = skrub_ops.build_staged_plan(spec, df, target="median_house_value")
    space = skrub_ops.get_action_space(plan)
    assert "model" in space
    assert set(space["model"]) == {
        "HistGradientBoostingRegressor",
        "RandomForestRegressor",
    }


# --- hyperparameter search ---------------------------------------------------

HP_RAW = {
    "model": [
        {
            "name": "sklearn.ensemble.RandomForestRegressor",
            "params": {"n_estimators": {"int": [100, 500]}},
        },
        {
            "name": "sklearn.ensemble.HistGradientBoostingRegressor",
            "params": {"learning_rate": {"float": [0.01, 0.3], "log": True}},
        },
    ]
}


def test_tuned_hps_surface_in_action_space():
    df = _california()
    plan = skrub_ops.build_staged_plan(
        resolve_spec(HP_RAW, task_type="regression"),
        df,
        target="median_house_value",
    )
    space = skrub_ops.get_action_space(plan)
    assert "model" in space
    assert "model__RandomForestRegressor__n_estimators" in space
    assert space["model__RandomForestRegressor__n_estimators"][0] == 100
    assert space["model__RandomForestRegressor__n_estimators"][-1] == 500
    assert "model__HistGradientBoostingRegressor__learning_rate" in space


def test_hp_range_is_clipped_to_allowed_envelope():
    raw = {
        "model": [
            {
                "name": "sklearn.ensemble.HistGradientBoostingRegressor",
                "params": {
                    "learning_rate": {"float": [1e-6, 5.0], "log": True}
                },
            }
        ]
    }
    df = _california()
    plan = skrub_ops.build_staged_plan(
        resolve_spec(raw, task_type="regression"),
        df,
        target="median_house_value",
    )
    vals = skrub_ops.get_action_space(plan)[
        "model__HistGradientBoostingRegressor__learning_rate"
    ]
    assert min(vals) >= 0.01 - 1e-9
    assert max(vals) <= 0.3 + 1e-9


def test_unknown_param_is_dropped_not_tuned():
    raw = {
        "model": [
            {
                "name": "sklearn.ensemble.RandomForestRegressor",
                "params": {"not_a_real_param": {"int": [1, 9]}},
            }
        ]
    }
    df = _california()
    plan = skrub_ops.build_staged_plan(
        resolve_spec(raw, task_type="regression"),
        df,
        target="median_house_value",
    )
    space = skrub_ops.get_action_space(plan)
    assert not any("not_a_real_param" in k for k in space)


def test_bare_path_works_without_params():
    spec = resolve_spec({"model": ["sklearn.ensemble.RandomForestRegressor"]})
    assert (
        spec["model"]["RandomForestRegressor"].get_params()["random_state"]
        == 42
    )


def test_onehot_encoder_is_forced_dense_and_unknown_safe():
    """skrub's DataOps carry pandas frames, which cannot hold a sparse matrix:
    an unforced OneHotEncoder raised inside build_staged_plan and killed the
    whole run. handle_unknown="error" is the same class of latent crash, one
    CV fold later."""
    spec = resolve_spec(
        {
            "stages": [
                {
                    "name": "fe",
                    "options": ["sklearn.preprocessing.OneHotEncoder"],
                }
            ],
            "model": ["sklearn.ensemble.RandomForestRegressor"],
        },
        task_type="regression",
    )
    ohe = spec["stages"][0]["options"][0]
    assert ohe.sparse_output is False
    assert ohe.handle_unknown != "error"


def test_sklearn_text_vectorizers_are_dropped_as_sparse():
    """sklearn's text vectorizers emit scipy-sparse matrices skrub's pandas
    DataOps can't carry, and (unlike OneHotEncoder) have no dense flag — an
    unforced TfidfVectorizer in the encoder slot killed the whole run at
    build (the `min_df`-on-1-row preview, then a sparse-output crash). They are
    screened out by `_emits_dataframe`; skrub-native encoders survive. See
    docs/BUG_LEDGER.md."""
    spec = resolve_spec(
        {
            "stages": [
                {
                    "name": "fe",
                    "options": [
                        {
                            "name": "sklearn.feature_extraction.text.TfidfVectorizer",
                            "params": {"min_df": {"int": [1, 5]}},
                        },
                        "sklearn.feature_extraction.text.CountVectorizer",
                        "sklearn.feature_extraction.text.HashingVectorizer",
                        "skrub.StringEncoder",
                    ],
                }
            ],
            "model": ["sklearn.linear_model.LogisticRegression"],
        },
        task_type="classification",
    )
    kept = [type(o).__name__ for o in spec["stages"][0]["options"]]
    assert "StringEncoder" in kept
    assert not any(v in kept for v in ("TfidfVectorizer", "CountVectorizer"))
    assert "HashingVectorizer" not in kept


def test_sparse_output_is_not_forced_onto_kwargs_constructors():
    """LGBM/XGB take **kwargs, so a permissive `_accepts_param` check would
    inject sparse_output straight into the booster."""
    pytest.importorskip("lightgbm")
    spec = resolve_spec(
        {"model": ["lightgbm.LGBMClassifier"]}, task_type="classification"
    )
    assert "sparse_output" not in spec["model"]["LGBMClassifier"].get_params()


def test_parse_json_rejects_a_fragment_from_a_truncated_response():
    """A response cut off at its output-token cap leaves valid JSON fragments in
    the text. A brace-scan returns one, and `{"float": [0.7, 1.0]}` silently
    became "the extended plan" — Optional Feature 3 no-op'd with no error."""
    truncated = (
        '```json\n{\n "model": ["sklearn.linear_model.Ridge"],\n "scoped_encodings": [\n'
        '  {"name": "t", "options": [\n'
        '    {"name": "sklearn.feature_extraction.text.TfidfVectorizer",\n'
        '     "params": {"max_df": {"float": [0.7, 1.0]},\n'
        '                "ngram_range": {'
    )
    with pytest.raises(json.JSONDecodeError, match="truncated"):
        parse_spec_json(truncated)


def test_parse_json_relaxes_python_none_literal():
    # A reasoning model (e.g. the school/OpenAI-compatible path) emitted Python
    # `None` instead of JSON `null` in class_weight, failing json.loads and
    # dropping the WHOLE plan to the minimal fallback spec (live on
    # traffic-violations). The repair pass now recovers it — and `None` becomes
    # real JSON null (Python None), NOT the string "None" (json_repair's own
    # default, which would be an invalid class_weight).
    raw = (
        'Thinking...\n```json\n{\n "model": [{"name": '
        '"sklearn.linear_model.LogisticRegression", "params": '
        '{"class_weight": {"choice": ["balanced", None]}}}]\n}\n```'
    )
    plan = parse_spec_json(raw)
    choice = plan["model"][0]["params"]["class_weight"]["choice"]
    assert choice == ["balanced", None]
    assert None in choice  # real null, not the string "None"


def test_parse_json_none_inside_a_string_is_preserved():
    # the literal relaxation must not touch words inside string values
    plan = parse_spec_json(
        '{"model": ["x"], "note": "None of True/False apply"}'
    )
    assert plan["note"] == "None of True/False apply"


def test_parse_json_repairs_trailing_comma_on_complete_object():
    # json_repair handles complete-but-malformed JSON (trailing comma)
    plan = parse_spec_json('{"model": ["sklearn.svm.SVC"], "stages": [],}')
    assert plan == {"model": ["sklearn.svm.SVC"], "stages": []}


def test_parse_json_valid_input_still_takes_the_strict_path():
    # the repair pass is a fallback; already-valid JSON is unchanged
    assert parse_spec_json('{"model": ["a"]}') == {"model": ["a"]}


def test_parse_json_prefers_the_plan_over_an_illustrative_snippet():
    """The model may fence the plan and then fence an example after it."""
    two_fences = (
        '```json\n{"model": ["sklearn.ensemble.RandomForestRegressor"]}\n```\n'
        'For instance a range looks like:\n```json\n{"int": [1, 9]}\n```'
    )
    assert parse_spec_json(two_fences) == {
        "model": ["sklearn.ensemble.RandomForestRegressor"]
    }


def test_assemble_dict_instead_of_list_is_reported_not_crashed():
    """A model emitted `assemble` as a single dict (schema wants a list) with
    nested per-column operation groups. It resolved to nothing and vanished with
    no error — the whole credit-fraud sweep ran flat-table. Now it drops cleanly
    and is reported."""
    raw = {
        "assemble": {
            "table": "products",
            "main_key": "ID",
            "aux_key": "basket_ID",
            "operations": [{"cols": ["cash_price"], "operations": ["sum"]}],
        },
        "model": ["sklearn.ensemble.RandomForestClassifier"],
    }
    spec = resolve_spec(
        raw,
        task_type="classification",
        aux_schemas={"products": ["basket_ID", "cash_price"]},
        main_columns=["ID", "fraud_flag"],
    )
    assert "assemble" not in spec  # malformed -> dropped, no crash
    assert "assemble" in spec["dropped_sections"]


def test_single_option_and_dropped_stages_are_flagged():
    spec = resolve_spec(
        {
            # a whole stage whose only option is unimportable -> vanishes
            "stages": [
                {"name": "gone", "options": ["sklearn.bogus.Nope"]},
                {
                    "name": "lonely",
                    "options": ["sklearn.preprocessing.StandardScaler"],
                },
            ],
            "model": ["sklearn.ensemble.RandomForestClassifier"],
        },
        task_type="classification",
    )
    assert "stage:gone" in spec["dropped_sections"]
    assert "stage:lonely" in spec["single_option_stages"]


# --- always-on backbone: cleaner + vectorizer (LLM knobs, bare-default root) ---


def test_cleaner_backbone_resolves_and_defaults_to_bare_cleaner():
    # a bare `cleaner` key -> plain Cleaner() (skrub defaults = robust root)
    out = resolve_spec(
        {"cleaner": {}, "model": ["sklearn.linear_model.Ridge"]},
        task_type="regression",
    )
    assert isinstance(out["cleaner"], skrub.Cleaner)
    assert repr(out["cleaner"]) == "Cleaner()"


def test_cleaner_knobs_become_choose_nodes_default_first():
    # LLM authors [True, False] but skrub's default (False) must seed the root,
    # so the resolver reorders the choice to put False first
    out = resolve_spec(
        {
            "cleaner": {
                "params": {"drop_if_constant": {"choice": [True, False]}}
            },
            "model": ["sklearn.linear_model.Ridge"],
        },
        task_type="regression",
    )
    df = pd.DataFrame({"a": ["x", "y", "x", "z"], "y": [1, 2, 3, 4]})
    plan = skrub_ops.build_staged_plan(out, df, target="y")
    space = skrub_ops.get_action_space(plan)
    assert space["cleaner__Cleaner__drop_if_constant"] == ["False", "True"]
    assert (
        skrub_ops.get_default_state(plan)["cleaner__Cleaner__drop_if_constant"]
        == "False"
    )


def test_vectorizer_backbone_slots_and_scalar_knobs():
    out = resolve_spec(
        {
            "vectorizer": {
                "params": {"cardinality_threshold": {"int": [10, 40]}},
                "slots": {
                    "high_cardinality": [
                        "skrub.StringEncoder",
                        "skrub.MinHashEncoder",
                    ]
                },
            },
            "model": ["sklearn.linear_model.Ridge"],
        },
        task_type="regression",
    )
    assert isinstance(out["vectorizer"], skrub.TableVectorizer)
    df = pd.DataFrame({"t": ["a b", "c", "d e", "f"], "y": [1, 2, 3, 4]})
    plan = skrub_ops.build_staged_plan(out, df, target="y")
    space = skrub_ops.get_action_space(plan)
    assert "vectorizer__high_cardinality" in space
    assert "vectorizer__TableVectorizer__cardinality_threshold" in space


def test_vectorizer_bare_key_is_plain_tablevectorizer():
    out = resolve_spec(
        {"vectorizer": {}, "model": ["sklearn.linear_model.Ridge"]},
        task_type="regression",
    )
    assert repr(out["vectorizer"]).startswith("TableVectorizer(")


def test_backbone_drops_unknown_param_not_the_operator():
    out = resolve_spec(
        {
            "cleaner": {
                "params": {
                    "not_a_real_param": {"choice": [1, 2]},
                    "drop_if_constant": {"choice": [False, True]},
                }
            },
            "model": ["sklearn.linear_model.Ridge"],
        },
        task_type="regression",
    )
    r = repr(out["cleaner"])
    assert "not_a_real_param" not in r and "drop_if_constant" in r
