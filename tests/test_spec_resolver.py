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
from machine_learning_engineering.spec_resolver import (
    parse_spec_json,
    resolve_spec,
    unknown_operators,
)

RAW = {
    "clean_options": ["skip", "skrub.Cleaner"],
    "encoder_options": ["skrub.GapEncoder", "skrub.MinHashEncoder"],
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

    assert None in spec["clean_options"]
    assert any(isinstance(o, skrub.Cleaner) for o in spec["clean_options"])

    assert all(o is not None for o in spec["encoder_options"])
    assert any(isinstance(o, skrub.GapEncoder) for o in spec["encoder_options"])

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
        "encoder_options": [
            "skrub.GapEncoder",
            "sklearn.preprocessing.NotARealThing",
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

    # the bad encoder is dropped; GapEncoder survives, and since it's then a
    # lone required-stage option, the skrub-default companion is appended
    assert any(
        isinstance(o, skrub.GapEncoder) for o in spec["encoder_options"]
    )
    assert not any(
        type(o).__name__ == "NotARealThing" for o in spec["encoder_options"]
    )
    assert set(spec["model"]) == {"RandomForestRegressor"}  # bad model dropped


def test_strict_raises_on_unknown():
    with pytest.raises(ValueError):
        resolve_spec(
            {
                "model": ["sklearn.ensemble.RandomForestRegressor"],
                "encoder_options": ["sklearn.bogus.Nope"],
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
            "encoder_options": ["sklearn.preprocessing.OneHotEncoder"],
            "model": ["sklearn.ensemble.RandomForestRegressor"],
        },
        task_type="regression",
    )
    ohe = spec["encoder_options"][0]
    assert ohe.sparse_output is False
    assert ohe.handle_unknown != "error"


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
    became "the extended plan" — Option 3 no-op'd with no error."""
    truncated = (
        '```json\n{\n "clean_options": ["skip"],\n "scoped_encodings": [\n'
        '  {"name": "t", "options": [\n'
        '    {"name": "sklearn.feature_extraction.text.TfidfVectorizer",\n'
        '     "params": {"max_df": {"float": [0.7, 1.0]},\n'
        '                "ngram_range": {'
    )
    with pytest.raises(json.JSONDecodeError, match="truncated"):
        parse_spec_json(truncated)


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
            "encoder_options": ["skip"],  # required stage -> vanishes entirely
            "clean_options": ["skrub.Cleaner"],  # one option, nothing to search
            "model": ["sklearn.ensemble.RandomForestClassifier"],
        },
        task_type="classification",
    )
    assert "encoder_options" in spec["dropped_sections"]
    assert "clean_options" in spec["single_option_stages"]


def test_lone_encoder_gets_skrub_default_companion():
    """`encoder` is required (no auto-skip), so a single option leaves it
    un-searchable. The resolver appends skrub's TableVectorizer default
    high-cardinality encoder so the stage becomes a real 2-option search, and
    the single-option diagnostic no longer fires."""
    import skrub

    spec = resolve_spec(
        {
            "encoder_options": ["skrub.MinHashEncoder"],
            "model": ["sklearn.ensemble.HistGradientBoostingRegressor"],
        },
        task_type="regression",
    )
    enc = spec["encoder_options"]
    assert len(enc) == 2
    assert any(isinstance(o, skrub.MinHashEncoder) for o in enc)  # LLM's pick
    assert any(isinstance(o, skrub.StringEncoder) for o in enc)   # skrub default
    assert enc[0].__class__ is skrub.MinHashEncoder  # LLM's pick stays the root
    assert "encoder_options" not in (spec.get("single_option_stages") or [])


def test_lone_encoder_that_is_already_the_default_stays_flagged():
    """No meaningful companion to add — so it is left single and flagged."""
    spec = resolve_spec(
        {
            "encoder_options": ["skrub.StringEncoder"],
            "model": ["sklearn.ensemble.HistGradientBoostingRegressor"],
        },
        task_type="regression",
    )
    assert len(spec["encoder_options"]) == 1
    assert "encoder_options" in spec["single_option_stages"]
