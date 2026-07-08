"""The scope stage — per-column encoder application via `scoped_encodings`.

Pins the reopened `scope` stage: the plan may direct WHICH columns a searchable
encoder applies to (`.skb.apply(cols=...)`), instead of TableVectorizer's
uniform high-cardinality routing. The load-bearing guarantees:

- column names are validated at resolve time (LLM-invented names dropped) and
  re-checked against the dataframe at build time;
- the runtime selector is missing-tolerant (a column dropped upstream degrades
  the group to a no-op, it does not zero the rollout);
- each group is a named `scope_<group>` choice with skip default, so it flows
  through action space / targeting / injection like any other operator stage.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

skrub = pytest.importorskip("skrub")

from machine_learning_engineering import skrub_ops, spec_resolver
from machine_learning_engineering.search_loop import (
    _augment_spec,
    run_search_loop,
)

from fixtures.golden_plan import make_toy_df


# --- the missing-tolerant selector ---------------------------------------------


def test_scope_selector_tolerates_missing_columns():
    df = make_toy_df(50)
    assert skrub_ops._scope_selector(["color", "ghost"]).expand(df) == ["color"]
    # regex metacharacters in a column name must be treated literally
    df2 = df.rename(columns={"color": "color (raw)"})
    assert skrub_ops._scope_selector(["color (raw)"]).expand(df2) == [
        "color (raw)"
    ]


# --- resolver validation ---------------------------------------------------------


def test_resolve_scoped_validates_cols_and_names():
    scoped = spec_resolver._resolve_scoped(
        [
            {
                "name": "color enc",
                "cols": ["color", "ghost"],
                "options": ["skip", "skrub.GapEncoder"],
            },
            {
                "name": "all_invented",
                "cols": ["nope"],
                "options": ["skrub.GapEncoder"],
            },
            {
                "name": "no_ops",
                "cols": ["color"],
                "options": ["not.allowed.Path"],
            },
        ],
        seed=42,
        main_columns=["x1", "x2", "color"],
    )
    assert len(scoped) == 1
    group = scoped[0]
    assert group["name"] == "color_enc"  # sanitized: no spaces, no '__'
    assert group["cols"] == ["color"]  # invented column dropped
    assert type(group["options"][0]).__name__ == "GapEncoder"


def test_resolve_spec_carries_scoped_encodings():
    raw = {
        "scoped_encodings": [
            {
                "name": "color_enc",
                "cols": ["color"],
                "options": ["skrub.GapEncoder", "skrub.MinHashEncoder"],
            }
        ],
        "model": ["sklearn.ensemble.HistGradientBoostingClassifier"],
    }
    spec = spec_resolver.resolve_spec(
        raw, task_type="classification", main_columns=["x1", "x2", "color"]
    )
    assert [g["name"] for g in spec["scoped_encodings"]] == ["color_enc"]


# --- build + search integration --------------------------------------------------


@pytest.fixture(scope="module")
def scoped_plan_df():
    df = make_toy_df()
    spec = {
        "scoped_encodings": [
            {
                "name": "color_enc",
                "cols": ["color", "ghost"],
                "options": [
                    skrub.GapEncoder(n_components=3, random_state=42),
                    skrub.MinHashEncoder(n_components=4),
                ],
            }
        ],
        "model": {
            "HGB": __import__(
                "sklearn.ensemble", fromlist=["x"]
            ).HistGradientBoostingClassifier(random_state=42),
        },
    }
    return skrub_ops.build_staged_plan(spec, df), df


def test_scope_group_in_action_space_with_skip_default(scoped_plan_df):
    plan, _ = scoped_plan_df
    space = skrub_ops.get_action_space(plan)
    assert space["scope_color_enc"] == ["skip", "GapEncoder", "MinHashEncoder"]
    assert skrub_ops.get_default_state(plan)["scope_color_enc"] == "skip"


def test_scoped_rollout_is_deterministic_and_scoreable(scoped_plan_df):
    plan, df = scoped_plan_df
    rollout = skrub_ops.make_rollout_fn(plan, df)
    on = rollout({"model": "HGB", "scope_color_enc": "GapEncoder"})
    off = rollout({"model": "HGB"})
    assert on > 0.0 and off > 0.0  # both configs really evaluate
    assert on == rollout({"model": "HGB", "scope_color_enc": "GapEncoder"})


def test_augment_spec_injects_into_scope_group():
    spec = {
        "scoped_encodings": [
            {
                "name": "color_enc",
                "cols": ["color"],
                "options": [skrub.GapEncoder(n_components=3, random_state=42)],
            }
        ],
        "model": {},
    }
    new = _augment_spec(spec, "scope_color_enc", [skrub.StringEncoder()])
    labels = [type(o).__name__ for o in new["scoped_encodings"][0]["options"]]
    assert labels == ["GapEncoder", "StringEncoder"]
    # original spec untouched
    assert len(spec["scoped_encodings"][0]["options"]) == 1


def test_default_state_is_appliable_when_encoder_options_have_tuned_hps():
    """Regression: an encoder_options entry with a tunable HP (a nested
    choice) makes skrub's own describe_defaults() abbreviate it to
    "GapEncoder(...)" — which does not match get_action_space's full-repr
    label. get_default_state must not round-trip through that string (the
    resulting root state would fail apply_state and every rollout would
    silently score 0.0 for the whole search)."""
    df = make_toy_df()
    spec = spec_resolver.resolve_spec(
        {
            "encoder_options": [
                {
                    "name": "skrub.GapEncoder",
                    "params": {"n_components": {"int": [10, 50]}},
                },
                "skrub.MinHashEncoder",
            ],
            "model": ["sklearn.ensemble.HistGradientBoostingClassifier"],
        },
        task_type="classification",
    )
    plan = skrub_ops.build_staged_plan(spec, df)
    start = skrub_ops.get_default_state(plan)
    assert start["encoder"] in skrub_ops.get_action_space(plan)["encoder"]
    skrub_ops.apply_state(plan, start)  # must not raise


# --- position + additive (scoped groups v2) --------------------------------------


def test_resolve_scoped_passes_position_and_additive():
    scoped = spec_resolver._resolve_scoped(
        [
            {
                "name": "color_feats",
                "cols": ["color"],
                "additive": True,
                "options": ["skrub.GapEncoder"],
            },
            {
                "name": "num_scale",
                "cols": ["x1"],
                "position": "post_encode",
                "options": ["sklearn.preprocessing.StandardScaler"],
            },
            {
                "name": "bad_pos",
                "cols": ["x2"],
                "position": "mid_encode",
                "options": ["skrub.GapEncoder"],
            },
        ],
        seed=42,
        main_columns=["x1", "x2", "color"],
    )
    by_name = {g["name"]: g for g in scoped}
    assert by_name["color_feats"]["additive"] is True
    assert "position" not in by_name["color_feats"]  # default stays implicit
    assert by_name["num_scale"]["position"] == "post_encode"
    assert "position" not in by_name["bad_pos"]  # invalid -> pre_encode default


def test_post_encode_group_applies_after_the_vectorizer():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler

    df = make_toy_df()
    spec = {
        "scoped_encodings": [
            {
                "name": "color_enc",
                "cols": ["color"],
                "options": [skrub.GapEncoder(n_components=3, random_state=42)],
            },
            {
                "name": "num_scale",
                "cols": ["x1"],
                "position": "post_encode",
                "options": [RobustScaler()],
            },
        ],
        "model": {"HGB": HistGradientBoostingClassifier(random_state=42)},
    }
    plan = skrub_ops.build_staged_plan(spec, df)
    assert skrub_ops.get_action_space(plan)["scope_num_scale"] == [
        "skip",
        "RobustScaler",
    ]
    # DAG order: pre-encode group -> TableVectorizer -> post-encode group
    skrub_ops.apply_state(
        plan,
        {
            "model": "HGB",
            "scope_color_enc": "GapEncoder",
            "scope_num_scale": "RobustScaler",
        },
    )
    steps = skrub_ops.get_steps_summary(plan)
    assert (
        steps.index("GapEncoder")
        < steps.index("TableVectorizer")
        < steps.index("RobustScaler")
    )
    rollout = skrub_ops.make_rollout_fn(plan, df)
    assert rollout({"model": "HGB", "scope_num_scale": "RobustScaler"}) > 0.0


def test_additive_group_keeps_original_and_appends_suffixed():
    df = make_toy_df(80)
    X = skrub.var("data", df.drop(columns=["target"]))
    group = {
        "name": "color_feats",
        "additive": True,
        "options": [skrub.MinHashEncoder(n_components=4)],
    }
    node = skrub_ops._apply_scope_group(X, group, ["color"])
    base_cols = list(df.drop(columns=["target"]).columns)
    assert list(node.skb.eval().columns) == base_cols  # skip default: a no-op
    skrub_ops.apply_state(node, {"scope_color_feats": "MinHashEncoder"})
    out = node.skb.eval()
    assert "color" in out.columns  # the original column SURVIVES
    added = [c for c in out.columns if c.endswith("__color_feats")]
    assert len(added) == 4  # derived columns appended, suffixed vs collisions


def test_additive_group_end_to_end_search():
    from sklearn.ensemble import HistGradientBoostingClassifier

    df = make_toy_df()
    spec = {
        "scoped_encodings": [
            {
                "name": "color_feats",
                "cols": ["color"],
                "additive": True,
                "options": [skrub.MinHashEncoder(n_components=4)],
            }
        ],
        "model": {"HGB": HistGradientBoostingClassifier(random_state=42)},
    }
    plan = skrub_ops.build_staged_plan(spec, df)
    assert skrub_ops.get_action_space(plan)["scope_color_feats"] == [
        "skip",
        "MinHashEncoder",
    ]
    assert skrub_ops.get_default_state(plan)["scope_color_feats"] == "skip"
    rollout = skrub_ops.make_rollout_fn(plan, df)
    on = rollout({"model": "HGB", "scope_color_feats": "MinHashEncoder"})
    assert on > 0.0
    assert on == rollout(
        {"model": "HGB", "scope_color_feats": "MinHashEncoder"}
    )


def test_inject_creates_flagged_scope_groups():
    from machine_learning_engineering.search_loop import _inject

    spec = {"model": {}}
    proposals = [
        {"name": "skrub.DatetimeEncoder", "cols": ["date"], "additive": True},
        {
            "name": "sklearn.preprocessing.StandardScaler",
            "cols": ["num"],
            "position": "post_encode",
        },
    ]
    new, kept = _inject(
        spec, "encoder", proposals, 42, valid_columns=["date", "num"]
    )
    assert len(kept) == 2
    by_name = {g["name"]: g for g in new["scoped_encodings"]}
    assert by_name["date_DatetimeEncoder"]["additive"] is True
    assert by_name["num_StandardScaler"]["position"] == "post_encode"
    assert "additive" not in by_name["num_StandardScaler"]


def test_search_loop_searches_scope_dimension():
    df = make_toy_df()
    spec = {
        "scoped_encodings": [
            {
                "name": "color_enc",
                "cols": ["color"],
                "options": [skrub.GapEncoder(n_components=3, random_state=42)],
            }
        ],
        "model": {
            "HGB": __import__(
                "sklearn.ensemble", fromlist=["x"]
            ).HistGradientBoostingClassifier(random_state=42),
        },
    }
    result = run_search_loop(
        spec, df, "target", scoring="accuracy", budget_per_step=5
    )
    assert "scope_color_enc" in result["action_space"]
    assert result["best_score"] > 0.0
