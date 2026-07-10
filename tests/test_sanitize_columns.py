"""Tests for the booster-safe column-name sanitizer (`_SanitizeColumns`).

skrub's encoders and AggJoiner emit column names carrying JSON special
characters (e.g. ``title: fee, reg``, from GapEncoder inside the
TableVectorizer). LightGBM rejects such feature names outright, so before the
sanitize step an injected booster failed to fit and every rollout silently
scored 0.0. These tests pin the fix: names are sanitized right before the
model, the step is invisible to the search (no new choice), and a booster
scores through a plan whose encoder produces dirty names.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

skrub = pytest.importorskip("skrub")

_BASE = os.path.join(
    os.path.dirname(__file__), "..", "machine_learning_engineering"
)
_spec = importlib.util.spec_from_file_location(
    "skrub_ops", os.path.join(_BASE, "skrub_ops.py")
)
skrub_ops = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = skrub_ops
_spec.loader.exec_module(skrub_ops)


# --- _safe_names (pure) -------------------------------------------------------


def test_safe_names_replaces_special_chars():
    assert skrub_ops._safe_names(["a: b, c", "x|y", "ok_1"]) == [
        "a__b__c",
        "x_y",
        "ok_1",
    ]


def test_safe_names_dedupes_collisions():
    assert skrub_ops._safe_names(["a|b", "a:b", "a_b"]) == [
        "a_b",
        "a_b_1",
        "a_b_2",
    ]


def test_safe_names_handles_empty_and_non_string():
    assert skrub_ops._safe_names(["", 7]) == ["col", "7"]


def test_sanitize_columns_transform_and_names_out():
    df = pd.DataFrame({"a: b": [1, 2], "a| b": [3, 4]})
    step = skrub_ops._SanitizeColumns().fit(df)
    out = step.transform(df)
    assert list(out.columns) == ["a__b", "a__b_1"]
    assert list(step.get_feature_names_out()) == ["a__b", "a__b_1"]
    # values pass through untouched; the input frame is not mutated
    assert out["a__b"].tolist() == [1, 2]
    assert list(df.columns) == ["a: b", "a| b"]


# --- integration: a booster fits through dirty encoder names -----------------


@pytest.fixture(scope="module")
def dirty_df():
    """Regression frame with a high-cardinality dirty-text column.

    The text column exceeds the TableVectorizer's high-cardinality threshold,
    so the GapEncoder encodes it and emits ``col: word, word`` feature names
    (the exact LightGBM-rejected shape). The signal lives in the numeric
    column, so the test isolates *fitting at all* from encoder quality.
    """
    rng = np.random.default_rng(0)
    n = 240
    words = ["fee", "reg", "tax", "levy", "duty", "toll", "fine", "rate"]
    text = [
        f"{words[i % 8]} {words[(i * 3) % 8]} #{i % 60}" for i in range(n)
    ]
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "desc": text,
            "x": x,
            "target": 3.0 * x + rng.normal(scale=0.3, size=n),
        }
    )


@pytest.fixture(scope="module")
def booster_plan(dirty_df):
    lightgbm = pytest.importorskip("lightgbm")

    return skrub_ops.build_staged_plan(
        {
            "encoder_options": [
                skrub.GapEncoder(n_components=3, random_state=0)
            ],
            "model": {
                "LGBM": lightgbm.LGBMRegressor(
                    n_estimators=30, n_jobs=1, random_state=0, verbose=-1
                )
            },
        },
        dirty_df,
    )


def test_sanitize_step_is_invisible_to_the_search(booster_plan):
    # the rename adds no choose_from: same action space, no gating entry
    assert set(skrub_ops.get_action_space(booster_plan)) == {
        "encoder",
        "model",
    }
    assert skrub_ops.get_choice_gating(booster_plan) == {}


def test_booster_scores_through_dirty_encoder_names(booster_plan, dirty_df):
    # before _SanitizeColumns, LightGBM raised on the GapEncoder names and
    # this rollout silently scored 0.0 (the exact live-run failure)
    rollout = skrub_ops.make_rollout_fn(booster_plan, dirty_df)
    assert rollout({"model": "LGBM"}) > 0.5
