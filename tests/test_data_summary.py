"""Tests for the LLM-facing data digest (make_data_summary)."""

import os

import pandas as pd

from machine_learning_engineering.data_summary import (
    infer_task_type,
    make_data_summary,
)


def _california() -> tuple[pd.DataFrame, str]:
    target = "median_house_value"
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "machine_learning_engineering",
        "tasks",
        "california-housing-prices",
        "train.csv",
    )
    return pd.read_csv(path), target


def test_summary_reports_shape_target_and_every_column():
    df, target = _california()
    s = make_data_summary(df, target=target)
    assert target in s
    assert "rows" in s and "columns" in s
    for col in df.columns:
        assert col in s
    assert "cardinality=" in s and "dtype=" in s and "missing=" in s


def test_summary_marks_target_and_infers_regression():
    df, target = _california()
    s = make_data_summary(df, target=target)
    assert "(TARGET)" in s
    assert "regression" in s
    assert infer_task_type(df, target) == "regression"


def test_summary_includes_head_rows():
    df, target = _california()
    s = make_data_summary(df, target=target, n_head_rows=3)
    assert "First 3 rows" in s


def test_summary_shows_examples_for_categoricals():
    df = pd.DataFrame(
        {"city": ["NYC", "LA", "NYC", "SF"], "target": [1, 0, 1, 0]}
    )
    s = make_data_summary(df, target="target")
    assert "examples=" in s  # categorical column lists example values
    assert "'NYC'" in s


def test_summary_reports_class_balance_for_classification_target():
    # a skewed binary target should surface its class proportions so the
    # analyst can flag imbalance to the planner
    df = pd.DataFrame({"x": range(100), "label": ["pos"] * 5 + ["neg"] * 95})
    s = make_data_summary(df, target="label")
    assert "class balance:" in s
    assert "'pos'=5.0%" in s and "'neg'=95.0%" in s


def test_summary_omits_class_balance_for_regression_target():
    df, target = _california()  # continuous float target
    s = make_data_summary(df, target=target)
    assert "class balance:" not in s
