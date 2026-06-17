"""Tests for the LLM-facing data digest (make_data_summary)."""

import os

import pandas as pd

from machine_learning_engineering.data_summary import infer_task_type, make_data_summary

TARGET = "median_house_value"


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


def test_summary_reports_shape_target_and_every_column():
    df = _california()
    s = make_data_summary(df, target=TARGET)
    assert TARGET in s
    assert "rows" in s and "columns" in s
    for col in df.columns:
        assert col in s
    assert "cardinality=" in s and "dtype=" in s and "missing=" in s


def test_summary_marks_target_and_infers_regression():
    df = _california()
    s = make_data_summary(df, target=TARGET)
    assert "(TARGET)" in s
    assert "regression" in s
    assert infer_task_type(df, TARGET) == "regression"


def test_summary_includes_head_rows():
    df = _california()
    s = make_data_summary(df, target=TARGET, n_head_rows=3)
    assert "First 3 rows" in s


def test_summary_shows_examples_for_categoricals():
    df = pd.DataFrame(
        {"city": ["NYC", "LA", "NYC", "SF"], "target": [1, 0, 1, 0]}
    )
    s = make_data_summary(df, target="target")
    assert "examples=" in s  # categorical column lists example values
    assert "'NYC'" in s
