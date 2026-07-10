"""Every task directory under `tasks/` must be loadable and self-consistent.

`scripts/stage_tasks.py` writes these from the cached `data/` datasets. The
failure modes are quiet ones — a target the description parser can't find (so
`load_task` silently falls back to the *last column*), a metric that disagrees
with the inferred task type (so the search optimizes `accuracy` and reports
RMSE), a leaked column that makes the task trivial, or an auxiliary table whose
join key shares no values with the main table (so every AggJoiner scores 0.0).
None of these raise; they just make the run meaningless. So assert them.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("skrub")

from machine_learning_engineering import metrics, pipeline  # noqa: E402

TASKS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "machine_learning_engineering", "tasks"
)
TASKS = sorted(
    name
    for name in os.listdir(TASKS_DIR)
    if os.path.isdir(os.path.join(TASKS_DIR, name))
)


def test_task_dir_is_not_empty():
    assert TASKS, "no staged tasks found"


@pytest.fixture(scope="module", params=TASKS)
def task(request):
    df, target, task_type, metric, desc, aux = pipeline.load_task(request.param)
    return request.param, df, target, task_type, metric, desc, aux


def test_target_is_parsed_not_guessed(task):
    name, df, target, *_, desc, _ = task
    # `_parse_target` falls back to the LAST column when its regex misses — a
    # silent way to train on the wrong column. Assert the description names it.
    assert pipeline._parse_target(desc, df) == target, (
        f"{name}: description does not name the target {target!r}"
    )
    assert target in df.columns


def test_metric_is_recognized_and_matches_task_type(task):
    name, _, _, task_type, metric, _, _ = task
    assert metrics.report_scorer(metric), f"{name}: unknown metric {metric!r}"
    declared = metrics.metric_task_type(metric)
    assert declared == task_type, (
        f"{name}: metric {metric!r} implies {declared}, task type is {task_type}"
    )


def test_search_scorer_exists_for_task(task):
    name, _, _, task_type, metric, _, _ = task
    assert metrics.search_scorer(task_type, metric)


def test_target_has_no_missing_values_after_load(task):
    # load_task drops unlabeled rows; nothing downstream tolerates NaN targets
    name, df, target, *_ = task
    assert not df[target].isna().any(), f"{name}: NaN targets survived load_task"


def test_no_single_feature_perfectly_predicts_the_target(task):
    """Catch leakage: a lone numeric feature with |r| > 0.999 to the target."""
    name, df, target, task_type, *_ = task
    if task_type != "regression":
        return
    numeric = df.select_dtypes("number").drop(columns=[target], errors="ignore")
    if numeric.empty:
        return
    corr = numeric.corrwith(df[target]).abs()
    worst = corr.idxmax() if len(corr) else None
    assert corr.max(skipna=True) < 0.999 or pd.isna(corr.max()), (
        f"{name}: column {worst!r} is a near-perfect proxy for {target!r} "
        f"(|r|={corr.max():.4f}) — leakage"
    )


def test_test_csv_matches_train_minus_target(task):
    name, df, target, *_ = task
    path = os.path.join(TASKS_DIR, name, "test.csv")
    if not os.path.exists(path):
        return  # relational tasks staged before this convention
    test = pd.read_csv(path)
    assert target not in test.columns, f"{name}: test.csv leaks the target"
    assert set(test.columns) == set(df.columns) - {target}


def test_aux_join_keys_overlap_the_main_table(task):
    """An aux table whose key shares no values with main silently joins to NaN."""
    name, df, target, _, _, _, aux = task
    if not aux:
        return
    for aux_name, adf in aux.items():
        overlaps = [
            len(set(adf[ac].dropna()) & set(df[mc].dropna()))
            for ac in adf.columns
            for mc in df.columns
            if mc != target
        ]
        assert overlaps and max(overlaps) > 0, (
            f"{name}: aux table {aux_name!r} shares no key values with the "
            f"main table — every AggJoiner would produce all-NaN columns"
        )
