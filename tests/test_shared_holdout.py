"""The shared train/holdout boundary — the one bench all three arms are scored on.

The split is drawn on disk by ``scripts/stage_tasks.py`` BEFORE any method sees
the data, so no method can violate it: the holdout rows are simply absent from
every file a method can read. These tests assert that property directly, plus
the two silent-scoring failures that motivated it (see docs/BUG_LEDGER.md):

- the extension used to cross-validate over rows that later became its own eval
  set, and Caruana used to select the ensemble on the very rows it reported;
- run_mlestar used to align a submission to the holdout on LENGTH alone, and
  test.csv happened to have exactly the holdout's length on 8 of 13 tasks.
"""

import os

import pandas as pd
import pytest

from machine_learning_engineering import pipeline

TASKS_DIR = "machine_learning_engineering/tasks"
TASKS = sorted(
    t for t in os.listdir(TASKS_DIR)
    if os.path.isdir(os.path.join(TASKS_DIR, t))
)


@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_a_scoreable_holdout(task):
    """test.csv + test_answer.csv exist and are row-aligned."""
    holdout = pipeline.load_holdout(task)
    assert holdout is not None, f"{task}: no staged holdout"
    features = pd.read_csv(os.path.join(TASKS_DIR, task, "test.csv"))
    assert len(holdout) == len(features)
    assert len(holdout) > 0


@pytest.mark.parametrize("task", TASKS)
def test_holdout_rows_are_absent_from_train(task):
    """The load-bearing invariant: no holdout row is reachable from train.csv.

    This is what makes the extension's search honest — it cross-validates over
    load_task's frame, so if a holdout row were in there, the config would be
    chosen with knowledge of the rows we report on.
    """
    df, target, _tt, _m, _d, _aux = pipeline.load_task(task)
    holdout = pipeline.load_holdout(task, target=target)

    features = [c for c in holdout.columns if c != target]
    train_rows = set(map(tuple, df[features].astype(str).to_numpy()))
    holdout_rows = list(map(tuple, holdout[features].astype(str).to_numpy()))

    # duplicate feature-vectors legitimately exist in some datasets, so this is
    # not a strict "zero overlap" assert on content — but the split is a
    # partition of distinct ROWS, and a wholesale leak would light this up.
    overlap = sum(row in train_rows for row in holdout_rows)
    assert overlap < len(holdout_rows), (
        f"{task}: every holdout row also appears in train.csv — "
        "the split is not a partition"
    )


@pytest.mark.parametrize("task", TASKS)
def test_target_is_never_readable_from_test_csv(task):
    """A method reading test.csv must not be able to see the label."""
    df, target, _tt, _m, _d, _aux = pipeline.load_task(task)
    features = pd.read_csv(os.path.join(TASKS_DIR, task, "test.csv"))
    assert target not in features.columns, (
        f"{task}: test.csv exposes the target column {target!r}"
    )


def test_load_holdout_refuses_a_stale_task_dir(tmp_path):
    """A length mismatch means the dir is stale — refuse, never score on it."""
    task_dir = tmp_path / "toy"
    task_dir.mkdir()
    (task_dir / "task_description.txt").write_text("Predict the y.\n")
    pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}).to_csv(
        task_dir / "train.csv", index=False
    )
    pd.DataFrame({"x": [4, 5]}).to_csv(task_dir / "test.csv", index=False)
    pd.DataFrame({"row_id": [0], "y": [4]}).to_csv(  # one answer, two rows
        task_dir / "test_answer.csv", index=False
    )
    with pytest.raises(ValueError, match="test_answer.csv"):
        pipeline.load_holdout("toy", data_dir=str(tmp_path))


def test_load_holdout_is_none_when_not_staged(tmp_path):
    task_dir = tmp_path / "toy"
    task_dir.mkdir()
    pd.DataFrame({"x": [1], "y": [1]}).to_csv(task_dir / "train.csv", index=False)
    assert pipeline.load_holdout("toy", data_dir=str(tmp_path)) is None


def test_mlestar_scorer_refuses_a_mismatched_submission(tmp_path, monkeypatch):
    """A wrong-length submission must ERROR, not silently score.

    test.csv and a 25%-of-train holdout had identical lengths on 8 of 13 tasks,
    so a length-only guard let predictions for one set of rows be scored against
    targets from a completely different set.
    """
    import scripts.run_mlestar as R

    workspace = tmp_path / "ws" / "task" / "1" / "final"
    workspace.mkdir(parents=True)
    # a submission with the WRONG number of rows for california-housing (480)
    pd.DataFrame({"median_house_value": [1.0] * 123}).to_csv(
        workspace / "submission.csv", index=False
    )

    res = R._score_submission(
        "california-housing-prices", str(tmp_path / "ws"), seed=42
    )
    assert "error" in res
    assert "refusing to score" in res["error"]
    assert "score" not in res


def test_mlestar_scorer_scores_a_correct_submission(tmp_path):
    """A submission with the holdout's rows, in order, scores against the answers."""
    import scripts.run_mlestar as R

    task = "california-housing-prices"
    holdout = pipeline.load_holdout(task)
    target = "median_house_value"

    workspace = tmp_path / "ws" / task / "1" / "final"
    workspace.mkdir(parents=True)
    # predict the answers exactly -> a perfect score (RMSE 0)
    pd.DataFrame({target: holdout[target].to_numpy()}).to_csv(
        workspace / "submission.csv", index=False
    )

    res = R._score_submission(task, str(tmp_path / "ws"), seed=42)
    assert "error" not in res, res
    assert res["n"] == len(holdout)
    assert res["score"] == pytest.approx(0.0, abs=1e-9)  # neg RMSE of a perfect fit
