"""Relational (multi-table) path — task loading, digest, spec validation, e2e.

Covers the wiring that turns the unit-tested assemble engine into a full path:
`load_task` discovers `aux_*.csv` tables, the data digest describes them (with
join-key candidates) to the analyst, `resolve_spec` validates LLM assemble
configs against the real schemas, and `run_pipeline` (agents mocked) searches a
plan whose action space includes the assemble stage. Fully offline.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("skrub")

import numpy as np
import pandas as pd

from machine_learning_engineering import pipeline, skrub_ops, spec_resolver
from machine_learning_engineering.data_summary import make_data_summary
from machine_learning_engineering.search_loop import run_search_loop

from fixtures.staged_plan import make_relational_data, relational_spec
from test_pipeline import FakeLlm


# --- task-dir fixture ---------------------------------------------------------


@pytest.fixture(scope="module")
def rel_task_dir(tmp_path_factory):
    """A multi-table task directory: train.csv + aux_aux.csv + description."""
    main, aux = make_relational_data()
    task_dir = tmp_path_factory.mktemp("tasks") / "rel-task"
    task_dir.mkdir()
    main.to_csv(task_dir / "train.csv", index=False)
    aux["aux"].to_csv(task_dir / "aux_aux.csv", index=False)
    (task_dir / "task_description.txt").write_text(
        "# Task\n\nPredict the target of each row.\n\n# Metric\n\naccuracy\n"
    )
    return task_dir


# --- load_task discovers auxiliary tables --------------------------------------


def test_load_task_discovers_aux_tables(rel_task_dir):
    df, target, task_type, metric, _, aux_tables = pipeline.load_task(
        "rel-task", data_dir=str(rel_task_dir.parent)
    )
    assert target == "target"
    assert task_type == "classification"
    assert metric == "accuracy"
    assert set(aux_tables) == {"aux"}
    assert list(aux_tables["aux"].columns) == ["id", "v"]


# --- the digest describes aux tables + join keys -------------------------------


def test_data_summary_includes_aux_tables_and_join_keys():
    main, aux = make_relational_data()
    summary = make_data_summary(main, "target", aux_tables=aux)
    assert "Auxiliary table 'aux'" in summary
    assert "Join-key candidates" in summary
    assert "id <-> id" in summary


# --- resolve_spec validates assemble configs -----------------------------------

_AUX_SCHEMAS = {"aux": ["id", "v"]}
_MAIN_COLS = ["id", "noise"]


def _resolve(entries):
    return spec_resolver._resolve_assemble(entries, _AUX_SCHEMAS, _MAIN_COLS)


def test_assemble_validation_keeps_a_good_entry():
    kept = _resolve(
        [{"name": "aux_mean", "table": "aux", "key": "id",
          "operations": ["mean", "bogus"], "cols": ["v", "nope"]}]
    )
    assert kept == [
        {"name": "aux_mean", "table": "aux", "operations": ["mean"],
         "key": "id", "cols": ["v"]}
    ]


def test_assemble_validation_drops_bad_entries():
    assert _resolve([{"table": "nope", "key": "id", "operations": ["mean"]}]) == []
    assert _resolve([{"table": "aux", "key": "missing", "operations": ["mean"]}]) == []
    assert _resolve([{"table": "aux", "key": "id", "operations": ["bogus"]}]) == []
    assert _resolve([{"table": "aux", "operations": ["mean"]}]) == []  # no key
    # single-table run (no aux schemas): the whole stage is dropped
    assert spec_resolver._resolve_assemble(
        [{"table": "aux", "key": "id", "operations": ["mean"]}], None, _MAIN_COLS
    ) == []


def test_assemble_validation_supports_main_aux_key_pair():
    kept = _resolve(
        [{"table": "aux", "main_key": "id", "aux_key": "id", "operations": ["max"]}]
    )
    assert kept[0]["main_key"] == "id" and kept[0]["aux_key"] == "id"
    assert "key" not in kept[0]


def test_resolve_spec_validates_assemble_and_names_unnamed_entries():
    raw = {
        "assemble": [
            {"table": "aux", "key": "id", "operations": ["mean"], "cols": ["v"]},
            {"table": "hallucinated", "key": "id", "operations": ["mean"]},
        ],
        "model": ["sklearn.ensemble.HistGradientBoostingClassifier"],
    }
    spec = spec_resolver.resolve_spec(
        raw, task_type="classification", aux_schemas=_AUX_SCHEMAS, main_columns=_MAIN_COLS
    )
    assert [e["name"] for e in spec["assemble"]] == ["aux_mean"]


# --- search loop threads aux tables through -------------------------------------


def test_run_search_loop_with_aux_tables():
    main, aux = make_relational_data()
    result = run_search_loop(
        relational_spec(), main, "target",
        scoring="accuracy", budget_per_step=6, aux_tables=aux,
    )
    assert "assemble" in result["action_space"]
    assert isinstance(result["best_score"], float)
    assert result["best_score"] > 0.0  # relational rollouts really ran


# --- search reward follows a bounded task metric (imbalanced credit-fraud) -----


def test_search_scorer_prefers_bounded_report_metric():
    from machine_learning_engineering import metrics

    assert metrics.search_scorer("classification", "roc_auc") == "roc_auc"
    assert metrics.search_scorer("classification", "accuracy") == "accuracy"
    # unbounded report metrics stay report-only; search keeps the default
    assert metrics.search_scorer("regression", "root_mean_squared_error") == "r2"
    assert metrics.search_scorer("classification") == "accuracy"


# --- stratified CV for imbalanced classification (credit-fraud's 1% positive) --


def test_cv_kwarg_stratifies_only_when_requested():
    assert skrub_ops._cv_kwarg(False, seed=42) == {}
    cv_kwargs = skrub_ops._cv_kwarg(True, seed=42)
    from sklearn.model_selection import StratifiedKFold

    assert isinstance(cv_kwargs["cv"], StratifiedKFold)


def _imbalanced_df(n=200, pos_rate=0.03, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < pos_rate).astype(int)
    y[:2] = 1  # guarantee at least 2 positives regardless of the draw
    return pd.DataFrame({"x": x, "target": y})


def test_stratified_rollout_avoids_nan_on_rare_target():
    """skrub's cross_validate defaults to plain KFold (no y/classifier passed
    to check_cv), so a random fold of a rare-target subsample can land on zero
    positives; sklearn's scorer then silently NaNs that fold. stratify=True
    must keep every rollout a finite score in [0, 1]."""
    df = _imbalanced_df()
    spec = {"model": {"LogReg": __import__(
        "sklearn.linear_model", fromlist=["x"]
    ).LogisticRegression(max_iter=500)}}
    plan = skrub_ops.build_staged_plan(spec, df)
    rollout = skrub_ops.make_rollout_fn(
        plan, df, min_rows=50, scoring="roc_auc", stratify=True
    )
    score = rollout({"model": "LogReg"})
    assert 0.0 <= score <= 1.0 and score == score  # finite, never NaN-derived


# --- end-to-end (agents mocked) --------------------------------------------------

_REL_ANALYSIS = """\
Task: binary classification, target 'target'. Main table has an uninformative
'noise' column and an 'id' key; the auxiliary table 'aux' holds per-id values
'v'. The signal is in per-id aggregates of aux.v — an aggregate join
(id <-> id) is the key opportunity. Models: HistGradientBoosting, LogisticRegression.
"""

_REL_SPEC = json.dumps({
    "assemble": [
        {"name": "aux_mean", "table": "aux", "key": "id",
         "operations": ["mean"], "cols": ["v"]},
        {"name": "aux_stats", "table": "bogus_table", "key": "id",
         "operations": ["mean"]},
    ],
    "model": [
        "sklearn.ensemble.HistGradientBoostingClassifier",
        "sklearn.linear_model.LogisticRegression",
    ],
})


def test_run_pipeline_relational_end_to_end_offline(rel_task_dir):
    model = FakeLlm().set_responses([_REL_ANALYSIS, _REL_SPEC])
    result = pipeline.run_pipeline(
        task_name="rel-task",
        data_dir=str(rel_task_dir.parent),
        budget=8,
        model=model,
        with_search=False,
    )
    assert not result["used_fallback_spec"]
    # the hallucinated table was dropped; the valid AggJoiner is searchable
    assert result["action_space"]["assemble"] == ["skip", "aux_mean"]
    assert "Auxiliary table 'aux'" in result["data_summary"]
    assert isinstance(result["best_search_score"], float)
    assert result["report"] is not None  # accuracy on the incumbent, aux passed
