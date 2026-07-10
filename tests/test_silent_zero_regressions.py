"""Regression tests for the silent-0.0 rollout bug class.

Every historical live-run bug had the same shape: a failure inside skrub's CV
is swallowed by the rollout's try/except (or NaN-ed by the scorer) and the
config silently scores 0.0 — the search keeps running and quietly returns
garbage (proba-only classifiers on roc_auc, LightGBM vs special-character
feature names, unstratified imbalanced folds). These tests probe the *next*
members of that class, found by auditing the same layer boundaries:

- decision_function-only classifiers (LinearSVC, SGD-hinge): the roc_auc
  proba shim forced ``predict_proba``, regressing them to silent 0.0;
- XGBClassifier on string-labeled targets (open-payments, midwest-survey):
  xgboost >= 1.6 requires integer-encoded ``y``, so every XGB rollout scored
  0.0 — and a plan whose *default* model was XGB crashed skrub's eager
  preview at construction;
- NaN in the target column: every fit raises "Input y contains NaN", zeroing
  the whole search (classification only dodged it by accident);
- unbounded-below rewards: one hugely negative r2 poisons ancestor Q values,
  violating the documented [0,1] reward invariant;
- torch on the pipeline import chain (adk_agent -> run_logging ->
  common_util): torch loads a second OpenMP runtime, and any later xgboost
  fit through skrub CV segfaults the WHOLE process on macOS-ARM — uncatchable
  by the rollout try/except, worse than a silent 0.0.

Plus two pinning tests for behaviors that were *suspected* bugs but hold up
(singleton classes under StratifiedKFold; LightGBM's own string-label
handling) so a refactor can't silently break them either.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

skrub = pytest.importorskip("skrub")

from machine_learning_engineering import skrub_ops  # noqa: E402
from machine_learning_engineering.spec_resolver import resolve_spec  # noqa: E402


def _clf_df(n=300, seed=0, labels=("neg", "pos")):
    """Binary frame with real signal; string labels by default."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.asarray(labels)[(x > 0).astype(int)]
    return pd.DataFrame({"x": x, "x2": rng.normal(size=n), "target": y})


def _reg_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {"x": x, "target": 3.0 * x + rng.normal(scale=0.1, size=n)}
    )


# --- decision_function-only classifiers on proba-ranking scorers --------------


def test_resolve_scoring_falls_back_to_decision_function():
    class MarginOnly:
        # skrub's learner raises AttributeError for a method the final
        # estimator lacks — mimic that exactly (no predict_proba here)
        def decision_function(self, X):
            return np.asarray([-2.0, -1.0, 1.0, 2.0])

    scorer = skrub_ops._resolve_scoring("roc_auc")
    assert scorer(MarginOnly(), None, np.array([0, 0, 1, 1])) == 1.0


def test_decision_function_only_classifier_scores_on_roc_auc():
    from sklearn.svm import LinearSVC

    # before the fallback, every fold NaN-ed (AttributeError inside the
    # scorer) and LinearSVC silently scored 0.0 on any roc_auc task
    df = _clf_df(labels=(0, 1))
    plan = skrub_ops.build_staged_plan(
        {"model": {"SVC": LinearSVC(random_state=0)}}, df
    )
    rollout = skrub_ops.make_rollout_fn(
        plan, df, scoring="roc_auc", stratify=True, target="target"
    )
    assert rollout({"model": "SVC"}) > 0.8


# --- XGBClassifier on string-labeled targets ----------------------------------


def test_resolved_xgb_fits_string_labels_as_only_model():
    pytest.importorskip("xgboost")
    df = _clf_df()
    # XGB as the ONLY (= default) model: this exact spec crashed
    # build_staged_plan outright before the shim (skrub eagerly previews the
    # fit, and xgboost raises "Invalid classes inferred from ... `y`")
    spec = resolve_spec(
        {
            "model": [
                {
                    "name": "xgboost.XGBClassifier",
                    "params": {"n_estimators": {"int": [10, 30]}},
                }
            ]
        },
        task_type="classification",
    )
    assert list(spec["model"]) == ["XGBClassifier"]  # label unchanged
    plan = skrub_ops.build_staged_plan(spec, df)
    rollout = skrub_ops.make_rollout_fn(
        plan, df, scoring="roc_auc", stratify=True, target="target"
    )
    # roc_auc also checks predict_proba column order against the labels
    assert rollout({"model": "XGBClassifier"}) > 0.8


def test_xgb_shim_roundtrips_original_labels():
    pytest.importorskip("xgboost")
    from machine_learning_engineering.spec_resolver import (
        _xgb_classifier_shim,
    )

    df = _clf_df(n=120)
    est = _xgb_classifier_shim()(n_estimators=10, n_jobs=1, verbosity=0)
    est.fit(df[["x", "x2"]], df["target"])
    assert list(est.classes_) == ["neg", "pos"]  # sorted, sklearn convention
    assert set(est.predict(df[["x", "x2"]])) <= {"neg", "pos"}


# --- NaN in the target column ---------------------------------------------


def test_rollout_survives_nan_target_rows():
    from sklearn.ensemble import HistGradientBoostingRegressor

    clean = _reg_df()
    dirty = clean.copy()
    dirty.loc[dirty.index[:5], "target"] = np.nan
    plan = skrub_ops.build_staged_plan(
        {"model": {"HGB": HistGradientBoostingRegressor(random_state=0)}},
        clean,
    )
    # the rollout gets the frame WITH unlabeled rows — before the drop, every
    # subsample fit raised "Input y contains NaN" and all configs scored 0.0
    rollout = skrub_ops.make_rollout_fn(
        plan, dirty, scoring="r2", target="target"
    )
    assert rollout({"model": "HGB"}) > 0.8


def test_load_task_drops_unlabeled_rows(tmp_path):
    pipeline = pytest.importorskip("machine_learning_engineering.pipeline")

    task_dir = tmp_path / "nan-task"
    task_dir.mkdir()
    df = _reg_df(n=50)
    df.loc[df.index[:3], "target"] = np.nan
    df.to_csv(task_dir / "train.csv", index=False)
    (task_dir / "task_description.txt").write_text(
        "Predict the target.\n\n# Metric\nrmse\n"
    )
    loaded, target, *_ = pipeline.load_task(
        "nan-task", data_dir=str(tmp_path)
    )
    assert target == "target"
    assert not loaded["target"].isna().any()
    assert len(loaded) == 47


# --- reward bounds --------------------------------------------------------


def test_bounded_reward_is_monotone_and_never_reaches_zero():
    # r2 is unbounded BELOW: one catastrophic config used to drag every
    # ancestor's Q with it. A clamp bounds it but flattens the whole negative
    # half, so a task where every config scores r2 < 0 leaves UCT choosing at
    # random. The squash bounds AND preserves order.
    raw = [-1e6, -50.0, -1.0, -0.3, 0.0, 0.5, 1.0]
    rewards = [skrub_ops._bounded_reward(s, "r2") for s in raw]
    assert rewards == sorted(rewards), "squash must be strictly increasing"
    assert all(0.0 < r <= 1.0 for r in rewards)
    assert rewards[-1] == 1.0  # r2 == 1 is the maximum
    assert skrub_ops._bounded_reward(0.0, "r2") == 0.5  # predicts the mean
    # 0.0 is reserved for FAILURES: no real score maps onto it
    assert min(rewards) > 0.0


def test_bounded_reward_passes_unit_scorers_through_unchanged():
    # classification baselines must not move
    for scorer in ("accuracy", "roc_auc", "f1"):
        assert skrub_ops._bounded_reward(0.664, scorer) == 0.664
    assert skrub_ops.reward_scale("accuracy") == "raw"
    assert skrub_ops.reward_scale("r2") == "1/(2 - r2)"


def test_negative_r2_config_ranks_above_a_failed_one():
    from sklearn.linear_model import LinearRegression

    # anti-signal target: the linear fit is worse than the fold mean (r2 < 0)
    df = pd.DataFrame({"x": np.arange(300, dtype=float)})
    df["target"] = np.where(df.x % 2 == 0, 1000.0, -1000.0) * (df.x + 1)
    plan = skrub_ops.build_staged_plan(
        {"model": {"LR": LinearRegression()}}, df
    )
    rollout = skrub_ops.make_rollout_fn(
        plan, df, scoring="r2", target="target"
    )
    bad = rollout({"model": "LR"})
    failed = rollout({"model": "DoesNotExist"})
    assert failed == 0.0
    # a bad-but-working config must still be searchable, i.e. beat a failure
    assert 0.0 < bad < 0.5


def test_all_sub_baseline_landscape_stays_ordered():
    """A hard task (every config r2 < 0) must not collapse to a flat landscape."""
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression

    # noise target: nothing predicts it, so every model lands below the mean
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"x": rng.normal(size=200), "target": rng.normal(size=200)}
    )
    plan = skrub_ops.build_staged_plan(
        {
            "model": {
                "LR": LinearRegression(),
                # always predicts a constant far from the mean -> much worse r2
                "Bad": DummyRegressor(strategy="constant", constant=50.0),
            }
        },
        df,
    )
    rollout = skrub_ops.make_rollout_fn(
        plan, df, scoring="r2", target="target"
    )
    lr, bad = rollout({"model": "LR"}), rollout({"model": "Bad"})
    # under the old [0,1] clamp BOTH were exactly 0.0 and UCT was blind here
    assert 0.0 < bad < lr < 0.5



# --- AggJoiner "mode" leaves an array in the cell on a tie -------------------


def test_aggjoiner_mode_really_does_emit_array_cells():
    """Pin the upstream behavior the fix exists for (so we notice if skrub fixes it)."""
    main = pd.DataFrame({"k": ["a", "b", "c"]})
    # 'b' has a modal TIE (Y and Z once each); 'c' has no aux rows at all
    aux = pd.DataFrame({"k": ["a", "a", "b", "b"], "city": ["X", "X", "Y", "Z"]})
    joined = skrub.AggJoiner(
        aux, "mode", main_key="k", aux_key="k", cols=["city"]
    ).fit_transform(main)
    cells = list(joined["city_mode"])
    assert cells[0] == "X"
    assert skrub_ops._is_array_cell(cells[1])  # the tie -> ["Y", "Z"]


def test_scalarize_aggregates_collapses_ties_and_leaves_scalars():
    df = pd.DataFrame({"a": ["X", np.array(["Y", "Z"]), np.nan], "n": [1, 2, 3]})
    out = skrub_ops._ScalarizeAggregates().fit(df).transform(df)
    assert out["a"].tolist()[:2] == ["X", "Y"]  # first element, deterministic
    assert out["n"].tolist() == [1, 2, 3]  # untouched
    # a frame with no array cells is returned unchanged (no needless copy)
    clean = pd.DataFrame({"a": ["X", "Y"]})
    assert skrub_ops._ScalarizeAggregates().fit(clean).transform(clean) is clean


def test_mode_assemble_option_scores_instead_of_silently_zeroing():
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    n = 240
    keys = [f"k{i % 40}" for i in range(n)]
    # aux rows chosen so many keys have a modal TIE on 'city' -- before
    # _ScalarizeAggregates the TableVectorizer died on the array cell and the
    # whole assemble option scored 0.0 (credit-fraud's basket_mode did this)
    aux = pd.DataFrame(
        {
            "k": [f"k{i % 40}" for i in range(80)],
            "city": ["Y" if i < 40 else "Z" for i in range(80)],
            "v": rng.normal(size=80),
        }
    )
    signal = np.asarray([int(k[1:]) for k in keys], dtype=float)
    main = pd.DataFrame(
        {"k": keys, "target": signal + rng.normal(scale=0.5, size=n)}
    )
    plan = skrub_ops.build_staged_plan(
        {
            "assemble": [
                {
                    "name": "city_mode",
                    "table": "aux",
                    "key": "k",
                    "operations": ["mode"],
                    "cols": ["city"],
                }
            ],
            "model": {"RF": RandomForestRegressor(n_estimators=20, random_state=0)},
        },
        main,
        target="target",
        aux_tables={"aux": aux},
    )
    rollout = skrub_ops.make_rollout_fn(
        plan, main, aux={"aux": aux}, main_var="data", scoring="r2",
        target="target",
    )
    assert rollout({"assemble": "city_mode", "model": "RF"}) > 0.5


# --- task type must agree with the declared metric ---------------------------


def test_declared_metric_overrules_cardinality_heuristic():
    from machine_learning_engineering.data_summary import infer_task_type

    # a 0.5-5.0 half-star rating has 10 distinct values, so the `nunique > 20`
    # rule calls it classification — the run then searches `accuracy` over
    # Classifiers while reporting RMSE. The declared metric must win.
    ratings = pd.DataFrame({"rating": [0.5, 1.0, 1.5, 2.0, 5.0] * 20})
    assert infer_task_type(ratings, "rating") == "classification"
    assert infer_task_type(ratings, "rating", "rmse") == "regression"
    assert infer_task_type(ratings, "rating", "r2") == "regression"
    # ... but only where the dtype allows it: string targets stay classification
    labels = pd.DataFrame({"y": ["a", "b"] * 20})
    assert infer_task_type(labels, "y", "rmse") == "classification"
    # an unrecognized metric falls back to the heuristic
    assert infer_task_type(ratings, "rating", "bogus") == "classification"


def test_search_and_report_scorers_agree_on_task_type():
    from machine_learning_engineering import metrics

    # the mismatch that made this a bug: search on `accuracy`, report RMSE
    for metric in ("rmse", "root_mean_squared_error", "r2"):
        assert metrics.metric_task_type(metric) == "regression"
        assert metrics.search_scorer("regression", metric) in {"r2"}
    for metric in ("accuracy", "roc_auc", "f1"):
        assert metrics.metric_task_type(metric) == "classification"


# --- torch must stay off the pipeline import chain ---------------------------


def test_pipeline_import_does_not_load_torch():
    import subprocess

    # torch's second OpenMP runtime segfaults any later xgboost fit inside
    # skrub CV (macOS-ARM, uncatchable) — the crash needs torch imported
    # FIRST, so it only reproduces in-process order-dependently; pin the
    # root cause instead: importing the whole agent->search stack must not
    # pull torch in (it reached the path via run_logging -> common_util)
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "from machine_learning_engineering import pipeline\n"
        "assert 'torch' not in sys.modules, 'torch leaked onto the pipeline import chain'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


# --- pinning: suspected members of the class that hold up -------------------


def test_singleton_class_does_not_zero_rollouts():
    from sklearn.ensemble import HistGradientBoostingClassifier

    # a class with ONE row (anomaly-flavored dirty data): StratifiedKFold
    # only warns and keeps the fold, so rewards must stay real — pin it
    df = _clf_df(labels=(0, 1))
    df.loc[df.index[-1], "target"] = 99
    plan = skrub_ops.build_staged_plan(
        {"model": {"HGB": HistGradientBoostingClassifier(random_state=0)}},
        df,
    )
    rollout = skrub_ops.make_rollout_fn(
        plan, df, scoring="accuracy", stratify=True, target="target"
    )
    assert rollout({"model": "HGB"}) > 0.8


def test_lgbm_accepts_string_labels():
    lightgbm = pytest.importorskip("lightgbm")

    # LightGBM does its own label encoding — pin that string-target tasks
    # keep working if that ever changes (xgboost already dropped theirs)
    df = _clf_df()
    plan = skrub_ops.build_staged_plan(
        {
            "model": {
                "LGBM": lightgbm.LGBMClassifier(
                    n_estimators=20, n_jobs=1, random_state=0, verbose=-1
                )
            }
        },
        df,
    )
    rollout = skrub_ops.make_rollout_fn(
        plan, df, scoring="roc_auc", stratify=True, target="target"
    )
    assert rollout({"model": "LGBM"}) > 0.8
