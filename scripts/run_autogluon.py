"""AutoGluon baseline for the time-budgeted benchmark.

A guaranteed, well-known AutoML baseline to compare the MCTS-skrub extension
against under a fixed wall-clock budget. It reuses the project's own task
loading, the SHARED seeded holdout split, and the SAME report scorer, so the
number it produces is directly comparable to the extension's ``holdout`` score
(and MLE-STAR's).

    load_task -> ensemble.holdout_split -> TabularPredictor.fit(time_limit)
      -> predict on the shared holdout -> ensemble._score -> uniform result.json

Key honesty notes:
- **Flat-table only.** AutoGluon gets the main ``train.csv`` and never the
  ``aux_*.csv`` relational tables, so ``relational: false``. On credit-fraud the
  extension's ability to join aux is exactly the differentiator to report.
- **Zero LLM cost** (``llm_calls=0``, all-zero ``tokens``) — it is not an
  LLM method; the token axis simply shows it at the origin.

Run:
    uv run python scripts/run_autogluon.py --task california-housing-prices \
        --time-budget-s 3600
    uv run python scripts/run_autogluon.py --task credit-fraud --time-budget-s 60

Needs the optional dep group:  uv sync --extra autogluon
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np

from machine_learning_engineering import ensemble, metrics, pipeline


# our metric name -> AutoGluon eval_metric (what AG optimizes internally). The
# EXTERNAL score we report is always computed by ensemble._score, so this only
# steers AutoGluon's own model selection.
_AG_EVAL_METRIC = {
    "root_mean_squared_error": "root_mean_squared_error",
    "rmse": "root_mean_squared_error",
    "mean_squared_error": "mean_squared_error",
    "mse": "mean_squared_error",
    "mean_absolute_error": "mean_absolute_error",
    "mae": "mean_absolute_error",
    "r2": "r2",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "log_loss": "log_loss",
}


def _problem_type(task_type: str, y) -> str:
    """AutoGluon problem_type from our task type + target cardinality."""
    if task_type == "regression":
        return "regression"
    return "binary" if int(y.nunique()) == 2 else "multiclass"


def run_autogluon(
    task: str,
    out_dir: str,
    time_budget_s: float = 3600.0,
    seed: int = 42,
    num_cpus: int = 1,
    presets: str = "best_quality",
) -> dict:
    """Fit AutoGluon under a time limit and score it on the shared holdout.

    ``num_cpus=1`` (default) pins every model to a single thread. This is
    REQUIRED on macOS-ARM: LightGBM/XGBoost bundle their own OpenMP runtime and
    a multi-threaded fit alongside torch's libomp segfaults uncatchably (the
    same double-libomp issue the extension dodges by pinning booster
    ``n_jobs=1`` in its REGISTRY — so single-thread boosters is a *consistent*
    constraint across both methods here, not a handicap on one). On a host
    without the conflict (most Linux) raise it to use AutoGluon's full
    parallelism.

    ``presets="best_quality"`` (default) turns on bagging + multi-layer
    stacking, which is what actually makes AutoGluon spend the given
    ``time_limit`` — the bare default preset (``medium_quality``) fits its
    model zoo once each and stops in a few seconds regardless of the budget,
    so a time-scaling comparison against it would be a flat line. Bagging
    still respects ``time_limit`` as a soft ceiling, not a target: on a small
    table (few hundred rows) AutoGluon may still finish well under budget once
    it has bagged/stacked everything worth trying — that's expected, not a
    bug, and mirrors our own rollouts' wall-clock variance.
    """
    from autogluon.tabular import TabularPredictor  # lazy: optional dep

    df, target, task_type, metric, desc, _aux = pipeline.load_task(task)
    scorer = metrics.report_scorer(metric)
    if scorer is None or scorer not in ensemble._METRIC_FNS:
        raise ValueError(f"no comparable report scorer for metric {metric!r}")

    # the SHARED bench: identical rows to the extension's holdout score
    train, holdout = ensemble.holdout_split(df, target, task_type, seed=seed)
    y_true = holdout[target]
    features = holdout.drop(columns=[target])

    base = {
        "method": "autogluon",
        "task": task,
        "task_type": task_type,
        "metric": metric,
        "target": target,
        "time_budget_s": time_budget_s,
        "llm_calls": 0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
        "tokens_by_agent": {},
        "relational": False,  # flat-table only; no aux join
        "presets": presets,
    }

    wall_start = time.perf_counter()
    try:
        predictor = TabularPredictor(
            label=target,
            problem_type=_problem_type(task_type, df[target]),
            eval_metric=_AG_EVAL_METRIC.get(metric),
            path=os.path.join(out_dir, "ag_models"),
            verbosity=1,
        ).fit(
            train,
            time_limit=time_budget_s,
            presets=presets,
            ag_args_fit={"num_cpus": num_cpus},  # single-thread boosters (libomp)
        )

        pred = np.asarray(predictor.predict(features))
        proba = None
        if task_type == "classification":
            # order proba columns by sorted class labels so column 1 is the
            # positive class — matches ensemble._score's proba[:, 1] convention.
            classes = sorted(predictor.class_labels)
            proba = predictor.predict_proba(features)[classes].to_numpy()
        score = ensemble._score(scorer, y_true, pred, proba)
    except Exception as exc:
        # a failed baseline (e.g. no flat signal on a relational task, or the
        # time limit too short) is a recorded data point, not a crash.
        return {
            **base,
            "status": "failed",
            "error": repr(exc)[:300],
            "wall_clock_s": round(time.perf_counter() - wall_start, 1),
            "holdout": None,
        }

    return {
        **base,
        "status": "ok",
        "wall_clock_s": round(time.perf_counter() - wall_start, 1),
        "holdout": {"scorer": scorer, "score": score},
        "leaderboard": predictor.leaderboard(silent=True).to_dict("records"),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="AutoGluon benchmark baseline.")
    parser.add_argument("--task", required=True, help="staged task name")
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=3600.0,
        help="AutoGluon fit time_limit in seconds (default 3600 = 1h)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=1,
        help="threads per model (default 1: REQUIRED on macOS-ARM to avoid the "
        "LightGBM/XGBoost double-libomp segfault; raise on Linux)",
    )
    parser.add_argument(
        "--presets",
        default="best_quality",
        help="AutoGluon presets (default best_quality: enables bagging + "
        "stacking so the fit actually spends time_limit; medium_quality "
        "fits its model zoo once each and returns in seconds regardless "
        "of budget)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="artifact dir (default: runs/autogluon_<task>_<ts>)",
    )
    args = parser.parse_args()

    out_dir = args.out or os.path.join(
        "runs", f"autogluon_{args.task}_{datetime.now():%Y%m%d-%H%M}"
    )
    os.makedirs(out_dir, exist_ok=True)
    result = run_autogluon(
        args.task,
        out_dir,
        time_budget_s=args.time_budget_s,
        seed=args.seed,
        num_cpus=args.num_cpus,
        presets=args.presets,
    )
    import json

    with open(
        os.path.join(out_dir, "result.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nAutoGluon | task: {result['task']} ({result['task_type']})")
    print(f"time budget: {result['time_budget_s']}s  |  "
          f"wall clock: {result['wall_clock_s']}s  |  "
          f"status: {result.get('status')}")
    hold = result.get("holdout")
    if hold:
        print(f"holdout {hold['scorer']}: {hold['score']:.4f}")
    else:
        print(f"FAILED: {result.get('error')}")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    _main()
