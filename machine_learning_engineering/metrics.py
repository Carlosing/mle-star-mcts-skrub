"""Metric configuration — kept separate from the search engine on purpose.

Two distinct roles (see docs/mcts-uct.md for why they must not be conflated):

1. SEARCH-REWARD scorer — drives MCTS rollouts. Must be ONE consistent,
   bounded, higher-is-better metric across every config in the tree, or the Q/N
   values UCT compares become meaningless. Bounded (~[0,1]) also keeps rewards
   on the same scale as the exploration term (c=0.5).

2. TASK/REPORT metric — the competition metric (e.g. RMSE). Used only to score
   the final incumbent for reporting, never for search. Lower-is-better is fine;
   sklearn orients it (RMSE -> neg_root_mean_squared_error).

A model's internal training loss (e.g. a NN criterion) is a THIRD thing — a
searchable hyperparameter of the model, not a CV scorer — so it lives in the
model registry/spec, evaluated by the single search scorer above.
"""

# Role 1 — the MCTS search reward, per task type (bounded, higher-is-better).
SEARCH_SCORER = {
    "regression": "r2",
    "classification": "accuracy",
}

# Role 2 — task/report metric name -> sklearn scorer (higher-is-better form).
_REPORT_SCORER = {
    "root_mean_squared_error": "neg_root_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "mean_squared_error": "neg_mean_squared_error",
    "mse": "neg_mean_squared_error",
    "mean_absolute_error": "neg_mean_absolute_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "log_loss": "neg_log_loss",
    "f1": "f1",
}


def search_scorer(task_type: str) -> str:
    """The MCTS reward scorer for a task type (raises on unknown task type)."""
    try:
        return SEARCH_SCORER[task_type]
    except KeyError:
        raise ValueError(f"no search scorer configured for task_type={task_type!r}")


def report_scorer(metric_name: str) -> str | None:
    """sklearn scorer for the competition metric, or None if unrecognized."""
    return _REPORT_SCORER.get((metric_name or "").strip().lower())
