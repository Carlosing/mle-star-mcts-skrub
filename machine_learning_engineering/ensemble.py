"""Top-k ensemble — a thin read-off over the persisted search, no new search.

This replaces MLE-STAR's iterative ensembler: the MCTS score cache already
ranks every distinct configuration evaluated, so the top-k incumbents come for
free. Each is fitted on a seeded train split of the full data and their
predictions on the held-out rows are averaged (regression) or soft-voted
(classification). No LLM, no extra rollouts — the only cost is k fits.
"""

import numpy as np
import pandas as pd
from sklearn import metrics as _sk_metrics

from machine_learning_engineering.skrub_ops import apply_state


def top_k_states(
    score_cache: dict, k: int = 3, defaults: dict | None = None
) -> list[dict]:
    """The k best *distinct pipelines* from the persisted score cache.

    Cache keys are `mcts.state_key` tuples, so states reconstruct exactly. Two
    keys can still name one pipeline: `apply_state` resets any omitted choice to
    its default, so {"model": "LGBM"} and {"model": "LGBM", "n_trees": <default>}
    fit identically. Passing `defaults` (from `skrub_ops.get_default_state`)
    fills omitted keys before de-duplicating, so k really is k distinct fits —
    otherwise the ensemble silently averages the same model k times and its
    score equals the incumbent's exactly.

    Example:
        top_k_states({(("model", "HGB"),): 0.9, (("model", "RF"),): 0.8}, k=1)
        # -> [{"model": "HGB"}]
    """
    ranked = sorted(score_cache.items(), key=lambda kv: -kv[1])
    picked: list[dict] = []
    seen: set[tuple] = set()
    for key, _ in ranked:
        state = dict(key)
        effective = {**(defaults or {}), **state}
        fingerprint = tuple(sorted(effective.items(), key=lambda kv: kv[0]))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        picked.append(state)
        if len(picked) == k:
            break
    return picked


# report-scorer name -> (metric fn, needs_proba, sign)
_METRIC_FNS = {
    "r2": (_sk_metrics.r2_score, False, 1.0),
    "neg_root_mean_squared_error": (
        _sk_metrics.root_mean_squared_error,
        False,
        -1.0,
    ),
    "neg_mean_squared_error": (_sk_metrics.mean_squared_error, False, -1.0),
    "neg_mean_absolute_error": (_sk_metrics.mean_absolute_error, False, -1.0),
    "accuracy": (_sk_metrics.accuracy_score, False, 1.0),
    "f1": (_sk_metrics.f1_score, False, 1.0),
    "roc_auc": (_sk_metrics.roc_auc_score, True, 1.0),
    "neg_log_loss": (_sk_metrics.log_loss, True, -1.0),
}


def _score(scoring: str, y_true, y_pred, proba):
    """Score predictions with a report-scorer name (higher-is-better sign).

    Example:
        _score("accuracy", [0, 1], [0, 1], None)  # -> 1.0
    """
    fn, needs_proba, sign = _METRIC_FNS[scoring]
    if needs_proba:
        value = fn(y_true, proba[:, 1] if proba.shape[1] == 2 else proba)
    else:
        value = fn(y_true, y_pred)
    return sign * float(value)


def evaluate_top_k(
    plan,
    states: list[dict],
    df: pd.DataFrame,
    target: str,
    task_type: str,
    scoring: str,
    aux: dict | None = None,
    main_var: str = "data",
    seed: int = 42,
    holdout_frac: float = 0.25,
) -> dict:
    """Fit the top-k configs and score the averaged/soft-voted ensemble.

    A seeded holdout split of the MAIN table serves as the comparison bench:
    each state is applied to the plan, a learner is fitted on the train rows
    (aux tables pass whole, as in CV), and predictions on the holdout are
    combined — mean for regression, mean predict_proba (argmax for label
    metrics) for classification. Returns the ensemble score next to each
    individual score so the lift over the single incumbent is explicit.

    Example:
        evaluate_top_k(plan, top_k_states(cache, 3), df, "target",
                       "classification", scoring="accuracy")
        # -> {"scorer": "accuracy", "ensemble_score": 0.91,
        #     "individual_scores": [0.89, 0.88, 0.85], "states": [...]}
    """
    if scoring not in _METRIC_FNS:
        raise ValueError(f"unsupported scoring for ensembling: {scoring!r}")
    if task_type == "classification":
        # stratified holdout: on an imbalanced target a random 25% draw can
        # land far from the class ratio (or miss a class), inflating variance
        holdout = df.groupby(df[target], group_keys=False).sample(
            frac=holdout_frac, random_state=seed
        )
    else:
        holdout = df.sample(frac=holdout_frac, random_state=seed)
    train = df.drop(index=holdout.index)
    train_env = {main_var: train, **(aux or {})}
    # the plan's X node drops the target itself, so the holdout keeps the
    # column (only fit mode ever evaluates the y mark)
    test_env = {main_var: holdout, **(aux or {})}
    y_true = holdout[target]

    predictions, probas, individual = [], [], []
    classes = None
    for state in states:
        apply_state(plan, state)
        learner = plan.skb.make_learner()
        learner.fit(train_env)
        pred = np.asarray(learner.predict(test_env))
        proba = None
        if task_type == "classification":
            try:
                proba = np.asarray(learner.predict_proba(test_env))
                classes = getattr(learner, "classes_", classes)
            except Exception:
                proba = None
        predictions.append(pred)
        probas.append(proba)
        individual.append(_score(scoring, y_true, pred, proba))

    if task_type == "classification" and all(p is not None for p in probas):
        mean_proba = np.mean(probas, axis=0)
        labels = np.unique(y_true) if classes is None else np.asarray(classes)
        ens_pred = labels[np.argmax(mean_proba, axis=1)]
        ens_score = _score(scoring, y_true, ens_pred, mean_proba)
    elif task_type == "classification":
        # no probabilities available -> majority vote over predicted labels
        stacked = np.stack(predictions)
        ens_pred = pd.DataFrame(stacked).mode(axis=0).iloc[0].to_numpy()
        ens_score = _score(scoring, y_true, ens_pred, None)
    else:
        ens_pred = np.mean(predictions, axis=0)
        ens_score = _score(scoring, y_true, ens_pred, None)

    return {
        "scorer": scoring,
        "k": len(states),
        "states": states,
        "individual_scores": individual,
        "ensemble_score": ens_score,
    }
