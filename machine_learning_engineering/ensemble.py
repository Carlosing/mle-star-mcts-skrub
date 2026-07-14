"""Top-k ensemble — a thin read-off over the persisted search, no new search.

This replaces MLE-STAR's iterative ensembler: the MCTS score cache already
ranks every distinct configuration evaluated, so the top-k incumbents come for
free. Each is fitted on a seeded train split of the full data and their
predictions on the held-out rows are averaged (regression) or soft-voted
(classification). No LLM, no extra rollouts — the only cost is k fits.
"""

from collections import Counter

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


def _caruana_select(n_members: int, score_indices, size: int) -> list[int]:
    """Caruana greedy ensemble selection (with replacement + early stop).

    ``score_indices(indices) -> float`` scores the ensemble formed by averaging
    the members at ``indices`` (repeats = weight), higher-is-better. Seeds with
    the single best member (so the ensemble is never worse than the best model
    on this holdout), then for up to ``size-1`` rounds adds the member whose
    inclusion most improves the score, STOPPING as soon as no member gives a
    positive gain. Selection with replacement means a strong member can repeat
    (implicit weighting); the early stop means a pool of near-duplicates
    collapses to a single member (no spurious lift) — which is exactly what
    makes it robust when the search returns very similar configs.

    Returns the selected member indices WITH repetition, e.g. ``[2, 0, 2]``.

    Reference: Caruana et al. 2004, "Ensemble Selection from Libraries of
    Models" — the method behind auto-sklearn's / AutoGluon's post-hoc ensembler.

    Example:
        _caruana_select(3, score_fn, size=3)  # -> [1, 1, 2]  (member 1 twice)
    """
    singles = [score_indices([i]) for i in range(n_members)]
    best_i = int(np.argmax(singles))
    selected = [best_i]
    best = singles[best_i]
    for _ in range(max(0, size - 1)):
        cand = [score_indices(selected + [i]) for i in range(n_members)]
        j = int(np.argmax(cand))
        if cand[j] <= best:  # no member improves the ensemble -> stop
            break
        selected.append(j)
        best = cand[j]
    return selected


def holdout_split(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    seed: int = 42,
    holdout_frac: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded ``(train, holdout)`` split of the main table — the shared bench.

    Single-sourced so every method compared in the benchmark (the extension,
    AutoGluon, MLE-STAR) scores on the *same* rows. Classification uses a
    stratified draw (an imbalanced target's random 25% can miss a class);
    regression a plain seeded sample.

    Example:
        train, holdout = holdout_split(df, "target", "classification")
        # train.index and holdout.index partition df.index deterministically
    """
    if task_type == "classification":
        holdout = df.groupby(df[target], group_keys=False).sample(
            frac=holdout_frac, random_state=seed
        )
    else:
        holdout = df.sample(frac=holdout_frac, random_state=seed)
    train = df.drop(index=holdout.index)
    return train, holdout


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
    size: int | None = None,
    holdout: pd.DataFrame | None = None,
) -> dict:
    """Fit a candidate POOL and Caruana-select a weighted ensemble.

    ``states`` is a candidate pool (e.g. the top-N distinct cache configs, N > k)
    — reaching past the incumbent's near-duplicate neighbours to whatever
    model-family diversity the search buried below the very top.
    ``_caruana_select`` greedily builds an ensemble of at most ``size`` members
    (default = the whole pool), weighting by selection multiplicity and adding a
    member only if it helps.

    ``holdout`` is the SHARED on-disk bench (``pipeline.load_holdout``: the
    ``test.csv`` rows no method trained on). When given, selection and reporting
    are kept strictly apart — the pool is fitted on an inner split of ``df`` and
    the ensemble is *selected* on the inner validation rows, then refitted on all
    of ``df`` and *scored* on ``holdout``. Selecting on the reported rows would
    make ``ensemble_score`` a greedy maximum over the published metric rather
    than an out-of-sample number.

    When ``holdout`` is None (no staged bench) it falls back to carving a split
    out of ``df`` and selecting on the same rows it reports — an optimistic
    upper bound. Prefer passing the staged holdout.

    Predictions are combined weightedly — weighted mean for regression, weighted
    mean ``predict_proba`` (argmax for label metrics) for classification,
    weighted vote when no proba. The legacy *unweighted* mean over the top-``size``
    members is returned too (``ensemble_score_mean``), computed on the same fits,
    so every call is a free A/B of Caruana vs the old combiner.

    Example:
        evaluate_top_k(plan, top_k_states(cache, 10), df, "target",
                       "classification", scoring="accuracy", size=3)
        # -> {"scorer": "accuracy", "k": 2, "pool_size": 10,
        #     "ensemble_score": 0.92, "ensemble_score_mean": 0.90,
        #     "weights": [0.67, 0.33], "selected_states": [...],
        #     "individual_scores": [0.89, ...]}
    """
    if scoring not in _METRIC_FNS:
        raise ValueError(f"unsupported scoring for ensembling: {scoring!r}")
    if size is None or size > len(states):
        size = len(states)

    def _fit_pool(fit_rows, eval_rows):
        """Fit every pool member on `fit_rows`, predict `eval_rows`."""
        fit_env = {main_var: fit_rows, **(aux or {})}
        # the plan's X node drops the target itself, so eval_rows keeps the
        # column (only fit mode ever evaluates the y mark)
        eval_env = {main_var: eval_rows, **(aux or {})}
        preds, probs, klasses = [], [], None
        for state in states:
            apply_state(plan, state)
            learner = plan.skb.make_learner()
            learner.fit(fit_env)
            preds.append(np.asarray(learner.predict(eval_env)))
            proba = None
            if task_type == "classification":
                try:
                    proba = np.asarray(learner.predict_proba(eval_env))
                    klasses = getattr(learner, "classes_", klasses)
                except Exception:
                    proba = None
            probs.append(proba)
        return preds, probs, klasses

    def _combiner(preds, probs, klasses, y):
        """Score a weighted ensemble of member `indices` (repeats carry weight)."""
        have_proba = task_type == "classification" and all(
            p is not None for p in probs
        )
        labels = np.unique(y) if klasses is None else np.asarray(klasses)

        def _score_indices(indices):
            if have_proba:
                mean_proba = np.mean([probs[i] for i in indices], axis=0)
                pred, proba = labels[np.argmax(mean_proba, axis=1)], mean_proba
            elif task_type == "classification":
                stacked = np.stack([preds[i] for i in indices])
                pred, proba = (
                    pd.DataFrame(stacked).mode(axis=0).iloc[0].to_numpy(),
                    None,
                )
            else:
                pred, proba = np.mean([preds[i] for i in indices], axis=0), None
            return _score(scoring, y, pred, proba)

        return _score_indices

    if holdout is None:
        # Legacy/offline path: no on-disk bench, so carve one out of `df`. The
        # ensemble is then SELECTED on the very rows it is scored on — an
        # optimistic upper bound, not an out-of-sample score. Only used where no
        # staged holdout exists (see pipeline.load_holdout).
        fit_rows, holdout = holdout_split(
            df, target, task_type, seed=seed, holdout_frac=holdout_frac
        )
        select_on_report_rows = True
    else:
        # Shared-bench path: `holdout` is the on-disk test.csv/test_answer.csv
        # that no method trained on. Fit on ALL of train.
        fit_rows = df
        select_on_report_rows = False

    y_true = holdout[target]

    if select_on_report_rows or len(states) == 1:
        # nothing to select (single member), or the legacy path
        predictions, probas, classes = _fit_pool(fit_rows, holdout)
        score_report = _combiner(predictions, probas, classes, y_true)
        score_select = score_report
    else:
        # SELECTION on an inner split of train — never on the reported holdout,
        # or Caruana would be greedily maximising the number we publish (exactly
        # the ensemble-overfitting failure we criticise MLE-STAR for).
        inner_fit, inner_val = holdout_split(
            df, target, task_type, seed=seed, holdout_frac=holdout_frac
        )
        sel_preds, sel_probs, sel_classes = _fit_pool(inner_fit, inner_val)
        score_select = _combiner(
            sel_preds, sel_probs, sel_classes, inner_val[target]
        )
        # REPORTING refits the pool on all of train and predicts the real holdout
        predictions, probas, classes = _fit_pool(fit_rows, holdout)
        score_report = _combiner(predictions, probas, classes, y_true)

    individual = [score_report([i]) for i in range(len(states))]

    # Caruana greedy weighted ensemble (robust to near-duplicate configs);
    # selected on `score_select`, reported on `score_report`.
    selected = _caruana_select(len(states), score_select, size)
    ens_score = score_report(selected)
    # legacy unweighted mean over the top-`size` pool members (the pool is
    # score-ranked, so this reproduces the old plain top-k ensemble) — a free A/B
    mean_score = score_report(list(range(size)))

    counts = Counter(selected)
    distinct = list(counts)  # pool indices, first-selected order
    total = len(selected)

    return {
        "scorer": scoring,
        "k": len(distinct),
        "pool_size": len(states),
        "states": states,
        "selected_states": [states[i] for i in distinct],
        "weights": [counts[i] / total for i in distinct],
        "individual_scores": individual,
        "ensemble_score": ens_score,
        "ensemble_score_mean": mean_score,
    }
