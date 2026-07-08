"""Staged fixture: a rich, per-stage operator menu for constructive search.

Stands in for `A_retriever`'s "dissected plan of choices at each step". Unlike
`golden_plan` (a fixed pipeline with tunable choices), this exposes structural
stages — scale / feature-engineering on/off, plus the model — so MCTS can
search the *construction* of the pipeline.

The dataset is built around a dominant feature *interaction* (x1*x2), so the
reward landscape rewards enrichment in a way that mirrors the project's thesis:
a linear model alone is weak, but climbs sharply once PolynomialFeatures is
added — while a tree model captures the interaction natively.
"""

import numpy as np
import pandas as pd


def make_interaction_df(n_rows: int = 800, seed: int = 0) -> pd.DataFrame:
    """Binary target driven mainly by an x1*x2 interaction + a categorical."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n_rows), rng.normal(size=n_rows)
    cat = rng.choice(["a", "b", "c"], size=n_rows)
    logits = 3.0 * x1 * x2 + (cat == "a") * 1.0
    target = (logits + rng.normal(scale=0.3, size=n_rows) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "target": target})


def staged_spec() -> dict:
    """The LLM-style per-stage operator menu (real estimator instances).

    Estimators are seeded so rollouts are reproducible (anti-pattern #6).
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    import skrub

    return {
        "clean_options": [None, skrub.Cleaner()],
        "encoder_options": [
            skrub.GapEncoder(random_state=42),
            skrub.MinHashEncoder(),
        ],
        "stages": [
            {"name": "scale", "options": [None, StandardScaler()]},
            {
                "name": "feature_eng",
                "options": [
                    None,
                    PolynomialFeatures(degree=2, include_bias=False),
                ],
            },
        ],
        "model": {
            "LogReg": LogisticRegression(max_iter=1000),
            "GBM": GradientBoostingClassifier(random_state=42),
            "RF": RandomForestClassifier(random_state=42),
        },
    }


def make_relational_data(n_ids: int = 400, seed: int = 0):
    """A main table + one auxiliary table, where the target depends on a
    per-id *aggregate* of the auxiliary table.

    Without the assemble (AggJoiner) stage the main table has no predictive
    feature (test ids are unseen), so the score is ~chance; aggregating the
    auxiliary 'v' by id surfaces the signal. This mirrors the enrichment story
    for the relational case.

    Returns (main_df, {"aux": aux_df}).
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(n_ids)

    per_id_mean = {}
    rows = []
    for i in ids:
        k = int(rng.integers(2, 6))
        center = rng.normal()
        vals = rng.normal(loc=center, scale=1.0, size=k)
        per_id_mean[i] = float(vals.mean())
        rows.extend((int(i), float(v)) for v in vals)
    aux = pd.DataFrame(rows, columns=["id", "v"])

    threshold = np.median(list(per_id_mean.values()))
    main = pd.DataFrame(
        {
            "id": ids,
            "noise": rng.normal(size=n_ids),  # a non-informative feature
            "target": [
                int(per_id_mean[i] + rng.normal(scale=0.3) > threshold)
                for i in ids
            ],
        }
    )
    return main, {"aux": aux}


def relational_spec() -> dict:
    """Staged spec with a searchable relational *assemble* stage."""
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    return {
        "assemble": [
            {
                "name": "aux_mean",
                "table": "aux",
                "operations": ["mean"],
                "key": "id",
                "cols": ["v"],
            },
            {
                "name": "aux_mean_max_min",
                "table": "aux",
                "operations": ["mean", "max", "min"],
                "key": "id",
                "cols": ["v"],
            },
        ],
        "model": {
            "GBM": GradientBoostingClassifier(random_state=42),
            "RF": RandomForestClassifier(random_state=42),
            "LogReg": LogisticRegression(max_iter=1000),
        },
    }
