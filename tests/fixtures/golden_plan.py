"""Golden fixture: a hand-written skrub DataOps plan with named choice nodes.

This stands in for what A_retriever will eventually generate, so both tracks
can be developed and tested without any LLM call. Mirrors the project brief
§4 example.
"""

import numpy as np
import pandas as pd


def make_toy_df(n_rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic binary-classification table: 2 numeric + 1 categorical + target."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n_rows)
    x2 = rng.normal(size=n_rows)
    cat = rng.choice(["red", "green", "blue"], size=n_rows)
    # target depends on the features so models can actually learn something
    logits = 1.5 * x1 - x2 + (cat == "red") * 1.0
    target = (logits + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "color": cat, "target": target})


def build_golden_plan(df: pd.DataFrame):
    """The reference skrub plan: TableVectorizer with an encoder choice,
    then a GBM/RF model choice with tunable hyperparameters.

    Every stochastic estimator is seeded — unseeded estimators make rollout
    scores non-deterministic and UCT values never converge (anti-pattern #6).
    """
    import skrub
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )

    data = skrub.var("data", df)
    X = data.drop(columns="target").skb.mark_as_X()
    y = data["target"].skb.mark_as_y()

    encoded = X.skb.apply(
        skrub.TableVectorizer(
            high_cardinality=skrub.choose_from(
                [skrub.GapEncoder(random_state=42), skrub.MinHashEncoder()],
                name="encoder",
            )
        )
    )

    pred = encoded.skb.apply(
        skrub.choose_from(
            {
                "GBM": GradientBoostingClassifier(
                    n_estimators=skrub.choose_int(50, 200, name="n_trees"),
                    learning_rate=skrub.choose_float(0.01, 0.3, log=True, name="lr"),
                    random_state=42,
                ),
                "RF": RandomForestClassifier(
                    n_estimators=skrub.choose_int(50, 200, name="rf_trees"),
                    random_state=42,
                ),
            },
            name="model",
        ).as_data_op(),
        y=y,
    )
    return pred
