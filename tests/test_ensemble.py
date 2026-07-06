"""Top-k ensemble — the thin read-off over the persisted score cache.

No new search, no LLM: `top_k_states` ranks the cache, `evaluate_top_k` fits
each config on a seeded train split and soft-votes / averages on the holdout.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("skrub")

from machine_learning_engineering import ensemble, spec_resolver
from machine_learning_engineering.search_loop import run_search_loop

from fixtures.golden_plan import make_toy_df
from fixtures.staged_plan import make_relational_data, relational_spec


def test_top_k_states_ranks_the_cache():
    cache = {
        (("model", "A"),): 0.7,
        (("model", "B"),): 0.9,
        (("model", "C"),): 0.8,
    }
    assert ensemble.top_k_states(cache, k=2) == [{"model": "B"}, {"model": "C"}]


@pytest.fixture(scope="module")
def toy_search():
    df = make_toy_df()
    spec = spec_resolver.resolve_spec(
        {
            "encoder_options": ["skrub.GapEncoder"],
            "model": [
                "sklearn.ensemble.HistGradientBoostingClassifier",
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.linear_model.LogisticRegression",
            ],
        },
        task_type="classification",
    )
    result = run_search_loop(spec, df, "target", scoring="accuracy", budget_per_step=8)
    return result, df


def test_evaluate_top_k_classification(toy_search):
    result, df = toy_search
    states = ensemble.top_k_states(result["score_cache"], k=3)
    report = ensemble.evaluate_top_k(
        result["plan"], states, df, "target", "classification", scoring="accuracy"
    )
    assert report["k"] == len(states)
    assert len(report["individual_scores"]) == report["k"]
    assert 0.0 <= report["ensemble_score"] <= 1.0
    # sanity: the ensemble is at least in the ballpark of its members
    assert report["ensemble_score"] >= min(report["individual_scores"]) - 0.05
    # deterministic: same states, same split, same result
    again = ensemble.evaluate_top_k(
        result["plan"], states, df, "target", "classification", scoring="accuracy"
    )
    assert again["ensemble_score"] == report["ensemble_score"]


def test_evaluate_top_k_roc_auc_uses_probabilities(toy_search):
    result, df = toy_search
    states = ensemble.top_k_states(result["score_cache"], k=2)
    report = ensemble.evaluate_top_k(
        result["plan"], states, df, "target", "classification", scoring="roc_auc"
    )
    assert 0.0 <= report["ensemble_score"] <= 1.0


def test_evaluate_top_k_relational_passes_aux_whole():
    main, aux = make_relational_data()
    result = run_search_loop(
        relational_spec(),
        main,
        "target",
        scoring="accuracy",
        budget_per_step=6,
        aux_tables=aux,
    )
    states = ensemble.top_k_states(result["score_cache"], k=2)
    report = ensemble.evaluate_top_k(
        result["plan"],
        states,
        main,
        "target",
        "classification",
        scoring="accuracy",
        aux=aux,
    )
    assert 0.0 <= report["ensemble_score"] <= 1.0


def test_evaluate_top_k_rejects_unknown_scorer(toy_search):
    result, df = toy_search
    with pytest.raises(ValueError):
        ensemble.evaluate_top_k(
            result["plan"],
            [{"model": "LogisticRegression"}],
            df,
            "target",
            "classification",
            scoring="not_a_scorer",
        )
