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
            "vectorizer": {"slots": {"high_cardinality": ["skrub.GapEncoder"]}},
            "model": [
                "sklearn.ensemble.HistGradientBoostingClassifier",
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.linear_model.LogisticRegression",
            ],
        },
        task_type="classification",
    )
    result = run_search_loop(
        spec, df, "target", scoring="accuracy", budget_per_step=8
    )
    return result, df


def test_caruana_selects_on_oof_predictions_not_on_the_reported_holdout(
    toy_search, monkeypatch
):
    """Selection and reporting must not touch the same rows.

    With a shared holdout, Caruana selects on out-of-fold predictions spanning
    ALL of train, and the ensemble is only scored on the holdout. Selecting on
    the holdout would make `ensemble_score` a greedy maximum over the number we
    publish — the ensemble-overfitting flaw we criticise MLE-STAR for.

    We spy on `_score` and record how many rows each call scored: selection must
    see all `len(train)` rows (that is the point of OOF over an inner split),
    reporting only the holdout rows, and the two counts must differ.
    """
    result, df = toy_search
    train_df = df.iloc[:80].reset_index(drop=True)
    holdout_df = df.iloc[80:].reset_index(drop=True)
    assert len(holdout_df) != len(train_df)  # counts must be distinguishable

    states = ensemble.top_k_states(result["score_cache"], k=3)
    seen = []
    real_score = ensemble._score

    def spy(scoring, y_true, pred, proba):
        seen.append(len(y_true))
        return real_score(scoring, y_true, pred, proba)

    monkeypatch.setattr(ensemble, "_score", spy)
    report = ensemble.evaluate_top_k(
        result["plan"],
        states,
        train_df,
        "target",
        "classification",
        scoring="accuracy",
        holdout=holdout_df,
        oof_splits=3,
    )

    assert len(train_df) in seen, "selection did not score OOF over all of train"
    assert len(holdout_df) in seen, "nothing scored the holdout rows"
    # every reported number is measured on the holdout
    assert len(report["individual_scores"]) == len(states)
    assert 0.0 <= report["ensemble_score"] <= 1.0


def test_legacy_selection_flag_restores_the_biased_path(toy_search):
    """`legacy_selection=True` selects on the reported rows — the pre-fix logic.

    Kept so results measured that way stay reproducible, and so the two can be
    A/B'd on an identical split. Its tell is that Caruana seeds with the best
    member ON THE REPORTED HOLDOUT and early-stops, so the ensemble can never
    come out below it — the very guarantee whose absence marks the honest path.
    """
    result, df = toy_search
    train_df = df.iloc[:80].reset_index(drop=True)
    holdout_df = df.iloc[80:].reset_index(drop=True)
    states = ensemble.top_k_states(result["score_cache"], k=3)

    kwargs = dict(
        plan=result["plan"], states=states, df=train_df, target="target",
        task_type="classification", scoring="accuracy", holdout=holdout_df,
    )
    legacy = ensemble.evaluate_top_k(**kwargs, legacy_selection=True)
    honest = ensemble.evaluate_top_k(**kwargs, legacy_selection=False)

    # every artifact records which logic produced it
    assert legacy["selection"] == "legacy_holdout"
    assert honest["selection"] == "oof_3fold"

    # the legacy path cannot lose to its own best member: it picked that member
    # by looking at the rows it reports on. That is the bias, stated as a test.
    assert legacy["ensemble_score"] >= max(legacy["individual_scores"]) - 1e-9


def test_pickled_ensemble_reproduces_its_holdout_predictions(toy_search, tmp_path):
    """The fitted ensemble round-trips through pickle and predicts identically."""
    import pickle

    result, df = toy_search
    train_df = df.iloc[:80].reset_index(drop=True)
    holdout_df = df.iloc[80:].reset_index(drop=True)
    states = ensemble.top_k_states(result["score_cache"], k=3)

    report = ensemble.evaluate_top_k(
        result["plan"], states, train_df, "target", "classification",
        scoring="accuracy", holdout=holdout_df,
    )
    predictor = report["predictor"]
    assert len(predictor.learners) == report["k"]
    assert len(predictor.weights) == report["k"]

    before = predictor.predict(holdout_df)
    path = tmp_path / "ensemble.pkl"
    path.write_bytes(pickle.dumps(predictor))
    after = pickle.loads(path.read_bytes()).predict(holdout_df)

    assert (before == after).all()
    # and it is the ensemble that was scored, not some other combination
    scored = ensemble._score("accuracy", holdout_df["target"], after, None)
    assert scored == pytest.approx(report["ensemble_score"], abs=1e-9)


def test_oof_folds_are_shared_by_every_candidate():
    """One fold assignment for the whole pool — the OOF matrix depends on it.

    Model i's prediction for row 5 and model j's are only combinable if neither
    trained on row 5. Per-candidate folds would make the matrix meaningless.
    """
    df = make_toy_df()
    a = ensemble._oof_folds(df, "target", "classification", seed=42, n_splits=3)
    b = ensemble._oof_folds(df, "target", "classification", seed=42, n_splits=3)
    assert len(a) == 3
    for (fit_a, val_a), (fit_b, val_b) in zip(a, b):
        assert (val_a == val_b).all()
        assert (fit_a == fit_b).all()
    # the validation folds partition the rows: every row predicted exactly once
    covered = sorted(i for _fit, val in a for i in val)
    assert covered == list(range(len(df)))


def test_oof_folds_fall_back_to_kfold_when_a_class_is_too_thin():
    """A class with fewer members than folds cannot be stratified — don't try.

    A fold landing on zero positives NaNs the scorer silently (skrub_ops._cv_kwarg
    documents the same trap).
    """
    df = make_toy_df().copy()
    df.loc[df.index[:-2], "target"] = 0  # leave one class with 2 members
    folds = ensemble._oof_folds(df, "target", "classification", seed=42, n_splits=3)
    assert len(folds) == 3  # produced KFold rather than raising


def test_evaluate_top_k_classification(toy_search):
    result, df = toy_search
    states = ensemble.top_k_states(result["score_cache"], k=3)
    report = ensemble.evaluate_top_k(
        result["plan"],
        states,
        df,
        "target",
        "classification",
        scoring="accuracy",
    )
    assert report["pool_size"] == len(states)
    assert 1 <= report["k"] <= len(states)  # Caruana caps at the pool size
    assert len(report["individual_scores"]) == report["pool_size"]
    assert 0.0 <= report["ensemble_score"] <= 1.0
    # Caruana seeds with the best single member + early-stops, so it is never
    # worse than the best pool member on this holdout
    assert report["ensemble_score"] >= max(report["individual_scores"]) - 1e-6
    # deterministic: same states, same split, same result
    again = ensemble.evaluate_top_k(
        result["plan"],
        states,
        df,
        "target",
        "classification",
        scoring="accuracy",
    )
    assert again["ensemble_score"] == report["ensemble_score"]


def test_evaluate_top_k_roc_auc_uses_probabilities(toy_search):
    result, df = toy_search
    states = ensemble.top_k_states(result["score_cache"], k=2)
    report = ensemble.evaluate_top_k(
        result["plan"],
        states,
        df,
        "target",
        "classification",
        scoring="roc_auc",
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


def test_caruana_selects_both_members_when_they_decorrelate():
    """Two individually-mediocre members whose errors cancel -> Caruana picks
    both and the ensemble strictly beats either alone."""
    import numpy as np

    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    m0 = y + np.array([1.0, -1, 1, -1, 1, -1])
    m1 = y + np.array([-1.0, 1, -1, 1, -1, 1])  # opposite error -> mean is exact
    members = [m0, m1]

    def score_idx(idx):
        pred = np.mean([members[i] for i in idx], axis=0)
        return ensemble._score("neg_root_mean_squared_error", y, pred, None)

    selected = ensemble._caruana_select(2, score_idx, size=2)
    assert set(selected) == {0, 1}
    assert score_idx(selected) > max(score_idx([0]), score_idx([1]))


def test_caruana_collapses_to_one_on_near_duplicate_pool():
    """A pool of identical members -> Caruana early-stops at size 1 (no spurious
    lift). This is the 'all configs very similar' case."""
    import numpy as np

    y = np.array([0.0, 1.0, 2.0, 3.0])
    biased = y + 0.5
    members = [biased, biased.copy(), biased.copy()]

    def score_idx(idx):
        pred = np.mean([members[i] for i in idx], axis=0)
        return ensemble._score("neg_root_mean_squared_error", y, pred, None)

    selected = ensemble._caruana_select(3, score_idx, size=3)
    assert len(selected) == 1  # adding a duplicate never improves -> stop


def test_evaluate_top_k_pool_and_weights(toy_search):
    """Caruana over a pool wider than `size`: pool_size is the fitted pool, k is
    the (capped) number of distinct selected members, weights normalise to 1,
    and the legacy unweighted score is reported for the A/B."""
    result, df = toy_search
    pool = ensemble.top_k_states(result["score_cache"], k=5)
    report = ensemble.evaluate_top_k(
        result["plan"], pool, df, "target", "classification",
        scoring="accuracy", size=3,
    )
    assert report["pool_size"] == len(pool)
    assert 1 <= report["k"] <= 3
    assert len(report["weights"]) == report["k"]
    assert abs(sum(report["weights"]) - 1.0) < 1e-9
    assert "ensemble_score_mean" in report
    assert report["ensemble_score"] >= max(report["individual_scores"]) - 1e-6


def test_top_k_states_collapses_states_that_are_the_same_pipeline():
    """`apply_state` resets omitted choices to their default, so a state that
    omits an HP and one that names it at its default fit identically. Without
    `defaults` the ensemble averages the same model k times and its score
    equals the incumbent's exactly."""
    defaults = {"model": "LGBM", "n_trees": 550}
    cache = {
        (("model", "LGBM"), ("n_trees", 550)): 0.90,  # explicit default
        (("model", "LGBM"),): 0.90,                   # same pipeline, omitted
        (("model", "LGBM"), ("n_trees", 100)): 0.85,  # genuinely different
    }
    assert len(ensemble.top_k_states(cache, k=3)) == 3  # old behaviour
    picked = ensemble.top_k_states(cache, k=3, defaults=defaults)
    assert len(picked) == 2
    assert {"model": "LGBM", "n_trees": 100} in picked
