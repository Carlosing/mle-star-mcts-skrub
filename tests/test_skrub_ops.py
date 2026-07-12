"""Smoke tests for the skrub wrapper layer.

    uv run python -m pytest tests/test_skrub_ops.py -v

These pin down the skrub 0.9 API behavior the project depends on
(see the module docstring in skrub_ops.py). If a skrub upgrade changes the
internal _evaluation helpers, these tests are the early-warning system.
"""

import os
import sys

import pytest
from fixtures.golden_plan import build_golden_plan, make_toy_df

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

skrub = pytest.importorskip("skrub")

from machine_learning_engineering import skrub_ops


@pytest.fixture(scope="module")
def df():
    return make_toy_df()


@pytest.fixture(scope="module")
def plan(df):
    return build_golden_plan(df)


# --- Track A: introspection -------------------------------------------------


def test_action_space_covers_all_named_choices(plan):
    space = skrub_ops.get_action_space(plan)
    assert set(space) == {"encoder", "model", "n_trees", "lr", "rf_trees"}
    assert space["model"] == ["GBM", "RF"]
    assert len(space["encoder"]) == 2
    # numeric choices are discretized within their declared bounds
    assert all(50 <= v <= 200 for v in space["n_trees"])
    assert all(isinstance(v, int) for v in space["n_trees"])
    assert all(0.01 <= v <= 0.3 for v in space["lr"])


def test_state_is_a_compact_dict(plan):
    state = skrub_ops.get_state(plan)
    print(state)
    assert isinstance(state, dict)
    assert state["model"] in ("GBM", "RF")


def test_steps_summary_mentions_the_vectorizer(plan):
    assert "TableVectorizer" in str(skrub_ops.get_steps_summary(plan))


def test_find_named_node(plan):
    assert skrub_ops.find_node(plan, "encoder") is not None
    assert skrub_ops.find_node(plan, "random") is None


# --- Track B: state application and rollouts ---------------------------------


def test_apply_state_rejects_unknown_names_and_options(plan):
    with pytest.raises(ValueError):
        skrub_ops.apply_state(plan, {"nonsense": 1})
    with pytest.raises(ValueError):
        skrub_ops.apply_state(plan, {"model": "DoesNotExist"})


def test_rollout_produces_a_real_score(plan, df):
    rollout = skrub_ops.make_rollout_fn(plan, df)
    score = rollout(skrub_ops.get_state(plan))
    # a real CV score on learnable data, not the 0.0 failure fallback
    assert 0.5 < score <= 1.0


def test_rollout_is_deterministic(plan, df):
    rollout = skrub_ops.make_rollout_fn(plan, df)
    state = {**skrub_ops.get_state(plan), "model": "RF", "rf_trees": 150}
    assert rollout(state) == rollout(state)


def test_rollout_reward_is_identical_across_n_jobs(plan, df):
    """CV fold-parallelism must not change the reward — folds are averaged
    (order-independent) on a seeded subsample, so n_jobs is purely a wall-clock
    knob. If 1 vs 6 ever diverged, the score cache (which assumes a state's
    reward is fixed) would split on machine core count."""
    state = {**skrub_ops.get_state(plan), "model": "RF", "rf_trees": 150}
    serial = skrub_ops.make_rollout_fn(plan, df, n_jobs=1)(state)
    parallel = skrub_ops.make_rollout_fn(plan, df, n_jobs=6)(state)
    assert serial == parallel


def test_booster_rollout_is_identical_and_crashfree_across_n_jobs():
    """The booster fork path: LGBM with the estimator's own n_jobs pinned to 1
    (REGISTRY) must roll out identically and without segfault whether CV folds
    run serially or forked across 6 workers."""
    import numpy as np
    import pandas as pd

    from machine_learning_engineering.spec_resolver import resolve_spec

    n = 300
    rs = np.random.RandomState(0)
    d = pd.DataFrame(
        {"a": rs.rand(n), "b": [f"t{i % 5}" for i in range(n)],
         "y": [i % 2 for i in range(n)]}
    )
    spec = resolve_spec(
        {"model": ["lightgbm.LGBMClassifier"]}, task_type="classification"
    )
    p = skrub_ops.build_staged_plan(spec, d, target="y")
    root = skrub_ops.get_default_state(p)
    serial = skrub_ops.make_rollout_fn(
        p, d, scoring="accuracy", target="y", stratify=True, n_jobs=1
    )(root)
    parallel = skrub_ops.make_rollout_fn(
        p, d, scoring="accuracy", target="y", stratify=True, n_jobs=6
    )(root)
    assert serial > 0.0 and serial == parallel


def test_llm_authored_datetime_slot_surfaces_weekday():
    """The LLM can make the vectorizer's `datetime` slot a search axis by
    authoring `vectorizer.slots.datetime`. skrub's stock DatetimeEncoder omits
    weekday, so an add_weekday=True option must out-score the bare default on a
    task whose label IS the weekday (no code-owned auto-menu — the option comes
    from the plan)."""
    import numpy as np
    import pandas as pd

    from machine_learning_engineering.spec_resolver import resolve_spec

    rng = np.random.default_rng(0)
    n = 150
    dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 700, n), unit="D"
    )
    d = pd.DataFrame(
        {
            "signup": dates.strftime("%Y-%m-%d"),
            "city": rng.choice(["NY", "LA", "SF"], n),
            "y": (dates.dayofweek >= 5).astype(int),  # label == is-weekend
        }
    )
    spec = resolve_spec(
        {
            "vectorizer": {
                "slots": {
                    "datetime": [
                        "skrub.DatetimeEncoder",  # stock default (no weekday)
                        {
                            "name": "skrub.DatetimeEncoder",
                            "params": {
                                "resolution": {"choice": ["day"]},
                                "add_weekday": {"choice": [True]},
                            },
                        },
                    ]
                }
            },
            "model": ["sklearn.ensemble.HistGradientBoostingClassifier"],
        },
        task_type="classification",
        main_columns=["signup", "city"],
    )
    p = skrub_ops.build_staged_plan(spec, d, target="y")
    space = skrub_ops.get_action_space(p)
    assert "vectorizer__datetime" in space
    root = skrub_ops.get_default_state(p)
    assert "weekday" not in root["vectorizer__datetime"].lower()  # stock root

    rollout = skrub_ops.make_rollout_fn(
        p, d, scoring="accuracy", target="y", stratify=True, n_jobs=2
    )
    weekday_opt = next(
        o for o in space["vectorizer__datetime"] if "add_weekday" in o
    )
    weekday_state = {**root, "vectorizer__datetime": weekday_opt}
    assert rollout(weekday_state) > rollout(dict(root))


def test_rollout_distinguishes_states(plan, df):
    rollout = skrub_ops.make_rollout_fn(plan, df)
    base = skrub_ops.get_state(plan)
    a = rollout({**base, "model": "GBM", "n_trees": 50})
    b = rollout({**base, "model": "RF", "rf_trees": 200})
    assert a != b  # different configs must be discriminable for UCT to work


def test_rollout_swallows_bad_configs(plan, df):
    rollout = skrub_ops.make_rollout_fn(plan, df)
    assert rollout({"model": "DoesNotExist"}) == 0.0


def test_roc_auc_scores_a_proba_only_classifier(plan, df):
    # RandomForest exposes predict_proba but NOT decision_function. skrub's
    # learner reports as a transformer, so sklearn's built-in roc_auc scorer
    # never reduces the 2-column proba to the positive class -> every fold used
    # to fail (0.0). _resolve_scoring fixes it (regression test for that).
    rollout = skrub_ops.make_rollout_fn(
        plan, df, scoring="roc_auc", stratify=True, target="target"
    )
    score = rollout({**skrub_ops.get_state(plan), "model": "RF"})
    assert 0.5 < score <= 1.0


def test_evaluate_full_scores_on_all_rows(plan, df):
    score = skrub_ops.evaluate_full(plan, skrub_ops.get_state(plan), df)
    assert 0.5 < score <= 1.0


# --- Track A: ablation and targeting -----------------------------------------


def test_ablation_scores_every_model_option(plan, df):
    results = skrub_ops.run_ablation(plan, "model", df)
    assert set(results) == {"GBM", "RF"}
    assert all(0.0 <= v <= 1.0 for v in results.values())


def test_pick_target_node_prefers_high_variance():
    results = {
        "boring": {"a": 0.5, "b": 0.5},
        "decisive": {"a": 0.2, "b": 0.9},
    }
    assert skrub_ops.pick_target_node(results) == "decisive"


# --- The end-to-end sanity check: MCTS over the golden plan ------------------


def test_mcts_search_improves_over_default(plan, df):
    from machine_learning_engineering import mcts

    root_state = skrub_ops.get_state(plan)
    rollout = skrub_ops.make_rollout_fn(plan, df)
    default_score = rollout(root_state)
    best_state, best_score, root = mcts.mcts_search(
        root_state,
        skrub_ops.get_action_space(plan),
        rollout,
        budget=15,
    )
    assert best_score >= default_score
    assert root.N == 15
