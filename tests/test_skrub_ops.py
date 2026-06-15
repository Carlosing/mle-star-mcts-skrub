"""Smoke tests for the skrub wrapper layer — run these INSIDE the Docker
container (skrub is not installed on the host):

    docker run --rm -v "$PWD":/app mle-star python -m pytest tests/test_skrub_ops.py -v

These pin down the skrub 0.9 API behavior the project depends on
(see the module docstring in skrub_ops.py). If a skrub upgrade changes the
internal _evaluation helpers, these tests are the early-warning system.
"""

import os
import sys

import pytest
import importlib.util
from fixtures.golden_plan import build_golden_plan, make_toy_df

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

skrub = pytest.importorskip("skrub")


_BASE = os.path.join(os.path.dirname(__file__), "..", "machine_learning_engineering")
_spec = importlib.util.spec_from_file_location(
    "skrub_ops", os.path.join(_BASE, "skrub_ops.py")
)
skrub_ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(skrub_ops)


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


def test_rollout_distinguishes_states(plan, df):
    rollout = skrub_ops.make_rollout_fn(plan, df)
    base = skrub_ops.get_state(plan)
    a = rollout({**base, "model": "GBM", "n_trees": 50})
    b = rollout({**base, "model": "RF", "rf_trees": 200})
    assert a != b  # different configs must be discriminable for UCT to work


def test_rollout_swallows_bad_configs(plan, df):
    rollout = skrub_ops.make_rollout_fn(plan, df)
    assert rollout({"model": "DoesNotExist"}) == 0.0


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
    _spec2 = importlib.util.spec_from_file_location(
        "mcts", os.path.join(_BASE, "mcts.py")
    )
    mcts = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(mcts)

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
