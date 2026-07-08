"""Composite golden-path test: MCTS (mcts.py) driving real skrub rollouts
(skrub_ops.py) over the golden plan. Run inside Docker or the local venv:

    docker run --rm -v "$PWD":/app mle-star python -m pytest tests/test_integration.py -v
    # or natively:  .venv/bin/python -m pytest tests/test_integration.py -v

Rollouts are seeded, so the search is deterministic — but rollout *values* can
shift across sklearn versions, so this file asserts STRUCTURE and RANGES, not
exact scores:
  * skrub config <-> MCTS state round-trips both ways;
  * every action-space option is a valid MCTS action;
  * rollouts land in [0, 1];
  * the search tree is well-formed (unique nodes, parent/child links,
    child.N <= parent.N, 0 <= Q <= N, mean reward in [0, 1]);
  * per-iteration statistics update by the right amounts.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Tree artifacts are written here (a gitignored scratch dir at the repo root).
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "temp"

skrub = pytest.importorskip("skrub")

_BASE = os.path.join(
    os.path.dirname(__file__), "..", "machine_learning_engineering"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_BASE, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


skrub_ops = _load("skrub_ops")
mcts = _load("mcts")

from fixtures.golden_plan import build_golden_plan, make_toy_df

BUDGET = 25


@pytest.fixture(scope="module")
def df():
    return make_toy_df()


@pytest.fixture(scope="module")
def plan(df):
    return build_golden_plan(df)


@pytest.fixture(scope="module")
def rollout(plan, df):
    return skrub_ops.make_rollout_fn(plan, df)


def _iter_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


# --- sanity: skrub config <-> MCTS state translation -------------------------


def test_state_key_is_order_independent():
    assert mcts.state_key({"a": 1, "b": 2}) == mcts.state_key({"b": 2, "a": 1})


def test_skrub_default_is_a_valid_mcts_state(plan):
    """get_state(plan) -> a dict usable directly as an MCTS root state."""
    state = skrub_ops.get_state(plan)
    assert isinstance(state, dict) and state
    assert mcts.state_key(state)  # hashable, no error


def test_every_action_space_option_is_a_valid_action(plan):
    """Each (choice, option) the action space offers applies without error —
    i.e. every MCTS action translates back to a configurable skrub plan."""
    space = skrub_ops.get_action_space(plan)
    for name, options in space.items():
        for opt in options:
            skrub_ops.apply_state(plan, {name: opt})  # must not raise


def test_state_roundtrips_through_the_plan(plan):
    """state -> apply_state -> get_state reflects the chosen (active) values,
    for both the GBM and RF branches (conditional choices)."""
    space = skrub_ops.get_action_space(plan)

    gbm = {"encoder": space["encoder"][1], "model": "GBM", "n_trees": 100}
    skrub_ops.apply_state(plan, gbm)
    got = skrub_ops.get_state(plan)
    assert got["model"] == "GBM"
    assert got["encoder"] == gbm["encoder"]
    assert got["n_trees"] == 100

    rf = {"model": "RF", "rf_trees": 200}
    skrub_ops.apply_state(plan, rf)
    got = skrub_ops.get_state(plan)
    assert got["model"] == "RF"
    assert got["rf_trees"] == 200
    assert "n_trees" not in got  # inactive under RF


def test_expand_produces_single_edit_neighbors(plan):
    """skrub's action space + mcts.expand => children differ from the parent in
    exactly one choice (the edit-distance topology)."""
    root_state = skrub_ops.get_state(plan)
    space = skrub_ops.get_action_space(plan)
    root = mcts.MCTSNode(state=root_state)
    tried = {mcts.state_key(root_state)}
    children = mcts.expand(root, space, tried)
    assert children
    for child in children:
        differing = [
            k for k in child.state if child.state[k] != root_state.get(k)
        ]
        assert len(differing) == 1


# --- rollout behaviour -------------------------------------------------------


def test_rollouts_are_in_unit_range_and_deterministic(plan, df, rollout):
    space = skrub_ops.get_action_space(plan)
    base = skrub_ops.get_state(plan)
    states = [base]
    for name, options in space.items():
        states.append({**base, name: options[-1]})
    for s in states:
        score = rollout(s)
        assert 0.0 <= score <= 1.0
        assert rollout(s) == score  # deterministic


# --- the full search: tree structure & statistics ----------------------------


def test_search_builds_a_well_formed_tree(plan, df, rollout):
    root_state = skrub_ops.get_state(plan)
    space = skrub_ops.get_action_space(plan)
    tried = {mcts.state_key(root_state)}
    best_state, best_score, root = mcts.mcts_search(
        root_state, space, rollout, budget=BUDGET, tried_states=tried
    )

    nodes = list(_iter_nodes(root))
    keys = [mcts.state_key(n.state) for n in nodes]

    # root visited exactly `budget` times; tree actually grew
    assert root.N == BUDGET
    assert len(nodes) > 1 and root.children

    # each configuration is a unique node, and the node set == tried_states
    assert len(keys) == len(set(keys))
    assert set(keys) == tried

    for n in nodes:
        # parent/child links are consistent
        for c in n.children:
            assert c.parent is n
            assert c.N <= n.N  # a child can't be visited more than its parent
        # Q is a sum of rewards in [0, 1], so 0 <= Q <= N and mean in [0, 1]
        assert -1e-9 <= n.Q <= n.N + 1e-9
        if n.N > 0:
            assert -1e-9 <= n.Q / n.N <= 1.0 + 1e-9

    # the reported best is real and not worse than the root baseline
    assert abs(rollout(best_state) - best_score) < 1e-9
    assert best_score >= rollout(root_state) - 1e-9


def test_per_iteration_statistics_update_correctly(plan, df, rollout):
    """Drive the search one iteration at a time and check the bookkeeping:
    each iteration adds exactly 1 visit to the root and a reward in [0, 1]."""
    root_state = skrub_ops.get_state(plan)
    space = skrub_ops.get_action_space(plan)
    tried = {mcts.state_key(root_state)}
    root = None

    for _ in range(10):
        prev_n = root.N if root is not None else 0
        prev_q = root.Q if root is not None else 0.0
        _, _, root = mcts.mcts_search(
            root_state, space, rollout, budget=1, root=root, tried_states=tried
        )
        assert root.N == prev_n + 1  # exactly one backprop reached the root
        delta_q = root.Q - prev_q
        assert -1e-9 <= delta_q <= 1.0 + 1e-9  # one reward in [0, 1]

    # tree persisted and accumulated across the 10 single-step calls
    assert root.N == 10
    assert len(tried) >= 1


def test_writes_real_search_tree_artifacts(plan, rollout):
    """Run a real skrub-driven search and persist the tree to <repo>/temp/
    (gitignored) as DOT + ASCII, then verify the written files carry real edge
    changes and final UCT scores."""
    root_state = skrub_ops.get_state(plan)
    space = skrub_ops.get_action_space(plan)
    _, _, root = mcts.mcts_search(root_state, space, rollout, budget=BUDGET)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    dot_path = ARTIFACT_DIR / "golden_tree.dot"
    txt_path = ARTIFACT_DIR / "golden_tree.txt"
    dot_path.write_text(mcts.to_dot(root))
    txt_path.write_text(mcts.print_tree(root))

    dot = dot_path.read_text()
    assert dot.startswith("digraph mcts {") and dot.rstrip().endswith("}")
    assert "UCT=" in dot and "->" in dot  # real edges + final UCT scores
    # edge labels reference real choice names from the plan's action space
    assert any(name in dot for name in space)
    assert txt_path.read_text().splitlines()[0].startswith("ROOT")
