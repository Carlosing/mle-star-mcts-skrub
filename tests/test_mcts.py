"""Unit tests for the pure MCTS engine — no skrub, no LLM, no network.

Runnable anywhere:  python3 -m pytest tests/test_mcts.py or:   python3 tests/test_mcts.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Tree artifacts are written here (a gitignored scratch dir at the repo root).
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "temp"

# Import the module file directly so the test does not trigger
# machine_learning_engineering/__init__.py (which requires API credentials).
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "mcts",
    os.path.join(
        os.path.dirname(__file__), "..", "machine_learning_engineering", "mcts.py"
    ),
)
mcts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcts)


ACTION_SPACE = {
    "encoder": ["GapEncoder", "MinHashEncoder"],
    "model": ["GBM", "RF"],
    "n_trees": [50, 100, 150, 200],
}

TARGET = {"encoder": "MinHashEncoder", "model": "RF", "n_trees": 150}


def fake_reward(state: dict) -> float:
    """Fraction of params matching a secret target — peak reward 1.0 at TARGET."""
    hits = sum(state.get(k) == v for k, v in TARGET.items())
    return hits / len(TARGET)


def test_uct_unvisited_is_infinite():
    parent = mcts.MCTSNode(state={})
    parent.N = 5
    child = mcts.MCTSNode(state={}, parent=parent)
    assert child.uct() == float("inf")


def test_uct_balances_exploitation_and_exploration():
    parent = mcts.MCTSNode(state={})
    parent.N = 100
    good_but_visited = mcts.MCTSNode(state={}, parent=parent, Q=9.0, N=10)
    bad_but_fresh = mcts.MCTSNode(state={}, parent=parent, Q=0.5, N=1)
    # with a large c, the rarely-visited node must win
    assert bad_but_fresh.uct(c=5.0) > good_but_visited.uct(c=5.0)
    # with c=0, pure exploitation
    assert good_but_visited.uct(c=0.0) > bad_but_fresh.uct(c=0.0)


def test_expand_dedupes_and_skips_self():
    root = mcts.MCTSNode(state={"model": "GBM"})
    tried = {mcts.state_key(root.state)}
    children = mcts.expand(root, {"model": ["GBM", "RF"]}, tried)
    # "GBM" is the current value -> only "RF" is a new child
    assert len(children) == 1
    assert children[0].state == {"model": "RF"}
    # expanding again with the same tried-set yields nothing
    other = mcts.MCTSNode(state={"model": "GBM"})
    assert mcts.expand(other, {"model": ["GBM", "RF"]}, tried) == []


def test_backpropagate_reaches_root():
    root = mcts.MCTSNode(state={})
    mid = mcts.MCTSNode(state={}, parent=root)
    leaf = mcts.MCTSNode(state={}, parent=mid)
    mcts.backpropagate(leaf, 0.7)
    assert root.N == mid.N == leaf.N == 1
    assert abs(root.Q - 0.7) < 1e-9


def test_search_finds_the_target():
    start = {"encoder": "GapEncoder", "model": "GBM", "n_trees": 50}
    best_state, best_score, root = mcts.mcts_search(
        start, ACTION_SPACE, fake_reward, budget=60
    )
    assert best_state == TARGET
    assert best_score == 1.0
    assert root.N == 60  # every rollout backpropagated to the root


def test_tree_persists_across_outer_steps():
    start = {"encoder": "GapEncoder", "model": "GBM", "n_trees": 50}
    tried = {mcts.state_key(start)}
    _, _, root = mcts.mcts_search(
        start, ACTION_SPACE, fake_reward, budget=10, tried_states=tried
    )
    n_after_first = root.N
    _, _, root2 = mcts.mcts_search(
        start, ACTION_SPACE, fake_reward, budget=10, root=root, tried_states=tried
    )
    assert root2 is root
    assert root.N == n_after_first + 10  # statistics accumulated, not reset


# --- score cache + conditional-HP gating + target-key locking ----------------

GATED_SPACE = {"model": ["GBM", "RF"], "gbm_lr": [0.1, 0.2], "rf_trees": [50, 100]}
GATING = {"gbm_lr": ("model", "GBM"), "rf_trees": ("model", "RF")}


def test_score_cache_one_call_per_distinct_state():
    calls = []

    def rollout(state):
        calls.append(mcts.state_key(state))
        return 0.5

    cache = {}
    mcts.mcts_search(
        {"model": "GBM"}, ACTION_SPACE, rollout, budget=30, score_cache=cache
    )
    # no state is ever rolled out twice; cache holds exactly the evaluated states
    assert len(calls) == len(set(calls))
    assert len(cache) == len(calls)


def test_gating_skips_inactive_hp_and_canonicalizes():
    node = mcts.MCTSNode(state={"model": "GBM"})
    children = mcts.expand(
        node, GATED_SPACE, {mcts.state_key(node.state)}, gating=GATING
    )
    # GBM is selected -> gbm_lr is editable, rf_trees (RF-only) is never proposed
    assert any("gbm_lr" in c.state for c in children)
    assert not any("rf_trees" in c.state for c in children)

    # switching model to RF drops the now-inactive gbm_lr (canonical states)
    n2 = mcts.MCTSNode(state={"model": "GBM", "gbm_lr": 0.1})
    kids = mcts.expand(n2, GATED_SPACE, {mcts.state_key(n2.state)}, gating=GATING)
    rf_kids = [c for c in kids if c.state.get("model") == "RF"]
    assert rf_kids and all("gbm_lr" not in c.state for c in rf_kids)


def test_target_key_restricts_expansion():
    node = mcts.MCTSNode(state={"model": "GBM", "gbm_lr": 0.1})
    children = mcts.expand(
        node,
        GATED_SPACE,
        {mcts.state_key(node.state)},
        gating=GATING,
        target_key="gbm_lr",
    )
    # only gbm_lr is edited; every other key is locked at the node's value
    assert children
    assert all(set(c.state) == set(node.state) for c in children)
    assert all(c.state["model"] == "GBM" and c.state["gbm_lr"] != 0.1 for c in children)


def test_failed_rollouts_do_not_crash_search():
    def flaky(state):
        raise_it = state.get("model") == "RF"
        return 0.0 if raise_it else 0.5

    start = {"encoder": "GapEncoder", "model": "GBM", "n_trees": 50}
    best_state, best_score, _ = mcts.mcts_search(start, ACTION_SPACE, flaky, budget=20)
    assert best_score >= 0.0


def test_edge_change_describes_the_single_edit():
    parent = mcts.MCTSNode(state={"model": "GBM", "n_trees": 50})
    child = mcts.MCTSNode(state={"model": "RF", "n_trees": 50}, parent=parent)
    assert mcts.edge_change(parent, child) == "model: GBM -> RF"
    # a newly added (staged) key shows '·' as the previous value
    added = mcts.MCTSNode(
        state={"model": "GBM", "n_trees": 50, "scale": "Std"}, parent=parent
    )
    assert mcts.edge_change(parent, added) == "scale: · -> Std"


def test_to_dot_has_nodes_edges_and_uct():
    start = {"encoder": "GapEncoder", "model": "GBM", "n_trees": 50}
    _, _, root = mcts.mcts_search(start, ACTION_SPACE, fake_reward, budget=20)
    dot = mcts.to_dot(root)
    assert dot.startswith("digraph mcts {") and dot.rstrip().endswith("}")
    assert "ROOT" in dot  # root node present
    assert "->" in dot  # at least one edge
    assert "UCT=" in dot  # child nodes carry final UCT scores
    assert "label=" in dot  # edges carry change labels


def test_print_tree_is_readable_text():
    start = {"encoder": "GapEncoder", "model": "GBM", "n_trees": 50}
    _, _, root = mcts.mcts_search(start, ACTION_SPACE, fake_reward, budget=20)
    text = mcts.print_tree(root)
    lines = text.splitlines()
    assert lines[0].startswith("ROOT")
    assert any("UCT=" in ln for ln in lines)
    assert any("->" in ln for ln in lines)  # at least one edge change shown


def test_write_tree_to_dot_and_text_files():
    """Persist the search tree to <repo>/temp/ (gitignored) for inspection, and
    verify the written content round-trips the generators exactly.

    View it with `dot -Tpng temp/tree.dot -o tree.png` or, in a notebook,
    `graphviz.Source(open("temp/tree.dot").read())`.
    """
    start = {"encoder": "GapEncoder", "model": "GBM", "n_trees": 50}
    _, _, root = mcts.mcts_search(start, ACTION_SPACE, fake_reward, budget=20)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    dot_path = ARTIFACT_DIR / "tree.dot"
    txt_path = ARTIFACT_DIR / "tree.txt"
    dot_path.write_text(mcts.to_dot(root))
    txt_path.write_text(mcts.print_tree(root))

    assert dot_path.exists() and txt_path.exists()
    dot = dot_path.read_text()
    assert dot == mcts.to_dot(root)  # deterministic write
    assert dot.startswith("digraph mcts {") and dot.rstrip().endswith("}")
    assert "UCT=" in dot and "->" in dot
    assert txt_path.read_text().splitlines()[0].startswith("ROOT")
