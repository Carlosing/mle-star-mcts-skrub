"""MCTS engine for pipeline configuration search (Track B).

Pure Python — no skrub, no LLM, no I/O. The action space and the reward
function are injected by the caller, so this module can be unit-tested with
fake rewards and reused unchanged when skrub rollouts are plugged in
(see skrub_ops.make_rollout_fn).

Design constraints from the project brief:
- UCT selection, expansion, backpropagation are code, never LLM.
- The MCTS state is the compact param dict, never a full script.
- The tree persists across outer-loop steps (pass `root` back in).
- c = 0.5 (not sqrt(2)): proxy rewards are noisy.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class MCTSNode:
    state: dict  # compact param dict (the MCTS state S)
    parent: Optional["MCTSNode"] = None  # forward ref: class not yet defined
    children: list = field(default_factory=list)
    Q: float = 0.0  # cumulative reward
    N: int = 0  # visit count

    def uct(self, c: float = 0.5) -> float:
        if self.N == 0:  # Unvisited nodes are infinitely promising
            return float("inf")
        return self.Q / self.N + c * math.sqrt(
            math.log(self.parent.N) / self.N
        )  # Otherwise, UCT formula

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def state_key(state: dict) -> tuple:
    """Canonical hashable key for a state dict, used for deduplication."""
    return tuple(sorted(state.items()))


def find_state_node(root: "MCTSNode", state: dict) -> Optional["MCTSNode"]:
    """Return the tree node whose state matches `state`, or None.

    `tried_states` dedups states globally, so at most one node ever carries a
    given state — this is how the outer loop turns its tracked incumbent
    (`best_state`) back into the node to start a local (bonus) phase from.

    Example:
        find_state_node(root, {"model": "RF"})  # -> the RF node, or None
    """
    key = state_key(state)
    for node in _iter_nodes(root):
        if state_key(node.state) == key:
            return node
    return None


def select(node: MCTSNode, c: float = 0.5) -> MCTSNode:
    """Descend the tree by UCT until a leaf is reached."""
    # If node is not expanded (i.e., has no children), return it as the selected node
    # Otherwise, select the child with the highest UCT value and continue descending
    while node.children:
        node = max(node.children, key=lambda n: n.uct(c))
    return node


def canonicalize(state: dict, gating: Optional[dict] = None) -> dict:
    """Drop gated keys whose parent isn't at its activating outcome.

    `gating` maps a conditional choice name -> (parent_name, activating_label)
    (see skrub_ops.get_choice_gating). A state stays *canonical* — it carries
    only the hyperparameters that are actually active for its selected model —
    so the cache and `tried_states` dedup don't treat inactive-HP variants as
    distinct configs. No-op when `gating` is None.
    """
    if not gating:
        return state
    return {
        k: v
        for k, v in state.items()
        if k not in gating or state.get(gating[k][0]) == gating[k][1]
    }


def expand(
    node: MCTSNode,
    action_space: dict[str, list],
    tried_states: set[tuple],
    gating: Optional[dict] = None,
    target_key=None,
) -> list[MCTSNode]:
    """Generate children by swapping one choice value at a time.

    `action_space` maps choice name -> list of options. It comes from the
    skrub plan's choice nodes (skrub_ops.get_action_space) — never from an LLM
    (anti-pattern #1).

    `gating` (model-gated HPs): a conditional key is only edited when its parent
    is at the activating outcome in this node, and new states are canonicalized
    so inactive HPs are dropped. `target_key`, if given, restricts expansion to
    a single choice (Optional Feature 1 — lock all other stages) or, as a set/list of
    names, to that group of choices.
    """
    children = []
    if target_key is None:
        items = list(action_space.items())
    else:
        keys = [target_key] if isinstance(target_key, str) else list(target_key)
        items = [(k, action_space.get(k, [])) for k in keys]
    for choice_name, options in items:
        # skip a gated child whose parent isn't active in this node
        if gating and choice_name in gating:
            parent, activating = gating[choice_name]
            if node.state.get(parent) != activating:
                continue
        for option in options:
            new_state = canonicalize(
                {**node.state, choice_name: option}, gating
            )
            key = state_key(new_state)
            if key not in tried_states and new_state != node.state:
                children.append(MCTSNode(state=new_state, parent=node))
                tried_states.add(key)
    node.children = children
    return children


def backpropagate(node: MCTSNode, reward: float) -> None:
    # For backpropagation, we update the current node and then move up to the parent
    while node:
        node.N += 1
        node.Q += reward
        node = node.parent


def mcts_search(
    root_state: dict,
    action_space: dict[str, list],
    rollout_fn: Callable[[dict], float],
    budget: int = 50,
    c: float = 0.5,
    root: Optional[MCTSNode] = None,
    tried_states: Optional[set[tuple]] = None,
    prior_fn: Optional[Callable[[list[MCTSNode]], None]] = None,
    gating: Optional[dict] = None,
    target_key=None,
    score_cache: Optional[dict] = None,
    start_node: Optional[MCTSNode] = None,
    deadline: Optional[float] = None,
) -> tuple[dict, float, MCTSNode]:
    """Run `budget` MCTS iterations and return (best_state, best_score, root).

    Pass the returned `root`, `tried_states` and `score_cache` back in on the
    next outer-loop step so UCT statistics accumulate and rollouts are not
    recomputed (anti-pattern #7: never reset the tree).

    `rollout_fn(state) -> float` must return a reward in [0, 1] and never
    raise (failed configs score 0.0, see skrub_ops.make_rollout_fn).

    `prior_fn`, if given, is called on freshly expanded children and may
    warm-start their Q/N (the optional LLM expansion prior, brief §6). It
    should seed EVERY fresh child (a neutral default for unrated options):
    unvisited nodes score inf in UCT, so seeding only some children would
    make the rated ones *less* attractive than their unseeded siblings.
    The first rollout after an expansion goes to the highest-UCT child.
    `gating`/`target_key` are forwarded to `expand` (model-gated HPs / stage
    locking; `target_key` may be a set of names to lock all but a group of
    choices). `score_cache` memoizes `state_key -> reward`; deterministic
    rollouts make this exact, so a config is evaluated at most once.
    `start_node`, if given, is where selection begins each iteration instead of
    the root — a descendant node lets a bonus phase explore *locally* around
    the incumbent while backprop still updates the whole path up to the root,
    so UCT statistics stay globally consistent.
    `deadline`, if given (an absolute `time.perf_counter()` value), stops the
    loop once wall-clock passes it — `budget` becomes an upper bound and time
    the real cap (whichever is hit first). The score cache is exact, so an early
    stop just yields fewer nodes; determinism/correctness are unaffected. The
    check is between iterations, so a single in-flight rollout can overshoot by
    up to one rollout's wall clock (bounded by `make_rollout_fn`'s `timeout_s`).
    """
    root_state = canonicalize(root_state, gating)
    # First call: create the root node and initialize tried_states with it
    if root is None:
        root = MCTSNode(state=root_state)
    if tried_states is None:
        tried_states = {state_key(root_state)}
    if score_cache is None:
        score_cache = {}

    def scored(state: dict) -> float:
        key = state_key(state)
        if key not in score_cache:
            score_cache[key] = rollout_fn(state)
        return score_cache[key]

    # Storing best configuration and score found during the search
    best_state, best_score = root_state, scored(root_state)

    # MCTS main loop
    descent_root = start_node if start_node is not None else root
    for _ in range(budget):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        # Select a unexpanded node (from `start_node` for a local bonus phase)
        leaf = select(descent_root, c)

        # Expand it if it's not a terminal state (cannot be expanded)
        if leaf.N > 0 or leaf is root:
            children = expand(
                leaf, action_space, tried_states, gating, target_key
            )
            if children:  #
                if prior_fn is not None:
                    prior_fn(children)
                    # roll out the child the prior likes best (mean Q/N; UCT's
                    # exploration term is undefined here — the parent may not
                    # have been backpropagated yet). Without priors keep
                    # first-child order.
                    leaf = max(
                        children,
                        key=lambda n: n.Q / n.N if n.N else float("inf"),
                    )
                else:
                    leaf = children[0]
        # Perform a rollout to get a reward (cached, once per distinct state)
        reward = scored(leaf.state)

        # Backpropagate that reward up the tree.
        backpropagate(leaf, reward)

        # Keep track of the best state and score found during the search.
        if reward > best_score:
            best_score, best_state = reward, leaf.state

    return best_state, best_score, root


def _iter_nodes(root):
    """Depth-first walk over every node in the tree.

    Used by `find_state_node`'s global lookup and by
    `search_loop.tree_action_values`'s tree-mined ablation.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)
