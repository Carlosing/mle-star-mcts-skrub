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


def select(node: MCTSNode, c: float = 0.5) -> MCTSNode:
    """Descend the tree by UCT until a leaf is reached."""
    # If node is not expanded (i.e., has no children), return it as the selected node
    # Otherwise, select the child with the highest UCT value and continue descending
    while node.children:
        node = max(node.children, key=lambda n: n.uct(c))
    return node


def expand(
    node: MCTSNode,
    action_space: dict[str, list],
    tried_states: set[tuple],
) -> list[MCTSNode]:
    """Generate children by swapping one choice value at a time.

    `action_space` maps choice name -> list of options. It comes from
    skrub's describe_param_grid() — never from an LLM (anti-pattern #1).
    """
    children = []
    for choice_name, options in action_space.items():
        for option in options:
            new_state = {**node.state, choice_name: option}
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
) -> tuple[dict, float, MCTSNode]:
    """Run `budget` MCTS iterations and return (best_state, best_score, root).

    Pass the returned `root` and `tried_states` back in on the next outer-loop
    step so UCT statistics accumulate (anti-pattern #7: never reset the tree).

    `rollout_fn(state) -> float` must return a reward in [0, 1] and never
    raise (failed configs score 0.0, see skrub_ops.make_rollout_fn).

    `prior_fn`, if given, is called on freshly expanded children and may
    warm-start their Q/N (the optional LLM expansion prior, brief §6).
    """
    # First call: create the root node and initialize tried_states with it
    if root is None:
        root = MCTSNode(state=root_state)
    if tried_states is None:
        tried_states = {state_key(root_state)}

    # Storing best configuration and score found during the search
    # Initialized to root's state and its rollout score
    best_state, best_score = root_state, rollout_fn(root_state)

    # MCTS main loop
    for _ in range(budget):
        # Select a unexpanded node
        leaf = select(root, c)

        # Expand it if it's not a terminal state (cannot be expanded)
        if leaf.N > 0 or leaf is root:
            children = expand(leaf, action_space, tried_states)
            if children:  #
                if prior_fn is not None:
                    prior_fn(children)
                leaf = children[0]
        # Perform a rollout to get a reward
        reward = rollout_fn(leaf.state)

        # Backpropagate that reward up the tree.
        backpropagate(leaf, reward)

        # Keep track of the best state and score found during the search.
        if reward > best_score:
            best_score, best_state = reward, leaf.state

    return best_state, best_score, root


# ---------------------------------------------------------------------------
# Visualization — store the search tree as Graphviz DOT (text) or ASCII.
# Edges carry the single-choice change; nodes carry the final UCT statistics.
# ---------------------------------------------------------------------------


def _iter_nodes(root):
    """Depth-first walk over every node in the tree."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


def edge_change(parent, child) -> str:
    """The choice change from `parent` to `child`, e.g. "model: GBM -> RF".

    Children are single-edit neighbors, so this is normally one entry. A key
    that the parent did not have (a newly decided staged stage) shows '·' as
    the previous value.
    """
    changes = []
    for key in sorted(set(parent.state) | set(child.state)):
        before, after = parent.state.get(key, "·"), child.state.get(key, "·")
        if before != after:
            changes.append(f"{key}: {before} -> {after}")
    return ", ".join(changes) if changes else "(no change)"


def _stats(node, c: float) -> str:
    """One-line node statistics: visits, cumulative reward, mean, final UCT."""
    if node.N == 0:
        return "unvisited"
    line = f"N={node.N} Q={node.Q:.2f} mean={node.Q / node.N:.3f}"
    if node.parent is not None:  # UCT is undefined at the root (no parent)
        line += f" UCT={node.uct(c):.3f}"
    return line


def to_dot(root, c: float = 0.5, max_edge_label: int = 40) -> str:
    """Render the search tree as a Graphviz DOT string.

    Nodes are labeled with N / Q / mean / final UCT; edges with the change that
    produced the child. Store or view it, e.g.:

        with open("tree.dot", "w") as f:
            f.write(to_dot(root))
        # shell:     dot -Tpng tree.dot -o tree.png
        # notebook:  import graphviz; graphviz.Source(to_dot(root))
    """
    nodes = list(_iter_nodes(root))
    ids = {id(n): f"n{i}" for i, n in enumerate(nodes)}
    lines = [
        "digraph mcts {",
        "  rankdir=TB;",
        '  node [shape=box, fontname="monospace"];',
    ]
    for node in nodes:
        tag = "ROOT" if node.parent is None else "node"
        label = f"{tag}\\n{_stats(node, c)}".replace('"', "'")
        style = ', style=filled, fillcolor="lightgray"' if node.parent is None else ""
        lines.append(f'  {ids[id(node)]} [label="{label}"{style}];')
        for child in node.children:
            elabel = edge_change(node, child)
            if len(elabel) > max_edge_label:
                elabel = elabel[: max_edge_label - 1] + "…"
            lines.append(
                f'  {ids[id(node)]} -> {ids[id(child)]} [label="{elabel.replace(chr(34), chr(39))}"];'
            )
    lines.append("}")
    return "\n".join(lines)


def print_tree(root, c: float = 0.5) -> str:
    """Zero-dependency ASCII view of the tree (returns a string).

    Each line shows the edge change into a node and its N / mean / UCT.
    Children are listed most-visited first.
    """
    out = []

    def walk(node, prefix):
        head = "ROOT" if node.parent is None else edge_change(node.parent, node)
        out.append(f"{prefix}{head}  [{_stats(node, c)}]")
        for child in sorted(node.children, key=lambda n: -n.N):
            walk(child, prefix + "  ")

    walk(root, "")
    return "\n".join(out)
