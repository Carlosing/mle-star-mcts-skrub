"""Outer search loop: persisted MCTS + ablation targeting + option injection.

Wraps repeated `mcts.mcts_search` calls with a persisted tree (`root`,
`tried_states`) and a `score_cache`, so the search refines across outer steps
instead of restarting. Two refinements layer on top:

- **Option 1 (no LLM):** after a broad first phase, mine per-stage action values
  from the tree, `pick_target_node`, then run a phase with `target_key` set so
  expansion focuses on that one stage (the rest stay at the incumbent's values).
- **Option 3 (one LLM call / outer step):** ask a `propose` callable for new
  operator paths for the targeted stage, validate + inject them into the spec,
  **rebuild the plan** (skrub has no mid-run splice), and continue on the same
  tree — so a run can keep an option that was not in the original plan.

The `propose` callable is injected (tests pass a fake); the LLM never enters the
inner search loop — at most `outer_steps` calls total.
"""

from collections import defaultdict

from machine_learning_engineering import mcts, spec_resolver
from machine_learning_engineering.skrub_ops import (
    build_staged_plan,
    get_action_space,
    get_choice_gating,
    get_default_state,
    make_rollout_fn,
    pick_target_node,
)


def tree_action_values(root) -> dict[str, dict]:
    """Mine `{choice_name: {option: mean_reward}}` from the persisted tree.

    Each visited non-root node is attributed to the single choice it edited
    relative to its parent; the node's mean reward (`Q/N`) is aggregated per
    (choice, option). Feeds `pick_target_node` (highest-variance stage wins).
    """
    agg: dict = defaultdict(lambda: defaultdict(list))
    for node in mcts._iter_nodes(root):
        if node.parent is None or node.N == 0:
            continue
        for key in set(node.state) | set(node.parent.state):
            if node.state.get(key) != node.parent.state.get(key) and key in node.state:
                agg[key][node.state[key]].append(node.Q / node.N)
    return {
        key: {opt: sum(vals) / len(vals) for opt, vals in opts.items()}
        for key, opts in agg.items()
    }


def _augment_spec(spec: dict, stage_key: str, instances: list) -> dict:
    """Return a copy of `spec` with new operator instances added to a stage."""
    new = dict(spec)
    if stage_key == "model":
        new["model"] = dict(spec.get("model", {}))
        for inst in instances:
            new["model"][type(inst).__name__] = inst
    elif stage_key == "encoder":
        new["encoder_options"] = list(spec.get("encoder_options", [])) + instances
    elif stage_key == "clean":
        new["clean_options"] = list(spec.get("clean_options", [])) + instances
    else:  # a named post-encoding stage (scale, feature_eng, ...)
        stages = [dict(s) for s in spec.get("stages", [])]
        for s in stages:
            if s.get("name") == stage_key:
                s["options"] = list(s["options"]) + instances
        new["stages"] = stages
    return new


def _label(inst, stage_key: str) -> str:
    """The action-space label the instance will get in `stage_key`."""
    return type(inst).__name__ if stage_key == "model" else repr(inst)


def _inject(spec, stage_key, proposed_paths, seed, current_labels=()):
    """Resolve allowed-list paths to instances, skipping any already present.

    Returns (new_spec, kept_paths). A path is dropped if it isn't importable /
    allowlisted (``_make`` returns None) or if its label is already an option in
    the stage (no duplicate outcomes).
    """
    current = set(current_labels)
    instances, kept = [], []
    for path in proposed_paths:
        inst = spec_resolver._make(path, {}, seed, stage_key)  # constructible + allowlisted
        if inst is None:
            continue
        label = _label(inst, stage_key)
        if label in current:
            continue
        current.add(label)
        instances.append(inst)
        kept.append(path)
    if not instances:
        return spec, []
    return _augment_spec(spec, stage_key, instances), kept


def run_search_loop(
    spec: dict,
    df,
    target: str,
    scoring: str | None = None,
    *,
    outer_steps: int = 1,
    budget_per_step: int = 20,
    c: float = 0.5,
    seed: int = 42,
    propose=None,
    n_propose: int = 3,
) -> dict:
    """Run the persisted, optionally-refined search and return a results dict.

    `outer_steps == 1` is a plain (gated) single MCTS phase. With more steps it
    targets a stage (Option 1) and, if `propose` is given, injects new options
    into that stage each subsequent step (Option 3). Returns the (possibly
    rebuilt) plan so the caller can score the incumbent on it.
    """
    plan = build_staged_plan(spec, df, target=target)
    action_space = get_action_space(plan)
    gating = get_choice_gating(plan)
    rollout = make_rollout_fn(plan, df, seed=seed, scoring=scoring)

    start = mcts.canonicalize(get_default_state(plan), gating)
    tried = {mcts.state_key(start)}
    cache: dict = {}
    root = None
    best_state, best_score = start, float("-inf")
    target_key = None
    injected: list[str] = []

    for step in range(max(1, outer_steps)):
        # Option 3: inject new options for the targeted stage, then rebuild.
        if step >= 1 and propose is not None and target_key and "__" not in target_key:
            ledger = tree_action_values(root).get(target_key, {})
            paths = propose(target_key, ledger, spec_resolver.allowed_operators()) or []
            paths = [p for p in paths[:n_propose] if p not in injected]
            spec, kept = _inject(
                spec, target_key, paths, seed, action_space.get(target_key, [])
            )
            if kept:
                plan = build_staged_plan(spec, df, target=target)
                action_space = get_action_space(plan)
                gating = get_choice_gating(plan)
                rollout = make_rollout_fn(plan, df, seed=seed, scoring=scoring)
                injected += kept

        bstate, bscore, root = mcts.mcts_search(
            start,
            action_space,
            rollout,
            budget=budget_per_step,
            c=c,
            root=root,
            tried_states=tried,
            gating=gating,
            target_key=target_key,
            score_cache=cache,
        )
        if bscore > best_score:
            best_state, best_score = bstate, bscore

        # Option 1: after the broad first phase, pick a stage to focus on.
        if step == 0 and outer_steps > 1:
            ledger = {
                k: v for k, v in tree_action_values(root).items() if "__" not in k
            }
            if ledger:
                try:
                    target_key = pick_target_node(ledger)
                except Exception:
                    target_key = None

    return {
        "best_state": best_state,
        "best_score": best_score,
        "root": root,
        "plan": plan,
        "action_space": action_space,
        "tried_states": tried,
        "score_cache": cache,
        "target_key": target_key,
        "injected_options": injected,
    }


# --- the real LLM proposer (one Gemini call per outer step) -------------------

import json as _json  # noqa: E402
import re as _re  # noqa: E402

_PATH_RE = _re.compile(r"\b(?:sklearn|skrub)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")


def _parse_paths(text: str) -> list[str]:
    """Extract dotted operator paths from an LLM reply (JSON array or prose)."""
    if not text:
        return []
    s = _re.sub(r"^```[a-zA-Z0-9]*", "", text.strip()).strip().strip("`").strip()
    try:
        data = _json.loads(s)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, str)]
    except Exception:
        pass
    return _PATH_RE.findall(s)  # tolerant fallback


def make_llm_proposer(model: str | None = None, n: int = 3, temperature: float = 0.4):
    """Return a `propose(stage_key, ledger, vocab) -> [paths]` backed by Gemini.

    One synchronous `google.genai` call per outer step (the LLM never enters the
    inner search loop). On any failure (quota, parse) it returns `[]`, so the
    search just continues without injection. genai is imported lazily so this
    module stays importable offline.
    """
    from google import genai  # lazy

    from machine_learning_engineering.shared_libraries import config

    client = genai.Client()  # reads GOOGLE_API_KEY / GOOGLE_GENAI_USE_VERTEXAI
    model = model or config.CONFIG.agent_model

    def propose(stage_key, ledger, vocab):
        kind = "models" if stage_key == "model" else "transformers"
        allowed = vocab.get(kind, [])
        prompt = (
            "You improve ONE stage of a tabular ML pipeline by proposing NEW "
            "operators to try next. Output ONLY a JSON array of full dotted import "
            f'paths (at most {n}), e.g. ["sklearn.preprocessing.RobustScaler"].\n\n'
            f"Stage: {stage_key!r} (kind: {kind}).\n"
            f"Operators already tried (label -> mean CV score): {ledger}.\n"
            f"Propose only NEW sklearn.* or skrub.* operators (not already tried) "
            f"that are likely to beat what was tried, suited to this stage. "
            f"Prefer these known-good ones: {allowed}."
        )
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temperature, "max_output_tokens": 256},
            )
            return _parse_paths(resp.text)
        except Exception:
            return []

    return propose
