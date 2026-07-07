"""Outer search loop: persisted MCTS + ablation targeting + option injection.

Wraps repeated `mcts.mcts_search` calls with a persisted tree (`root`,
`tried_states`) and a `score_cache`, so the search refines across outer steps
instead of restarting. Two refinements layer on top:

- **Option 1 (no LLM):** after every fixed-budget slice, mine per-stage action
  values from the tree, `pick_target_node`, then run the next slice with
  `target_key` set so expansion focuses on that one stage (the rest stay at
  the incumbent's values). Re-targeting happens between slices by default
  (`retarget=False` keeps the first pick).
- **Option 3 (one LLM call between slices):** before each slice after the
  first, ask a `propose` callable for new operator paths for the currently
  targeted stage, validate + inject them into the spec, **rebuild the plan**
  (skrub has no mid-run splice), and continue on the same tree — so a run can
  keep an option that was not in the original plan.

The `propose` callable is injected (tests pass a fake); the LLM never enters the
inner search loop — at most `outer_steps - 1` calls total.
"""

import inspect
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


def _make_prior_fn(priors: dict[str, dict[str, float]], pseudo_count: int = 1):
    """Pure-lookup expansion prior from the plan's per-option weights.

    Seeds EVERY fresh child (see mcts.mcts_search docstring for why): the
    child's single changed option is looked up in `priors` — the LLM's 0-1
    confidence emitted alongside the plan, zero extra LLM calls — and missing
    options get a neutral 0.5. The pseudo-count keeps the prior weak: one real
    rollout weighs as much as the entire prior.

    Example:
        prior_fn = _make_prior_fn({"model": {"HGB": 0.9, "Ridge": 0.2}})
        prior_fn(children)   # child changing model->HGB enters with Q=0.9, N=1
    """

    def prior_fn(children):
        for child in children:
            parent_state = child.parent.state if child.parent else {}
            weight = 0.5
            for key, value in child.state.items():
                if parent_state.get(key) != value:
                    weight = (priors.get(key) or {}).get(str(value), 0.5)
                    break
            child.Q += weight * pseudo_count
            child.N += pseudo_count

    return prior_fn


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
    elif stage_key.startswith("scope_"):  # a scoped-encoding group
        groups = [dict(g) for g in spec.get("scoped_encodings", [])]
        for g in groups:
            if f"scope_{g['name']}" == stage_key:
                g["options"] = list(g["options"]) + instances
        new["scoped_encodings"] = groups
    else:  # a named post-encoding stage (scale, feature_eng, ...)
        stages = [dict(s) for s in spec.get("stages", [])]
        for s in stages:
            if s.get("name") == stage_key:
                s["options"] = list(s["options"]) + instances
        new["stages"] = stages
    return new


def _label(inst, stage_key: str) -> str:
    """The action-space label the instance will get in `stage_key`.

    Model and scoped-encoding stages use dict-labeled choices (class names);
    list-based stages label options by their repr.
    """
    if stage_key == "model" or stage_key.startswith("scope_"):
        return type(inst).__name__
    return repr(inst)


def _inject(spec, stage_key, proposals, seed, current_labels=(), valid_columns=None):
    """Resolve allowed-list proposals to instances, skipping any already present.

    Returns (new_spec, kept_paths). A proposal is a dotted path, or
    ``{"name": path, "cols": [...]}`` to *scope* the operator to specific
    columns (optionally with ``"additive": true`` and/or
    ``"position": "post_encode"``, which carry onto the new group): when the
    target is the shared ``encoder`` stage, a scoped proposal becomes a NEW
    scoped-encoding group (a new search dimension); when the target is already
    a ``scope_*`` group the operator joins that group (its columns are fixed).
    ``cols`` are validated against ``valid_columns``; a proposal is dropped if
    its path isn't importable / allowlisted (``_make`` returns None) or its
    label is already an option in
    the stage (no duplicate outcomes).
    """
    current = set(current_labels)
    group_names = {g["name"] for g in spec.get("scoped_encodings", [])}
    instances, new_groups, kept = [], [], []
    for prop in proposals:
        extra = prop if isinstance(prop, dict) else {}
        path, cols = (prop, None) if isinstance(prop, str) else (
            prop.get("name"), prop.get("cols")
        )
        inst = spec_resolver._make(path, {}, seed, stage_key)  # constructible + allowlisted
        if inst is None:
            continue
        cols = [c for c in (cols or []) if valid_columns is None or c in valid_columns]
        if cols and not stage_key.startswith("scope_"):
            # scoped proposal on the shared encoder stage -> a new scope group
            name = spec_resolver._sanitize_name(f"{cols[0]}_{type(inst).__name__}")
            if name in group_names:
                continue
            group_names.add(name)
            group = {"name": name, "cols": cols, "options": [inst]}
            if extra.get("position") == "post_encode":
                group["position"] = "post_encode"
            if extra.get("additive") is True:
                group["additive"] = True
            new_groups.append(group)
            kept.append(path)
            continue
        label = _label(inst, stage_key)
        if label in current:
            continue
        current.add(label)
        instances.append(inst)
        kept.append(path)
    if not instances and not new_groups:
        return spec, []
    new = _augment_spec(spec, stage_key, instances) if instances else dict(spec)
    if new_groups:
        new["scoped_encodings"] = list(new.get("scoped_encodings", [])) + new_groups
    return new, kept


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
    retarget: bool = True,
    aux_tables: dict | None = None,
    priors: dict | None = None,
    context_text: str | None = None,
    stratify: bool = False,
) -> dict:
    """Run the persisted, optionally-refined search and return a results dict.

    `outer_steps == 1` is a plain (gated) single MCTS phase. With more steps
    the search runs as fixed-budget SLICES on the same persisted tree,
    interleaved with refinement: after every slice the tree is re-mined and a
    stage is (re-)targeted (Option 1; `retarget=False` keeps the first pick
    for the whole run), and, if `propose` is given, one proposer call injects
    new options into the current target before each subsequent slice
    (Option 3) — so `outer_steps=4, budget_per_step=15` means
    `search 15 → propose → 15 → propose → 15 → propose → 15`, at most
    `outer_steps - 1` LLM calls total. Returns the (possibly rebuilt) plan so
    the caller can score the incumbent on it. For relational tasks pass
    `aux_tables={name: df}` — auxiliary tables join whole (never subsampled)
    and survive the Option-3 plan rebuild.

    `priors` (default: `spec["priors"]`, collected by the resolver from the
    plan's per-option confidence weights) warm-starts freshly expanded
    children via `_make_prior_fn` — zero extra LLM calls. `context_text`
    (typically the data digest) is forwarded to the proposer as part of its
    evidence context along with the cross-stage ledger and the incumbent.
    `stratify=True` for classification (especially imbalanced targets) uses a
    seeded `StratifiedKFold` instead of skrub's plain-`KFold` default, so a
    subsampled fold can't silently land on zero positives (see
    `skrub_ops._cv_kwarg`).
    """

    def _build(current_spec):
        plan = build_staged_plan(current_spec, df, target=target, aux_tables=aux_tables)
        rollout = make_rollout_fn(
            plan, df, seed=seed, scoring=scoring, aux=aux_tables, main_var="data",
            stratify=stratify, target=target,
        )
        return plan, get_action_space(plan), get_choice_gating(plan), rollout

    plan, action_space, gating, rollout = _build(spec)
    priors = priors if priors is not None else spec.get("priors")
    prior_fn = _make_prior_fn(priors) if priors else None
    feature_columns = [c for c in df.columns if c != target]

    start = mcts.canonicalize(get_default_state(plan), gating)
    tried = {mcts.state_key(start)}
    cache: dict = {}
    root = None
    best_state, best_score = start, float("-inf")
    target_key = None
    injected: list[str] = []

    def _proposal_path(p) -> str:
        return p if isinstance(p, str) else p.get("name")

    for step in range(max(1, outer_steps)):
        # Option 3: inject new options for the targeted stage, then rebuild.
        if step >= 1 and propose is not None and target_key and "__" not in target_key:
            values = tree_action_values(root)
            context = {
                "stage_values": {k: v for k, v in values.items() if "__" not in k},
                "incumbent": best_state,
                "incumbent_score": best_score,
                "digest": context_text,
                "columns": feature_columns,
            }
            proposals = _call_propose(
                propose,
                target_key,
                values.get(target_key, {}),
                spec_resolver.allowed_operators(),
                context,
            ) or []
            proposals = [
                p for p in proposals[:n_propose] if _proposal_path(p) not in injected
            ]
            spec, kept = _inject(
                spec,
                target_key,
                proposals,
                seed,
                action_space.get(target_key, []),
                valid_columns=feature_columns,
            )
            if kept:
                plan, action_space, gating, rollout = _build(spec)
                injected += kept

        bstate, bscore, root = mcts.mcts_search(
            start,
            action_space,
            rollout,
            budget=budget_per_step,
            c=c,
            root=root,
            tried_states=tried,
            prior_fn=prior_fn,
            gating=gating,
            target_key=target_key,
            score_cache=cache,
        )
        if bscore > best_score:
            best_state, best_score = bstate, bscore

        # Option 1: after each slice, (re-)pick the stage to focus next on.
        # `retarget=False` keeps the first pick for the whole run (the old
        # fixed-target behavior, kept A/B-able for the sweep harness).
        if outer_steps > 1 and step < outer_steps - 1 and (retarget or target_key is None):
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


# --- the real LLM proposer (one Gemini call between search slices) ------------

import json as _json  # noqa: E402
import re as _re  # noqa: E402

_PATH_RE = _re.compile(r"\b(?:sklearn|skrub)(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")


def _parse_paths(text: str) -> list[str]:
    """Extract dotted operator paths from an LLM reply (JSON array or prose)."""
    return [p for p in _parse_proposals(text) if isinstance(p, str)]


def _parse_proposals(text: str) -> list:
    """Parse proposer output: dotted paths and/or scoped-operator objects.

    Accepts a JSON array whose items are path strings or
    ``{"name": <path>, "cols": [<column names>]}``; falls back to extracting
    dotted sklearn/skrub paths from prose.

    Example:
        _parse_proposals('["skrub.MinHashEncoder", {"name": "skrub.GapEncoder", "cols": ["title"]}]')
        # -> ["skrub.MinHashEncoder", {"name": "skrub.GapEncoder", "cols": ["title"]}]
    """
    if not text:
        return []
    s = _re.sub(r"^```[a-zA-Z0-9]*", "", text.strip()).strip().strip("`").strip()
    try:
        data = _json.loads(s)
        if isinstance(data, list):
            out = []
            for x in data:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, dict) and isinstance(x.get("name"), str):
                    cols = [c for c in (x.get("cols") or []) if isinstance(c, str)]
                    item = {"name": x["name"], "cols": cols}
                    if x.get("position") == "post_encode":
                        item["position"] = "post_encode"
                    if x.get("additive") is True:
                        item["additive"] = True
                    out.append(item)
            return out
    except Exception:
        pass
    return _PATH_RE.findall(s)  # tolerant fallback


def _call_propose(propose, stage_key, ledger, vocab, context):
    """Invoke a proposer, passing `context` only if its signature takes it.

    Keeps the old three-argument proposer protocol working (tests and any
    user-supplied callable) while the built-in proposer receives the richer
    evidence dict (cross-stage ledger, incumbent, data digest).
    """
    try:
        params = inspect.signature(propose).parameters
        wants_context = len(params) >= 4 or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        wants_context = False
    if wants_context:
        return propose(stage_key, ledger, vocab, context)
    return propose(stage_key, ledger, vocab)


def make_llm_proposer(model: str | None = None, n: int = 3, temperature: float = 0.4):
    """Return a `propose(stage_key, ledger, vocab, context) -> [proposals]`
    backed by Gemini.

    One synchronous `google.genai` call between search slices (the LLM never
    enters the inner search loop). The `context` dict (built by
    `run_search_loop`) grounds the proposal in search evidence: the cross-stage
    tree-mined ledger, the incumbent config + score, the data digest, and the
    real column names — so the model proposes against what was actually
    measured, not blind. For encoder/scope stages a proposal may be
    `{"name": path, "cols": [...]}` to scope an operator to specific columns,
    optionally with `"additive": true` (keep the originals) and
    `"position": "post_encode"` (run after vectorization). On any failure
    (quota, parse) it returns `[]`, so the search just continues without
    injection. genai is imported lazily so this module stays importable offline.
    """
    from google import genai  # lazy

    from machine_learning_engineering.shared_libraries import config

    client = genai.Client()  # reads GOOGLE_API_KEY / GOOGLE_GENAI_USE_VERTEXAI
    model = model or config.CONFIG.agent_model

    def propose(stage_key, ledger, vocab, context=None):
        kind = "models" if stage_key == "model" else "transformers"
        allowed = vocab.get(kind, [])
        ctx = context or {}
        scoped_hint = ""
        if stage_key == "encoder" or stage_key.startswith("scope_"):
            scoped_hint = (
                'An item may also be {"name": <path>, "cols": [<exact column '
                "names>]} to scope an operator to specific columns; add "
                '"additive": true when the ORIGINAL columns must be kept '
                "alongside the derived output (e.g. date-part extraction), "
                'and "position": "post_encode" to run after vectorization '
                "(numeric columns only). Known "
                f"columns: {ctx.get('columns', [])}.\n"
            )
        digest = ctx.get("digest")
        prompt = (
            "You improve ONE stage of a tabular ML pipeline by proposing NEW "
            "operators to try next. Output ONLY a JSON array of full dotted import "
            f'paths (at most {n}), e.g. ["sklearn.preprocessing.RobustScaler"].\n'
            + scoped_hint
            + f"\nStage to improve: {stage_key!r} (kind: {kind}).\n"
            f"Options tried at this stage (label -> mean CV reward): {ledger}.\n"
            f"Evidence from the whole search tree (per stage, option -> mean "
            f"reward): {ctx.get('stage_values', {})}.\n"
            f"Current best config (reward {ctx.get('incumbent_score')}): "
            f"{ctx.get('incumbent')}.\n"
            + (f"Data digest:\n{digest}\n" if digest else "")
            + f"Propose only NEW sklearn.* or skrub.* operators (not already tried) "
            f"that are likely to beat what was tried, suited to this stage. "
            f"Prefer these known-good ones: {allowed}."
        )
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temperature, "max_output_tokens": 512},
            )
            return _parse_proposals(resp.text)
        except Exception:
            return []

    return propose
