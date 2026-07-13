"""Outer search loop: persisted MCTS + ablation targeting + option injection.

Wraps repeated `mcts.mcts_search` calls with a persisted tree (`root`,
`tried_states`) and a `score_cache`, so the search refines across outer steps
instead of restarting. Two refinements layer on top:

- After every fixed-budget slice, mine per-stage action values from the tree
  and `pick_target_node` a focus stage — passed to the proposer as
  `target_stage`, a HINT about where the search sees the most unresolved
  variance. It does NOT constrain expansion: replay A/Bs showed the variance
  ledger elects `model` on essentially every pick, so locking the other
  stages starved off-target injected options until the bonus phase. Re-picking
  happens between slices by default (`retarget=False` keeps the first pick).
- **Option 3 (one LLM call between slices):** before each slice after the
  first, hand the `propose` callable the WHOLE current raw plan plus search
  evidence and expect back an *extended* plan (any/multiple stages, with HP
  ranges). Only additions survive: `_merge_raw_plans` unions the two plans by
  operator path and never touches an existing entry, so every state in the
  persisted tree stays appliable. The merged plan is re-resolved through
  `resolve_spec` (the import allow-list stays the safety envelope), the plan
  is **rebuilt** (skrub has no mid-run splice), and search continues on the
  same tree — so a run can keep an option that was not in the original plan,
  and an injected operator arrives *tuned* (its HP ranges enter the action
  space), not as a bare default.
- **Focused-refinement bonus phase (no LLM):** after the whole budget is
  spent, run one final `ceil(total / 4)` slice that starts selection at the
  incumbent node and explores ALL of its single-edit neighbors — encoder,
  scale, assemble and hyperparameters alike (UCT's own "untested ⇒ ∞" rule
  prioritizes the never-tried edits). This is the biased exploration around
  the best config that stage-targeting was originally for; an earlier HP-only
  variant no-op'd whenever the incumbent's HP grid was exhausted, which at
  larger budgets was every run. Off switch: `refinement_phase=False`.

The `propose` callable is injected (tests pass a fake); the LLM never enters the
inner search loop — at most `outer_steps - 1` calls total. The bonus phase adds
zero LLM calls.
"""

import json
import time
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
            if (
                node.state.get(key) != node.parent.state.get(key)
                and key in node.state
            ):
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


def _entry_path(entry):
    """The operator path identifying a raw-plan option (or None)."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("name")
    return None


def _tuned_params(entry) -> set:
    """The hyperparameter names an option entry tunes (empty for a bare path)."""
    if isinstance(entry, dict):
        return set((entry.get("params") or {}).keys())
    return set()


def _entry_signature(entry) -> tuple:
    """An option's full identity: its path plus a canonical view of its params.

    Two entries share a signature only when they are the same operator tuned
    identically — so a bare operator and a tuned re-proposal of it are DISTINCT.
    That is exactly what lets the tuned variant be ADDED as a sibling instead of
    collapsed onto the bare one (which path-only dedup did, silently dropping it).
    """
    params = entry.get("params") if isinstance(entry, dict) else None
    return (_entry_path(entry), json.dumps(params or {}, sort_keys=True))


def _merge_option_lists(old: list, new, repr_labeled: bool = True) -> list:
    """Union two raw option lists; existing entries are never modified.

    On ``repr_labeled`` stages (scoped options and backbone slots, whose
    action-space label is the operator *repr*) dedup is by full signature (path +
    tuned-param rules), not path alone, so a proposer that re-emits an operator
    carrying NEW hyperparameters — a bare ``skrub.TextEncoder`` gaining an
    ``n_components`` range — is kept as a distinct sibling rather than silently
    dropped (the bare and tuned reprs are different options).

    The top-level model list is NOT repr-labeled: ``resolve_spec`` keys models on
    ``type(inst).__name__``, so two same-class entries collapse onto one at
    resolve (last wins). There (``repr_labeled=False``) the original path-only
    dedup holds — a new model class is added, a re-proposal of an existing class
    is dropped — since a same-class sibling cannot be represented anyway.

    Numeric re-tuning is the one thing kept out of repr-labeled stages too. A
    resolved param becomes a ``choose_*`` node named ``<stage>__<Class>__<param>``,
    so a second same-path entry that RE-tunes a param already tuned upstream
    (MinHash n_components ``[50, 200]`` -> ``[50, 300]``) would mint a duplicate
    node name and make the action space — which persisted tree states key on —
    unreadable. Params whose node name is already claimed are stripped from the
    newcomer (keeping only genuinely new dimensions); an exact re-send is
    dropped before stripping, so an unchanged proposal is always a no-op.

    Example:
        _merge_option_lists(
            ["skrub.TextEncoder"],
            [{"name": "skrub.TextEncoder",
              "params": {"n_components": {"int": [50, 300]}}}])
        # -> ["skrub.TextEncoder",
        #     {"name": "skrub.TextEncoder", "params": {"n_components": ...}}]
    """
    if not repr_labeled:  # class-name-labeled (models): dedup by path only
        seen_paths = {_entry_path(e) for e in old}
        added = []
        for e in new or []:
            path = _entry_path(e)
            if path and path not in seen_paths:
                seen_paths.add(path)
                added.append(e)
        return list(old) + added

    seen = {_entry_signature(e) for e in old}
    tuned: dict = defaultdict(set)  # path -> param names already given a node
    for e in old:
        tuned[_entry_path(e)] |= _tuned_params(e)
    added = []
    for e in new or []:
        path = _entry_path(e)
        if path is None:
            continue
        if _entry_signature(e) in seen:
            continue  # exact re-send of an existing option
        collide = _tuned_params(e) & tuned[path]
        if collide:  # drop params whose choose-node name is already claimed
            e = dict(e)
            e["params"] = {
                k: v
                for k, v in (e.get("params") or {}).items()
                if k not in collide
            }
            if not e["params"]:
                del e["params"]
            if _entry_signature(e) in seen:
                continue  # collapsed onto an option already present
        added.append(e)
        seen.add(_entry_signature(e))
        tuned[path] |= _tuned_params(e)
    return list(old) + added


def _merge_raw_plans(old: dict, new: dict) -> dict:
    """Merge a proposed plan into the current raw plan, strictly additively.

    Each stage is unioned by operator signature (path + tuned params) for
    `model`, by entry name for `stages`, `scoped_encodings`, `assemble` (new
    names are appended whole, matching names only gain options), and by
    additive param/slot union for the `cleaner` / `vectorizer` backbones
    (`_merge_backbone`). Existing entries are NEVER modified or removed: the
    persisted tree's states reference the current action-space labels, so a
    mutation would orphan already-scored nodes. A re-proposal that only ADDS
    hyperparameters to an existing operator therefore enters as a distinct
    sibling (`_merge_option_lists`), never as an in-place edit. A partial
    proposal is fine — missing stages mean
    "no change". Validation is not done here: the merged plan goes through
    `resolve_spec`, whose import allow-list / schema checks drop anything
    unusable.

    Example:
        _merge_raw_plans(
            {"model": ["sklearn.linear_model.Ridge"]},
            {"model": [{"name": "lightgbm.LGBMRegressor",
                        "params": {"n_estimators": {"int": [100, 600]}}}]})
        # -> {"model": ["sklearn.linear_model.Ridge",
        #               {"name": "lightgbm.LGBMRegressor", "params": {...}}]}
    """
    merged = dict(old)
    if new.get("model"):
        old_models = spec_resolver._iter_model_items(old.get("model"))
        new_models = spec_resolver._iter_model_items(new["model"])
        merged["model"] = _merge_option_lists(
            old_models, new_models, repr_labeled=False
        )
    for key in ("stages", "scoped_encodings", "assemble"):
        if not new.get(key):
            continue
        entries = [dict(e) for e in (old.get(key) or [])]
        by_name = {e.get("name"): e for e in entries}
        for prop in new[key]:
            if not isinstance(prop, dict):
                continue
            current = by_name.get(prop.get("name"))
            if current is None:  # a whole new stage / group / join config
                entries.append(prop)
                by_name[prop.get("name")] = prop
            elif key != "assemble" and prop.get("options"):
                # same name -> only its option menu may grow
                current["options"] = _merge_option_lists(
                    list(current.get("options") or []), prop["options"]
                )
        merged[key] = entries
    for key in ("cleaner", "vectorizer"):
        if new.get(key):
            merged[key] = _merge_backbone(old.get(key) or {}, new[key])
    return merged


def _merge_backbone(old: dict, new) -> dict:
    """Additively union a ``cleaner``/``vectorizer`` backbone spec.

    New scalar ``params`` names are added; an existing param's rule is NEVER
    changed (a persisted tree state keys on its current choose-node range, so a
    re-tune would orphan scored nodes — same invariant as `_merge_option_lists`).
    Each ``slots`` option list is unioned by operator path.
    """
    if not isinstance(new, dict):
        return old if isinstance(old, dict) else {}
    merged = dict(old) if isinstance(old, dict) else {}
    params = dict(merged.get("params") or {})
    for pname, rule in (new.get("params") or {}).items():
        params.setdefault(pname, rule)  # existing param never changes
    if params:
        merged["params"] = params
    slots = dict(merged.get("slots") or {})
    for slot, items in (new.get("slots") or {}).items():
        slots[slot] = _merge_option_lists(list(slots.get(slot) or []), items)
    if slots:
        merged["slots"] = slots
    return merged


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
    raw_spec: dict | None = None,
    resolve=None,
    retarget: bool = True,
    aux_tables: dict | None = None,
    priors: dict | None = None,
    context_text: str | None = None,
    stratify: bool = False,
    refinement_phase: bool = True,
    timeout_s: float | None = 60.0,
    n_subsample_seeds: int = 1,
    n_jobs: int = 6,
    time_budget_s: float | None = None,
) -> dict:
    """Run the persisted, optionally-refined search and return a results dict.

    `outer_steps == 1` is a plain (gated) single MCTS phase. With more steps
    the search runs as fixed-budget SLICES on the same persisted tree,
    interleaved with refinement: after every slice the tree is re-mined and a
    focus stage is (re-)picked (Option 1; `retarget=False` keeps the first
    pick for the whole run) — forwarded to the proposer as its `target_stage`
    hint, never as an expansion lock — and, if `propose` is given, one
    proposer call fires before each subsequent slice (Option 3) — so
    `outer_steps=4, budget_per_step=15` means `search 15 → propose → 15 →
    propose → 15 → propose → 15`, at most `outer_steps - 1` LLM calls total.
    Returns the
    (possibly rebuilt) plan so the caller can score the incumbent on it. For
    relational tasks pass `aux_tables={name: df}` — auxiliary tables join
    whole (never subsampled) and survive the Option-3 plan rebuild.

    Option 3 needs three pieces: `propose(plan_json, context) -> dict | None`
    (returns a whole raw plan that EXTENDS `plan_json`; any/multiple stages,
    with HP ranges), `raw_spec` (the current plan as raw JSON dict — the
    thing the proposer sees and extends) and `resolve` (callable raw dict ->
    resolved spec, typically a `resolve_spec` closure carrying task_type /
    aux_schemas / main_columns). The returned plan is merged strictly
    additively (`_merge_raw_plans`), re-resolved, and the plan rebuilt; the
    action-space diff is recorded in `injected_options` (`"key"` for a whole
    new dimension, `"key:label"` for a new option in an existing one). A
    proposal that fails to parse/resolve, or adds nothing new, is skipped and
    the search continues unchanged. Without `raw_spec`/`resolve`, `propose`
    is never called.

    `priors` (default: `spec["priors"]`, collected by the resolver from the
    plan's per-option confidence weights) warm-starts freshly expanded
    children via `_make_prior_fn` — zero extra LLM calls. `context_text`
    (typically the data digest) is forwarded to the proposer as part of its
    evidence context along with the cross-stage ledger and the incumbent.
    `stratify=True` for classification (especially imbalanced targets) uses a
    seeded `StratifiedKFold` instead of skrub's plain-`KFold` default, so a
    subsampled fold can't silently land on zero positives (see
    `skrub_ops._cv_kwarg`).

    `refinement_phase=True` (default) appends a focused-refinement bonus phase
    after the main budget is spent: `ceil(total / 4)` extra rollouts that
    start selection at the incumbent node and explore ALL of its single-edit
    neighbors — structural stages (encoder/scale/assemble) and gated HPs
    alike — UCT and backprop unchanged. This is the targeting the design
    originally aimed at: biased exploration around the best config. (An
    earlier HP-only variant, `hp_refine`, no-op'd whenever the incumbent's HP
    grid was exhausted.) The dims actually edited during the phase are
    returned in `refined_dims`.

    `timeout_s` (default 60s) caps each rollout's wall clock: a config whose CV
    runs longer scores 0.0, so a free-form HP range that makes one fit
    pathologically slow can't stall the search (see `skrub_ops._time_limit`).

    `n_subsample_seeds > 1` averages every rollout's reward over that many
    seeded subsamples (pure code, no LLM; see `make_rollout_fn`) — the
    denoiser for imbalanced tasks where single-draw rewards are too noisy for
    UCT to converge on. The per-rollout wall-clock cap scales with it.

    `time_budget_s`, if given, caps the whole search by wall clock: each MCTS
    slice stops at the shared deadline (`budget_per_step` becomes an upper
    bound), the outer-slice loop breaks once the deadline passes, and the
    focused-refinement bonus phase is skipped when no time remains. This is how
    a run fills a fixed compute budget (e.g. 1 hour) at CONSTANT LLM cost — the
    time-vs-quality axis of the benchmark. `None` keeps the pure rollout-count
    budget. The two combine: whichever cap (rollouts or time) is hit first wins.
    """
    deadline = (
        time.perf_counter() + time_budget_s
        if time_budget_s is not None
        else None
    )

    def _build(current_spec):
        plan = build_staged_plan(
            current_spec, df, target=target, aux_tables=aux_tables
        )
        # Fold parallelism is inter-process (joblib forks CV folds into separate
        # address spaces), which is orthogonal to the intra-process OpenMP
        # double-init that segfaults boosters — that trigger is the ESTIMATOR's
        # own n_jobs, pinned to 1 in REGISTRY. Verified safe over 80 booster
        # rollouts at CV n_jobs=6, so booster plans keep the full parallelism.
        rollout = make_rollout_fn(
            plan,
            df,
            seed=seed,
            scoring=scoring,
            aux=aux_tables,
            main_var="data",
            stratify=stratify,
            target=target,
            timeout_s=timeout_s,
            n_subsample_seeds=n_subsample_seeds,
            n_jobs=n_jobs,
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
    refined_dims: list[str] = []
    proposal_injection_error: str | None = None
    explicit_priors = priors is not None

    for step in range(max(1, outer_steps)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        # Option 3: ask for a whole extended plan, merge additively, rebuild.
        if (
            step >= 1
            and propose is not None
            and raw_spec is not None
            and resolve is not None
        ):
            values = tree_action_values(root)
            context = {
                "stage_values": {
                    k: v for k, v in values.items() if "__" not in k
                },
                "incumbent": best_state,
                "incumbent_score": best_score,
                "digest": context_text,
                "columns": feature_columns,
                "target_stage": target_key,
                "vocab": spec_resolver.allowed_operators(),
            }
            try:
                proposal = propose(raw_spec, context)
            except Exception:
                proposal = None
            merged = (
                _merge_raw_plans(raw_spec, proposal)
                if isinstance(proposal, dict)
                else raw_spec
            )
            if merged != raw_spec:
                try:
                    new_spec = resolve(merged)
                    rebuilt = _build(new_spec)
                except Exception as exc:
                    # a proposal that merges but won't resolve/build is dropped
                    # so the search keeps going — but record WHY, so an empty
                    # `injected_options` from a failed injection is
                    # distinguishable from one where the proposer added nothing.
                    proposal_injection_error = f"{type(exc).__name__}: {exc}"[
                        :300
                    ]
                else:
                    old_space = action_space
                    spec, raw_spec = new_spec, merged
                    plan, action_space, gating, rollout = rebuilt
                    injected += [
                        key
                        for key in action_space
                        if key not in old_space  # a whole new dimension
                    ] + [
                        f"{key}:{label}"
                        for key, labels in action_space.items()
                        if key in old_space
                        for label in labels
                        if label not in old_space[key]
                    ]
                    if not explicit_priors and new_spec.get("priors"):
                        # injected entries may carry priors of their own
                        prior_fn = _make_prior_fn(new_spec["priors"])

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
            score_cache=cache,
            deadline=deadline,
        )
        if bscore > best_score:
            best_state, best_score = bstate, bscore

        # Option 1: after each slice, (re-)pick the focus stage. This is a
        # proposer hint only (`target_stage` in its context) — expansion stays
        # unlocked, since the variance ledger elects `model` on essentially
        # every pick and a lock starved off-target injections until the bonus
        # phase. `retarget=False` keeps the first pick for the whole run.
        if (
            outer_steps > 1
            and step < outer_steps - 1
            and (retarget or target_key is None)
        ):
            ledger = {
                k: v
                for k, v in tree_action_values(root).items()
                if "__" not in k
            }
            if ledger:
                try:
                    target_key = pick_target_node(ledger)
                except Exception:
                    target_key = None

    # Focused-refinement bonus phase: spend ceil(total/4) extra rollouts on
    # ALL single-edit neighbors of the incumbent (structure and HPs alike),
    # starting selection at the incumbent node (local exploration; UCT and
    # backprop are unchanged).
    if refinement_phase and not (
        deadline is not None and time.perf_counter() >= deadline
    ):
        best_node = mcts.find_state_node(root, best_state)
        if best_node is not None:
            incumbent = dict(best_state)
            before = set(tried)
            bonus = -(
                -max(1, outer_steps) * budget_per_step // 4
            )  # ceil(total/4)
            bstate, bscore, root = mcts.mcts_search(
                start,
                action_space,
                rollout,
                budget=bonus,
                c=c,
                root=root,
                tried_states=tried,
                prior_fn=prior_fn,
                gating=gating,
                score_cache=cache,
                start_node=best_node,
                deadline=deadline,
            )
            if bscore > best_score:
                best_state, best_score = bstate, bscore
            # the dims the phase actually edited: where a state first tried
            # during the phase differs from the incumbent it started from
            refined_dims = sorted(
                {
                    k
                    for key in tried - before
                    for k in set(dict(key)) | set(incumbent)
                    if dict(key).get(k) != incumbent.get(k)
                }
            )

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
        "refined_dims": refined_dims,
        "proposal_injection_error": proposal_injection_error,
    }


# --- the real LLM proposer (one provider call between search slices) ----------

import json as _json  # noqa: E402


def _parse_plan(text: str) -> dict | None:
    """Parse proposer output into a raw plan dict, or None if unusable.

    Example:
        _parse_plan('```json\\n{"model": ["lightgbm.LGBMRegressor"]}\\n```')
        # -> {"model": ["lightgbm.LGBMRegressor"]}
        _parse_plan("sorry, no JSON")  # -> None
    """
    if not text:
        return None
    try:
        plan = spec_resolver.parse_spec_json(text)
        return plan if isinstance(plan, dict) else None
    except Exception:
        return None


def make_llm_proposer(
    model: str | None = None,
    temperature: float = 0.4,
    log_dir: str | None = None,
):
    """Return `propose(plan_json, context) -> dict | None` for the active provider.

    Native Gemini (`gemini-*`, with `google_search` grounding) or any
    OpenAI-compatible endpoint (school), selected exactly like the ADK agents —
    so `PROVIDER=school` gets Option 3 too, no code change.

    One `google.genai` call between search slices (the LLM never enters the
    inner search loop). The proposer sees the WHOLE current raw plan plus the
    search evidence in `context` (built by `run_search_loop`: the cross-stage
    tree-mined ledger, the incumbent config + score, the data digest, the real
    column names, the currently targeted stage and the allowed-operator
    vocabulary) and returns a raw plan that EXTENDS it — new operators on any
    stage, new scoped groups, and hyperparameter ranges on every new entry, so
    an injected operator competes tuned rather than at bare defaults. Only
    additions are applied (`_merge_raw_plans`); the merged plan is re-resolved
    through the import allow-list. On any failure (quota, parse) it returns
    None, so the search just continues without injection. genai is imported
    lazily so this module stays importable offline.

    Web search is attached (Gemini-native) with the SAME safety net as the
    analyst / plan_author, now symmetric across both directions: search is tried
    first, and if a grounded call yields no usable proposals (google_search
    grounding intermittently eats the output), the call is retried once with
    search OFF before giving up.

    `log_dir`: if given, each proposer call's prompt + output is written to
    `<log_dir>/proposer_<k>_request.json` / `_response.json`, numbered by call
    order (first call -> 1), mirroring the ADK agents' `<agent>_<phase>.json`
    logs. A response file holds one record, or two when the search-off retry
    fires (the grounded attempt, then the retry).
    """
    import json
    import os
    from datetime import datetime, timezone

    from machine_learning_engineering.shared_libraries import config

    model = model or config.CONFIG.agent_model
    # Provider-agnostic, mirroring the analyst/plan_author agents: a `gemini-*`
    # id calls native Gemini (with the Gemini-only `google_search` grounding);
    # any other id (school / OpenAI-compatible) calls that endpoint via the
    # OpenAI client, no web search. Option 3 follows PROVIDER with no code
    # change. Clients imported lazily so the module stays importable offline.
    # lazy so the logic layer stays importable offline (run_logging pulls in
    # google.adk at import); only reached when a proposer is actually built.
    from machine_learning_engineering.run_logging import extract_usage

    # per-proposer token accounting, exposed as `propose.token_totals` so the
    # pipeline can report Option-3's LLM cost without changing propose()'s
    # signature. `calls` counts real completions (both attempts of a retry).
    token_totals = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}

    def _tally(resp) -> None:
        usage = extract_usage(resp)
        token_totals["prompt"] += usage["prompt"]
        token_totals["completion"] += usage["completion"]
        token_totals["total"] += usage["total"]
        token_totals["calls"] += 1

    use_search = isinstance(model, str) and "gemini" in model.lower()
    if use_search:
        from google import genai  # lazy
        from google.genai import types as genai_types  # lazy

        _gclient = genai.Client()  # reads GOOGLE_API_KEY / GOOGLE_GENAI_USE_VERTEXAI
        search_tools = [
            genai_types.Tool(google_search=genai_types.GoogleSearch())
        ]

        def _complete(prompt: str, tools) -> str:
            resp = _gclient.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=16384,  # reasoning + a whole extended plan
                    tools=tools,
                ),
            )
            _tally(resp)
            return resp.text or ""
    else:
        from openai import OpenAI  # lazy

        _oclient = OpenAI(
            api_key=os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("API_BASE")
            or os.environ.get("OPENAI_API_BASE"),
        )
        # the OpenAI API wants the bare model id; strip the LiteLlm `openai/`
        # routing prefix carried by SCHOOL_ROOT_AGENT_MODEL.
        _omodel = (
            model.split("/", 1)[1] if model.startswith("openai/") else model
        )
        search_tools = None

        def _complete(prompt: str, tools) -> str:  # tools unused off-Gemini
            resp = _oclient.chat.completions.create(
                model=_omodel,
                temperature=temperature,
                max_tokens=16384,
                messages=[{"role": "user", "content": prompt}],
            )
            _tally(resp)
            return resp.choices[0].message.content or ""

    call_count = [0]  # proposer-call counter; first call is logged as 1

    def _log(call_num: int, phase: str, record: dict) -> None:
        """Append a record to `<log_dir>/proposer_<call_num>_<phase>.json`."""
        if log_dir is None:
            return
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"proposer_{call_num}_{phase}.json")
        records: list = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    records = json.load(f)
            except (OSError, ValueError):
                records = []
        records.append(record)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def _attempt(prompt: str, tools):
        """One completion -> (raw_text, parsed_plan | None); ("", None) on error.

        Reasoning models (school) burn output tokens thinking; the 16384 cap and
        the plan-shape check in `_parse_plan` together turn a truncated,
        fragment-yielding response into a clean None (no injection) rather than
        a silently-wrong partial plan.
        """
        try:
            text = _complete(prompt, tools)
            return text, _parse_plan(text)
        except Exception:
            return "", None

    def propose(plan_json, context=None):
        call_count[0] += 1
        call_num = call_count[0]
        ctx = context or {}
        vocab = ctx.get("vocab") or {}
        digest = ctx.get("digest")
        prompt = (
            "You maintain the search space of a tabular ML pipeline optimized "
            "by an external MCTS engine (you never run the search). Below is "
            "the CURRENT plan: a JSON menu of operators and hyperparameter "
            "ranges per stage.\n"
            "Return ONLY a JSON object with the same schema that EXTENDS this "
            "plan. It must be a superset: NEVER remove or modify existing "
            "entries — only additions are applied. Add the few NEW options "
            "most likely to beat the evidence below, on any stage(s): new "
            "model / feature-eng stage operators as full dotted import paths "
            "(sklearn.*, skrub.*, lightgbm.*, xgboost.*), and/or new "
            'scoped_encodings groups ({"name", "cols", "options"}, optionally '
            '"additive": true to keep the original columns and '
            '"position": "post_encode" to run after vectorization), and/or '
            'backbone knobs via "cleaner": {"params": {...}} and "vectorizer": '
            '{"params": {...}, "slots": {"high_cardinality"|"low_cardinality"|'
            '"numeric"|"datetime": [<paths>]}} (each omitted knob keeps skrub\'s '
            "default; list a boolean's skrub default first). ALWAYS "
            'give each new operator a generous hyperparameter range via '
            '"params" ({"int": [lo, hi]}, {"float": [lo, hi], "log": true} or '
            '{"choice": [...]}) so it competes tuned — but avoid upper bounds '
            "that blow up training time. You may rate a new option with "
            '"prior": 0.0-1.0. List new operators most-robust-first, and if a '
            "new operator is a stronger, more robust default than what the plan "
            "already has (e.g. a model that tolerates NaNs/unscaled features "
            "where the current default does not), give it a high prior — the "
            "search visits high-prior options first, since additions cannot "
            "displace the existing root config.\n\n"
            f"Current plan:\n{_json.dumps(plan_json, indent=1, default=str)}\n\n"
            f"Search evidence (per stage, option -> mean CV reward): "
            f"{ctx.get('stage_values', {})}.\n"
            f"Current best config (reward {ctx.get('incumbent_score')}): "
            f"{ctx.get('incumbent')}.\n"
            f"Stage the search is currently focused on (a good place to add "
            f"options, not the only one): {ctx.get('target_stage')!r}.\n"
            f"Known columns: {ctx.get('columns', [])}.\n"
            + (f"Data digest:\n{digest}\n" if digest else "")
            + (
                "Use the google_search tool to check for current SOTA "
                "operators and good hyperparameter ranges before you answer.\n"
                if use_search
                else ""
            )
            + f"Known-good operators: {vocab}."
        )

        def _stamp() -> str:
            return datetime.now(timezone.utc).isoformat()

        _log(
            call_num,
            "request",
            {
                "ts": _stamp(),
                "agent": "proposer",
                "call": call_num,
                "phase": "request",
                "target_stage": ctx.get("target_stage"),
                "search": use_search,
                "prompt": prompt,
            },
        )
        text, proposal = _attempt(prompt, search_tools)
        _log(
            call_num,
            "response",
            {
                "ts": _stamp(),
                "agent": "proposer",
                "call": call_num,
                "phase": "response",
                "search": use_search,
                "output": text,
                "proposal": proposal,
            },
        )
        # Symmetric safety net (mirrors plan_author): a grounded call can return
        # nothing usable; retry once with search OFF before giving up.
        if use_search and proposal is None:
            text, proposal = _attempt(prompt, None)
            _log(
                call_num,
                "response",
                {
                    "ts": _stamp(),
                    "agent": "proposer",
                    "call": call_num,
                    "phase": "response",
                    "search": False,
                    "retry": True,
                    "output": text,
                    "proposal": proposal,
                },
            )
        return proposal

    propose.token_totals = token_totals  # read by pipeline for the cost report
    return propose
