# Project state & roadmap

**Thesis:** adapt MLE-STAR into an **MCTS search over skrub DataOps pipelines**.
LLM agents read the data and author a *rich JSON plan* (a menu of operators +
hyperparameter ranges per stage); a pure-code MCTS engine then searches that
space — structure **and** hyperparameters — over a fixed evaluation budget. The
LLM never does the search; it only proposes the space (**O(1) LLM calls per
task**, plus at most one call per outer step for Option 3 — never inside the
inner search loop).

_Last updated: 2026-07-01 (end of Week 1). Offline suite: **73 passed, 2 skipped**
(`uv run python -m pytest tests/ -q`, ~4.5 min; the 2 skipped are the gated live
Gemini smoke tests)._

## Architecture — three layers

```
  AGENT LAYER (Gemini via ADK)            LOGIC LAYER (pure Python, no LLM)
  data_analyst ─► plan_author ──JSON──►   spec_resolver ─► skrub_ops ─► search_loop ─► mcts
  (reads digest, (rich plan: ops +        (names+HP ->     (build plan,  (persist tree, (UCT
   web search)    HP ranges)               seeded instances) action space, ablation,      search)
                                                            rollouts)     option inject)
        └──────────── pipeline.py (driver) wires the whole thing ───────────┘
```

Design rule: **clients may be provider-native, but the logic layer imports no
LLM client.** See [agent-architecture.md](agent-architecture.md).

## End-to-end flow (`pipeline.run_pipeline`)

1. **load_task** → dataframe + target + task_type + metric (parses `task_description.txt`).
2. **make_data_summary** → compact EDA digest (the only thing the LLM sees of the data).
3. **ADK agents** → `data_analyst` (web search) → `plan_author` → JSON plan in state.
4. **resolve_spec** → JSON names/HP-ranges → seeded estimators + `choose_*` nodes (allowed-list, no `eval`).
5. **build_staged_plan** → skrub DataOps plan; **get_action_space** → the MCTS search space; **get_choice_gating** → model→HP gate.
6. **run_search_loop** → persisted-tree MCTS over `budget` rollouts (score cache + gated HPs); `outer_steps>1` adds ablation targeting (Option 1) and, with a proposer, per-stage option injection (Option 3).
7. **report** → score the incumbent on the competition metric (RMSE, etc.); write `runs/<task>_<ts>/{result.json,summary.md}`.

## Module map

| File | Role |
|---|---|
| `mcts.py` | MCTS engine (UCT select/expand/backprop, **persistent tree, score cache, `gating`/`target_key` in `expand`, `canonicalize`**, `prior_fn` hook, DOT/ASCII viz) |
| `skrub_ops.py` | skrub glue: `build_staged_plan` (incl. relational `assemble`), `get_action_space`, **`get_choice_gating`**, `apply_state`, seeded rollouts (configurable `scoring`), `run_ablation`, `pick_target_node` |
| `search_loop.py` | **outer loop**: persisted MCTS across steps; `tree_action_values` (tree-mined ablation), Option 1 targeting + non-target locking, Option 3 option injection (`_inject`/`_augment_spec`), `make_llm_proposer` (one Gemini call/outer step) |
| `spec_resolver.py` | LLM JSON → seeded estimators **+ HP `choose_*`**; curated allowed-list registry; clips HP ranges; `assemble` passthrough |
| `data_summary.py` | `make_data_summary` — EDA digest for the analyst |
| `adk_agent.py` | ADK graph `data_analyst → plan_author`; `google_search`; `build_root_agent` factory |
| `metrics.py` | search-reward scorer (per task) vs report metric (competition) |
| `pipeline.py` | end-to-end driver + CLI (`--budget`, `--outer-steps`, `--refine`) |
| `run_logging.py` | sanity log of prompts+outputs to JSONL (`log_dir`) |
| `agent.py`, `sub_agents/`, `eval/` | legacy MLE-STAR / OpenAI template (decoupled; kept for merge, **not on the MCTS path**) |
| `probe_gemini.py` | standalone model/quota probe |

---

## Current state — done & tested ✅

**The Week-1 spine is complete and green.** Everything below has offline tests.

**Search-quality core (Week 1)**
- ✅ **MCTS engine** — UCT select/expand/backprop, persistent tree, DOT/ASCII viz.
- ✅ **Score cache** — `mcts.score_cache` memoizes `state_key → reward`; deterministic
  rollouts make it exact, so each distinct config is evaluated at most once
  (`test_score_cache_one_call_per_distinct_state`, and asserted across outer steps).
- ✅ **Conditional (model-gated) HP nesting (CASH fix)** — `get_choice_gating` reads
  skrub's conditional-children graph; `expand` only edits an HP when its parent model
  is selected, and `canonicalize` drops inactive HPs so the cache/dedup don't split on
  them (`test_gating_skips_inactive_hp_and_canonicalizes`,
  `test_run_states_are_model_gated_canonical`).
- ✅ **Option 1 — tree-mined ablation + non-target locking** — `search_loop.tree_action_values`
  mines per-stage deltas from the persisted tree (no fresh rollouts), `pick_target_node`
  chooses the highest-variance stage, `target_key` locks the rest and refocuses expansion
  on that stage (`test_targeting_picks_an_operator_stage`, `test_target_key_restricts_expansion`).
- ✅ **Option 3 — LLM per-stage option injection** — after targeting, one proposer call/outer
  step suggests new operator paths for the target stage; `_inject` allow-lists + de-dupes
  them, the plan is rebuilt, and search continues on the same tree. **A run keeps a pipeline
  containing an option not in the original plan** (`test_run_keeps_an_option_not_in_the_plan`).
  The LLM never enters the inner loop (≤ `outer_steps` calls total; `make_llm_proposer`).

**Logic + agent layers (pre-existing, still green)**
- ✅ **skrub layer** — staged plan build, action space, `apply_state`, seeded rollouts, `run_ablation`.
- ✅ **Spec resolver** — allowed-list operators **+ hyperparameter search** (clipped ranges, seeded, no `eval`).
- ✅ **ADK agent graph** on native Gemini (free AI Studio key); per-stack web search; rich JSON plan authored by the LLM (no hand-written menu).
- ✅ **End-to-end driver** with search-vs-report scoring split; run artifacts (`result.json` + `summary.md`) written per run.
- ✅ **Offline tests for every layer** (agents mocked via `FakeLlm`) + 2 gated live smoke tests.
- ✅ Python pinned to 3.13; `gemini-2.5-flash` default model.

**Datasets & fixtures**
- ✅ Task dirs present: `california-housing-prices`, `employee-salaries`, `midwest-survey`, `open-payments`.
- ✅ Offline agent-I/O fixtures: `california_agent_io` (regression), `open_payments_agent_io` (classification).

**Relational assemble — engine built & unit-tested (not yet wired end-to-end; see roadmap)**
- ✅ `build_staged_plan` supports an `assemble` stage (`AggJoiner` over `aux_tables`, `skip` as default),
  `resolve_spec` passes `assemble` through, and rollouts accept `aux`/`main_var`.
- ✅ Unit-tested on synthetic relational data: the join lifts a near-chance score
  (`test_staged.py::test_assemble_improves_relational_score`,
  `test_assemble_stage_in_action_space_with_clean_labels`).

Run all: `uv run python -m pytest tests/ -q`  (or `make test`)
Run live pipeline: `make run-live BUDGET=15` · with targeting+injection: `make run-refine`

---

## What's left to implement — prioritized, with timeline

We are at **end of Week 1 (~Jul 1)**; deadline **Tue Jul 15, 18:00**. Week 1 landed on
schedule, so the remaining work is Week 2 (second axis + amplifiers + sweeps) and Week 3
(evaluation + writeup). Judged by the invariant: *reduces rollouts / adds a real flexibility
axis without moving the LLM into the inner loop.*

### Week 2 — second axis + amplifiers + experimentation (Jul 3 – Jul 9)

| Item | Priority | Status | Est. |
|---|---|---|---|
| **Relational assemble — end-to-end auto-config** | **High** (differentiator) | Engine done; wiring left | Mon–Tue (Jul 3–4) |
| **`prior_fn` warm-start (free form)** | Medium | Engine hook only | Wed (Jul 5) |
| **Top-k ensemble** | Medium | Not started | Thu (Jul 6) |
| **Experimentation harness + sweeps** | **High** (A-grade) | Not started | Fri + weekend (Jul 7–9) |

1. **Relational assemble — end-to-end auto-config.** The skrub mechanism exists and is
   unit-tested; what's left is the full path:
   - a **multi-table task loader** (`load_task` reads a single `train.csv` today) + a real
     relational dataset (e.g. `fetch_credit_fraud`; not yet in `tasks/`);
   - thread `aux_tables`/`main_var` through `run_search_loop` → `build_staged_plan` /
     `make_rollout_fn` (the driver never passes `aux` today);
   - have `plan_author` **propose `AggJoiner` configs** from the digest (schema + prompt);
   - **`AggTarget` leakage guard** — target-based aggregation must be computed inside the CV
     fold; treat it as a guarded, leakage-checked action (only `AggJoiner` exists now).
   *Done when:* a multi-table run auto-configures an `AggJoiner` and the lift is real on
   held-out data, not just validation.

2. **`prior_fn` warm-start (free form only).** The engine hook exists (`mcts.mcts_search(prior_fn=…)`,
   called on freshly expanded children) but is unwired. Build the *free* version only:
   extend `spec_resolver`'s schema with an optional per-option prior weight, have `plan_author`
   emit it **in its existing call**, and make `prior_fn` a pure lookup that seeds child `Q` + a
   small pseudo-count `N`. **Zero new LLM calls.** If it can't fold into the existing call, cut it.
   *Done when:* children enter with a prior-seeded Q and a run reaches a comparable score in
   fewer rollouts.

3. **Top-k ensemble (thin read-off).** Read the top-k distinct incumbents off the persisted
   tree, fit, and average / soft-vote. No LLM, no new search. *Done when:*
   `top_k_ensemble(tree, k=…)` beats the single incumbent on ≥1 dataset.

4. **Experimentation harness + sweeps (the A-grade differentiator).** No harness exists yet
   (`eval/` is the legacy ADK template, off the MCTS path). Build a small driver that runs the
   pipeline across seeds/datasets and produces: **`c`-sweep** {0.3, 0.5, 0.7, 1.0}, **ablation-loop
   on/off** (Option 1), **outer-step budget split**, **Option 3 dosage**. *Done when:* 2–3 sweep
   figures each with a one-sentence finding.

> **Hard feature freeze: end of Week 2 (Sun Jul 9).** After this — evaluation, debugging, writing only.

### Week 3 — freeze, evaluate, write, buffer (Jul 10 – Jul 15)

| Item | Est. |
|---|---|
| **Full evaluation** across all demo datasets | Wed–Thu (Jul 10–11) |
| **Three headline figures + MLE-STAR comparison table + slides** | Fri–Sat (Jul 12–13) |
| **Buffer** (dataset debugging, Gemini quota, dry-run) | Sun–Mon (Jul 13–14) |
| **Submit** | Tue Jul 15, before 18:00 |

- **Full evaluation** on `employee_salaries`, `credit_fraud`, `california_housing`
  (+ `midwest_survey`/`bike_sharing` if clean).
- **Three headline comparisons:** (1) **flexibility lift** — fixed space vs Option 3;
  (2) **relational lift** — assemble vs flat on `credit_fraud` (ideally vs an AutoGluon
  flat-table baseline); (3) **targeted-refinement lift** — Option 1 on vs off. Plus the
  ensemble lift and the `c`-sweep.
- **Writeup + slides:** the MLE-STAR comparison table (mechanism / debug cost / token cost /
  leakage handling / adaptivity) + the figures.

### Ops / housekeeping (fit in around the above)

- `uv lock` reconcile when online (`make sync`); local/Docker version parity.
- Acquire/stage the relational dataset (`fetch_credit_fraud`) under `tasks/`.
- Stable live-eval runs once Gemini quota is steady.

### Explicitly cut (do not reopen)

- Full MLE-STAR iterative ensembler (the thin top-k read-off replaces it).
- ArchPilot-style restart, mid-search GEN/progressive-widening — wrong fit (our scorer is fixed), wrong cost class.
- `scope` / `post-process` skrub stages, per-model training loss as a searchable HP — future work.
- Per-expansion or per-rollout LLM calls — the invariant; rejected regardless of per-call cost.

### Fallback

If relational assemble isn't stable end-to-end by midweek 2, ship **Option 1 + Option 3 +
top-k ensemble** as the evaluated result and present relational as designed/partially-built
future work. **Protect Week-3 evaluation time above any single feature.**

---

## Key design decisions (for slides / Q&A)

- **Structured JSON plan, not code-gen** — no `eval` of model output, central
  seeding (determinism MCTS needs), schema-validated, MCTS-friendly named choices.
- **Allowed-list registry** (not dynamic import) — novel ops/HPs are dropped or
  clipped, never executed; Option 3's injected paths go through the same gate.
- **Two scorers** — bounded higher-is-better reward for *search*; the competition
  metric only for the *final report* (they must not be conflated).
- **Persist the tree, don't restart** — our scorer is fixed, so a config's reward never
  changes when the target moves; the tree is a running ablation we mine for free.
- **LLM-call complexity is the cost model** — O(1) per task, ≤ O(outer steps) with Option 3;
  never O(expansions) or O(rollouts). The single real cost is CV rollouts.
- **Provider-native clients, shared logic** — Gemini (ADK) and OpenAI stacks stay
  separate; the MCTS/skrub core is client-agnostic and import-clean.
</content>
</invoke>
