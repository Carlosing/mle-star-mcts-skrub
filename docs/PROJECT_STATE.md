# Project state & roadmap

**Thesis:** adapt MLE-STAR into an **MCTS search over skrub DataOps pipelines**.
LLM agents read the data and author a *rich JSON plan* (a menu of operators +
hyperparameter ranges per stage); a pure-code MCTS engine then searches that
space — structure **and** hyperparameters — over a fixed evaluation budget. The
LLM never does the search; it only proposes the space (**O(1) LLM calls per
task**, plus at most one call between search slices for Optional Feature 3 — never inside
the inner search loop).

_Last updated: **2026-07-15 — submission.** The code is feature-complete and the
offline suite is green. The **shipped headline results are knowingly biased in
our own favour** and are reported as such: they predate the on-disk
train/holdout split (commit `517f954`), so the search selected its configs with
the eval rows visible while AutoGluon's did not. The fix is in the code, and the
bias has since been **spot-measured on 4 tasks via zero-quota replays** (small;
conclusively ~0.015 roc_auc on credit-fraud, within run-to-run noise elsewhere).
Read [Hand-off — the published numbers are biased](#hand-off--the-published-numbers-are-biased)
before you write a single number down. Historical bugs (symptom → cause → fix →
test) are catalogued in [BUG_LEDGER.md](BUG_LEDGER.md); ideas deliberately NOT
implemented are in [FUTURE_WORK.md](FUTURE_WORK.md)._

## Hand-off — the published numbers are biased

**Decision (2026-07-14): ship the existing results and disclose the bias,
rather than re-run the benchmark.** There was no time (and, at the end, no API
availability) to re-run the full live benchmark before the deadline. This
section is the record of *exactly* what the bias is, so any writeup can state
it precisely instead of hand-waving.

### What is and isn't wrong with the archived results

They are **not** garbage, and they are **not** incomparable. The two arms are on a
common bench:

- Both the extension and AutoGluon scored the **identical rows** — both called
  `ensemble.holdout_split(train.csv, target, task_type, seed=42, frac=0.25)`, the
  same function with the same seed, and `train.csv` was byte-stable across every
  archived run.
- Both **fit on the same 75%** and scored the same 25%. The final-fit protocol is
  symmetric.

The asymmetry is in **model selection**, and it is one-directional:

> `pipeline.run_pipeline` handed the **full** `train.csv` to `run_search_loop`, so
> every MCTS rollout cross-validated over rows that were also in the 25% holdout.
> The extension therefore *chose* its pipeline, model and hyperparameters with the
> eval rows visible. AutoGluon chose its models having only ever seen the 75%.

So the archived extension score is **optimistically biased**; AutoGluon's is
clean. The reported extension−AutoGluon delta is an **upper bound** on the true
delta. (Full write-up: BUG_LEDGER #26–#29.)

### The bias has been spot-measured (2026-07-15)

The "cheap way out" was partially taken: four tasks' captured plans were
replayed on the fixed on-disk split at **zero API cost**
(`scripts/replay_from_run.py` — same plan, budget and proposals as the archived
run). Result: the bias is **real but small — smaller than per-task run-to-run
variance on 3 of the 4 probes**. Only credit-fraud (12k rows) resolves it
cleanly: clean 0.807 roc_auc vs archived 0.826, below *every* archived run for
that task — ~0.015 roc_auc of optimism in the predicted direction.
bike-sharing and traffic-violations moved *within noise or better* when clean;
117-row country-happiness is variance-dominated either way. Full table:
[EXPERIMENTAL_RESULTS.md](../EXPERIMENTAL_RESULTS.md) caveat 1.

### How to report it

- ✅ **Quote `result["holdout"]["score"]`** — the *incumbent* on the shared 25%.
  This is what `scripts/make_figures.py` plots, and it carries **only** the
  selection bias above.
- ❌ **Do NOT quote `ensemble_score`** from the Caruana-era archived runs
  (anything after `4490d2e`). There, `_caruana_select` greedily *maximised* the
  holdout score and we then published that maximum — a **second, larger** bias
  stacked on the first, and precisely the ensemble-overfitting flaw our own
  report criticises MLE-STAR for. The simplest safe rule: **report the
  incumbent, not the ensemble.**
- State the bias **direction and cause**; its magnitude is now spot-measured
  (~0.015 roc_auc where measurable) but NOT measured everywhere — write
  "optimistic, small where measurable", never "negligible".
- **MLE-STAR's 13 archived runs** (`results/mle-star-*/`) are its **own internal
  validation score**, not the shared holdout. See
  [the three score bases](#the-three-arms-are-scored-on-three-different-bases).

### The three arms are scored on three different bases

| Arm | Scored on | Caveat |
|---|---|---|
| Extension (archived) | `holdout_split(train, seed=42)` | **selected** on those same rows → optimistic (BUG_LEDGER #26); spot-measured ~0.015 roc_auc on credit-fraud |
| AutoGluon (archived) | `holdout_split(train, seed=42)` | clean (never saw them) |
| MLE-STAR (13 runs) | **its own internal validation** (`submission_code_exec_result.score`) | **self-reported**, not any shared holdout |

The MLE-STAR numbers came from runs made *before* the targeted `test.csv`
existed, so MLE-STAR could only score on a holdout it carved itself.
`scripts/convert_mlestar_final_state.py` ingested each run's `final_state.json`
and records "MLE-STAR's own internal score … *not* a re-score against the shared
holdout." Caption its bars as **indicative, self-reported**; do not present the
MLE-STAR gap as a like-for-like holdout comparison. Notes: `videogame-sales`
produced no runnable script (a reportable failure — MLE-STAR is bounded by API
calls); `credit-fraud` scored roc_auc 0.500 (chance).

**A clean three-way would need** MLE-STAR re-run through `scripts/run_mlestar.py`
(which *does* score against the shared `test.csv`/`test_answer.csv`) plus fresh
extension + AutoGluon runs — the extension via zero-quota replay, AutoGluon via
a CPU re-run.

### Also still true

- **Any *fresh* run is clean** — the code is fixed. On a new run, check
  `ensemble.selection == "oof_3fold"` (not `legacy_holdout`) before quoting it.
- **Wide high-cardinality tasks remain intractable at a real budget**
  (traffic-violations projected ~15h at full width; GapEncoder forfeits on the
  60s rollout cap). See [BUG_LEDGER § Open / flagged](BUG_LEDGER.md#open--flagged-not-fixed).

## Architecture — three layers

```
  AGENT LAYER (Gemini via ADK)            LOGIC LAYER (pure Python, no LLM)
  data_analyst ─► plan_author ──JSON──►   spec_resolver ─► skrub_ops ─► search_loop ─► mcts
  (reads digest, (rich plan: ops +        (names+HP ->     (build plan,  (persist tree, (UCT
   web search)    HP ranges)               seeded instances) action space, ablation,      search)
                                                            rollouts)     option inject)
        └──────────── pipeline.py (driver) wires the whole thing ───────────┘
```

Design rule: **one ADK stack drives either provider (native Gemini or
OpenAI/compatible via `LiteLlm`), switched by `ROOT_AGENT_MODEL`; the logic
layer imports no LLM client.** See [agent-architecture.md](agent-architecture.md).

## End-to-end flow (`pipeline.run_pipeline`)

0. **The split already exists on disk.** `scripts/stage_tasks.py` (`make stage-tasks`)
   wrote `train.csv` / `test.csv` / `test_answer.csv` once, before any method ran.
   `load_task` reads **only** `train.csv`; `load_holdout` reads the other two and
   is handed to the *scorers* alone. The search physically cannot see the holdout.
1. **load_task** → dataframe + target + task_type + metric (parses `task_description.txt`).
2. **make_data_summary** → compact EDA digest (the only thing the LLM sees of the data).
3. **ADK agents** → `data_analyst` (web search) → `plan_author` → JSON plan in state.
4. **resolve_spec** → JSON names/HP-ranges → seeded estimators + `choose_*` nodes (allowed-list, no `eval`).
5. **build_staged_plan** → skrub DataOps plan; **get_action_space** → the MCTS search space; **get_choice_gating** → model→HP gate.
6. **run_search_loop** → persisted-tree MCTS over `budget` rollouts (score cache + gated HPs),
   amplified by the three **Optional Features** (older commit messages call 1 and 3 "Option 1/3"):
   - **Optional Feature 1 — ablation targeting**: `outer_steps>1` runs the budget as fixed-size
     slices on one tree, re-mining the tree between slices to pick a focus stage — a proposer
     hint only, expansion is never locked (`retarget=`).
   - **Optional Feature 2 — prior warm-start**: plan options may carry `"prior": 0.0–1.0`;
     `resolve_spec` collects them and `_make_prior_fn` seeds every fresh child's Q/N (neutral
     0.5 default), the AlphaZero policy-prior pattern at zero extra LLM calls.
   - **Optional Feature 3 — mid-search plan injection**: with a proposer, one LLM call before
     each subsequent slice asks for a whole *extended plan* (`search → propose → search → …`;
     merged strictly additively via `_merge_raw_plans`, re-resolved and rebuilt, so injected
     operators arrive with their HP ranges).
   After the budget, a **focused-refinement bonus phase** (`ceil(total/4)` rollouts, no LLM)
   descends from the incumbent node and explores ALL of its single-edit neighbors
   (`refinement_phase=`, dims in `refined_dims`).
7. **ensemble** (`top_k > 1`) → Caruana greedy selection over a candidate pool of the
   top `max(2*top_k, ensemble_pool)` distinct cache configs. Members are **selected**
   on 3-fold out-of-fold predictions spanning all of train, then **refit on all of
   train and scored on the shared holdout** — selection rows and reported rows are
   never the same rows (`ensemble.selection` records which logic ran).
8. **report** → score the incumbent on the competition metric by CV (`report`) *and*
   on the shared on-disk holdout (`holdout` — the cross-method comparable number);
   write `runs/<task>_<ts>/{result.json,summary.md,ensemble.pkl}`.

## Module map

| File | Role |
|---|---|
| `mcts.py` | MCTS engine (UCT select/expand/backprop, **persistent tree, score cache, `gating`/`target_key` in `expand`, `canonicalize`**, `prior_fn` hook) |
| `skrub_ops.py` | skrub glue: `build_staged_plan` (incl. relational `assemble` + fixed `_ScalarizeAggregates`/`_SanitizeColumns` steps), `get_action_space`, **`get_choice_gating`**, `apply_state`, seeded rollouts (profile-aware subsample sizing, optional seed-averaged rewards, per-rollout 60s wall-clock cap, CV folds parallel via `n_jobs`), `run_ablation`, `pick_target_node` |
| `search_loop.py` | **outer loop**: persisted MCTS as fixed-budget slices; `tree_action_values` (tree-mined ablation), Optional Feature 1 focus pick (`retarget=`, a proposer hint), Optional Feature 3 **whole-plan extending injection**, `make_llm_proposer` (≤ one provider call between slices); post-budget **focused-refinement bonus phase** |
| `spec_resolver.py` | LLM JSON → seeded estimators **+ HP `choose_*`**; **import** allow-list (roots `sklearn`/`skrub`/`lightgbm`/`xgboost`) is the safety envelope; free-form HP ranges unless curated in `REGISTRY` (then clipped); per-param safety nets; `assemble` passthrough |
| `data_summary.py` | `make_data_summary` — EDA digest for the analyst |
| `adk_agent.py` | ADK graph `data_analyst → plan_author`; `build_root_agent`; `_resolve_model` env-switches native Gemini vs `LiteLlm`; `google_search` Gemini-only |
| `metrics.py` | search-reward scorer (adopts a bounded task metric like roc_auc) vs report metric (competition) |
| `ensemble.py` | **Caruana ensemble read-off** over the persisted score cache: `top_k_states`, `_caruana_select` (weighted, with replacement, early stop), `evaluate_top_k` (OOF selection / holdout reporting on disjoint rows, `selection` stamped), `holdout_split`, `EnsemblePredictor` (`ensemble.pkl`) |
| `pipeline.py` | end-to-end driver + CLI; multi-table `load_task` (reads **only `train.csv`**); `load_holdout`; `save_run_artifacts` |
| `runner.py` | minimal OpenAI-compatible runner for the **revived MLE-STAR baseline**; `llm_call` is the single chokepoint the benchmark harness wraps for call/token/time caps |
| `run_logging.py` | sanity log of prompts+outputs to JSONL, incl. per-call `tokens` |
| `agent.py`, `sub_agents/`, `shared_libraries/`, `eval/` | the **upstream MLE-STAR agent, revived as the benchmark baseline** (driven by `scripts/run_mlestar.py` through `runner.py`). Off the MCTS path. `shared_libraries/web_search_util.py` (DuckDuckGo) backs its model-retrieval step |
| `probe_gemini.py`, `probe_school.py` | standalone model/quota probes |

### Benchmark + evaluation scripts

| File | Role |
|---|---|
| `scripts/stage_tasks.py` | **Draws the one and only train/holdout split, on disk** (`make stage-tasks`); encodes per-dataset knowledge (leaky columns, subsample size, aux joins) |
| `scripts/stage_credit_fraud.py` | downloads + stages the relational credit-fraud task |
| `scripts/run_autogluon.py` | AutoGluon baseline on the **same** holdout + time budget. Flat-table only; `NUM_CPUS=1` on macOS-ARM (libomp) |
| `scripts/run_mlestar.py` | the revived MLE-STAR under hard caps (max LLM calls, per-call token bound, deadline); token cost otherwise *unbounded* |
| `scripts/claude_agents.py`, `scripts/run_claude_pipeline.py` | offline Claude-authored plans + replay proposer → full runs at **zero API quota** |
| `scripts/replay_from_run.py` | re-runs the search from a stored run's captured plan (`spec_raw`) — used both for A/Bs and for the fixed-split bias spot-measurements |
| `scripts/convert_mlestar_final_state.py` | ingests external MLE-STAR `final_state.json` runs into the uniform `result.json` schema |
| `scripts/collect_results.py` | `runs/` → the small git-shareable `results/` mirror |
| `scripts/make_figures.py` | `results/` → `figures/` (quality_at_cost 5×2 grid, token_cost, proposal_scaling, mechanism_table.md, comparison.csv). All three methods emit one uniform `result.json` schema |

---

## Development history — as the commits tell it

The repo's 58 commits fall into five phases. (`git log --oneline --reverse` is
the source of truth; commit hashes below are anchors into it.)

### Phase 1 — upstream import and the OpenAI port (`0eb79c1` → `241beeb`)

The repo began as Google's MLE-STAR ADK sample (`338cad9 Import MLE-STAR
project`) plus team infrastructure: Dockerfile (`0381d7e`), the university
(GWDG) OpenAI-compatible API credentials (`106f630`). The first substantial
engineering was a **port of MLE-STAR off the ADK runtime onto plain
OpenAI-compatible calls**: a `ManagerAgent` with a token budget and kill switch
(`fd2f9da`), a sub-agent execution protocol (`8ed6658`), the full multi-agent
pipeline (`cce0b7e`), and the working MVP port (`fac8795`, reconstructed in
`241beeb`). This ported agent later became the **benchmark baseline** — it was
never deleted.

### Phase 2 — the pivot: MCTS engine + skrub layer (`6673725` → `6be659f`)

The project's actual thesis arrived in one stroke: `6673725 Add MCTS engine,
skrub_ops staged pipelines, tests, docs` — the UCT engine, the staged skrub
plan builder, and their offline tests, all import-clean of any LLM client. In
parallel the agent layer moved onto **Google ADK with native Gemini**
(`e7db9c2`, `e4a8f6d` — which also introduced the skrub-based data digest given
to the planner), with unit/integration tests for the revived-baseline utilities
(`6be659f`).

### Phase 3 — search capability build-out (`b775203` → `2c85ecc`)

The search grew its distinguishing features, roughly one axis per commit:

- `b775203` — metric-aware search rewards; offline pipeline testing on cached
  Claude-generated plans (the zero-quota driver pattern).
- `8d6b921` — live run artifacts (`result.json`/`summary.md`) + the Makefile.
- `307559e` — the **import allow-list** generalizing operator resolution.
- `a1093a6` — **conditional (model-gated) HP expansion, the score cache, and
  Optional Feature 3 injected options** — the CASH fix and the mid-search flexibility axis.
- `e61e469` — **relational table-join wiring** (AggJoiner assemble), ensemble
  improvements, the plan proposer, stratified-KFold fixes.
- `b9dd9da` — **interleaved propose-between-slices** (`search → propose →
  search`), additive vs in-place scoped operations.
- `2c85ecc` — incumbent-focused HP refinement (the bonus phase's ancestor), the
  per-rollout time limit, free-form LLM hyperparameter ranges; the legacy
  OpenAI client deprecated in favour of the env-switched ADK stack.

### Phase 4 — hardening to feature freeze (`e7c80a8` → `cc14f34`)

- `e7c80a8` — **DuckDuckGo web search integrated into MLE-STAR's model
  retrieval** (the baseline's analogue of the analyst's `google_search`).
- `cf40f12` — the silent proba-scoring bug fixed (skrub's transformer-typed
  learner zeroed roc_auc for proba-only classifiers); search capability added
  to planner/proposer prompts.
- `f8b710e` — pre-freeze bug-fix sweep; the offline suite consolidated.
- `d5e0f91` — the task datasets staged.
- `0f9d618` — the **second provider**: OpenAI-compatible (school/GWDG)
  integration for all three agents, plan-parser robustness, `n_jobs=6` CV
  parallelism, USAGE docs.
- `cc14f34` — per-call token/time logging; live toxicity validation.

### Phase 5 — evaluation integrity and the benchmark (`8cc66f9` → `617d2ba`)

Everything after the freeze is **evaluation plumbing or an integrity fix** — no
new search features; the LLM-call invariant was never touched:

| Commit | What | Why it matters |
|---|---|---|
| `8cc66f9` | **Always-on skrub backbones** — `clean`/`encode` are a bare `Cleaner()` + `TableVectorizer()`; their knobs LLM-authored like HPs | the old code-owned fallback menus were arbitrary; the space is now entirely LLM-proposed, code-validated |
| `3c00e95` | **TextEncoder backbone cache + JSON parse safety nets** | 1 embedding-model load per run, not per rollout; qwen's Python-literal JSON recovered instead of falling back to a bare plan |
| `4490d2e` | **Caruana ensemble** (greedy weighted selection over a top-N pool) + `_ImputeNumeric` NaN sanitization after aggregation | reaches past near-duplicate cousins for model-family diversity; post-join NaNs were silently failing models |
| `42536c6`, `ed932c4` | **Retarget lock removed; sweep harness removed** | replay A/Bs showed the variance ledger elects `model` on essentially every pick (a null A/B axis) while the lock starved off-target injections; budget/N_PROPOSES are chosen by wall-clock + call count, not swept |
| `d0cc01d` | AutoGluon preset → `best_quality`; FUTURE_WORK.md | the baseline actually spends its time budget |
| `421c571` … `ca5c144` | results ingested for all 10 benchmark tasks (extension + AutoGluon), replay/proposal-scaling plots | the comparison dataset |
| `7867f06` | **merge: the web-search MLE-STAR baseline adopted into the MCTS branch** | the three-way benchmark united in one tree |
| `517f954` | **fix: the train/holdout split drawn on disk before any method sees the data** | **the integrity fix** — see the Hand-off section; BUG_LEDGER #26–#29 |
| `dbc0f77` | **OOF ensemble selection** — Caruana selects on 3-fold out-of-fold rows, reports on the untouched holdout, `selection` stamped | selecting on the reported rows was the exact flaw our report criticises MLE-STAR for |
| `eb94f55` | `--legacy-ensemble` flag, pickled `EnsemblePredictor`, root `figures/` | pre-fix results stay reproducible; the two ensemble paths are A/B-able |
| `da3763a` | MLE-STAR results ingested into `results/` | the third arm appears in the figures (self-reported basis — see Hand-off) |
| `617d2ba` | figures finalized: 5×2 quality-at-cost grid, combined token cost | the shipped figure set |

---

## Current state ✅

- **Feature-complete; offline suite green** (427 tests, ~5 min, agents mocked
  with `FakeLlm`, real skrub CV — `make test`).
- **All 10 benchmark tasks have archived results** for the extension and
  AutoGluon (shared `holdout_split` basis) and MLE-STAR (self-reported basis);
  figures rendered in `figures/`.
- **Evaluation integrity fixed in code** (on-disk split, OOF ensemble
  selection); archived pre-fix results disclosed, bias spot-measured via
  4 zero-quota replays.
- Submission docs at the repo root: `README.md`, `CONTRIBUTIONS.md`,
  `EXPERIMENTS.md`, `EXPERIMENTAL_RESULTS.md`.

### What's next (if the project continues)

1. **The clean three-way re-run** — MLE-STAR through `run_mlestar.py` (scores
   the shared bench), fresh extension runs (live, or replay for zero quota),
   AutoGluon CPU re-runs. Turns every disclosed caveat into a measured number.
2. **More replay probes** — each archived run has a captured plan; every added
   replay sharpens the bias estimate at zero API cost.
3. Ideas deliberately deferred: [FUTURE_WORK.md](FUTURE_WORK.md).

---

## Key design decisions (for slides / Q&A)

- **Structured JSON plan, not code-gen** — no `eval` of model output, central
  seeding (determinism MCTS needs), schema-validated, MCTS-friendly named choices.
- **Import-level allow-list** (not dynamic import) — the safety envelope is the
  import root (`sklearn`/`skrub`/`lightgbm`/`xgboost`); a hallucinated operator is
  dropped, never executed. HP ranges free-form unless curated in `REGISTRY`.
- **Two scorers** — bounded higher-is-better reward for *search*; the competition
  metric only for the *final report*.
- **The bench is drawn on disk, before any method runs** — enforced by absence,
  not convention: the holdout rows are simply not in any file the search can
  read. One level down, the ensemble **selects** on out-of-fold rows and
  **reports** on the holdout — a number you fitted to is not a number you may
  publish.
- **Persist the tree, don't restart** — the scorer is fixed, so a config's reward
  never changes when the target moves; the tree is a running ablation mined for free.
- **LLM-call complexity is the cost model** — O(1) per task, ≤ O(outer steps)
  with Optional Feature 3; never O(expansions) or O(rollouts).
- **One ADK stack, env-switched provider, shared logic** — `ROOT_AGENT_MODEL`
  selects native Gemini or OpenAI/compatible with no code change; the MCTS/skrub
  core is client-agnostic and import-clean.

## Explicitly cut (do not reopen)

- Full MLE-STAR iterative ensembler (the Caruana read-off in `ensemble.py`
  replaces it — it reuses the search's own score cache, so it costs no new rollouts).
- ArchPilot-style restart, mid-search GEN/progressive-widening — wrong fit (our
  scorer is fixed), wrong cost class.
- `post-process` skrub stage, per-model training loss as a searchable HP — future work.
  (`scope` was on this list but was **reopened by decision and shipped** as
  `scoped_encodings`.)
- Per-expansion or per-rollout LLM calls — the invariant; rejected regardless of
  per-call cost.
- The sweep harness (removed `42536c6`) — do not rebuild it to A/B `retarget`;
  that axis was measured and is null.
