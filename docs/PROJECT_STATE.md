# Project state & roadmap

**Thesis:** adapt MLE-STAR into an **MCTS search over skrub DataOps pipelines**.
LLM agents read the data and author a *rich JSON plan* (a menu of operators +
hyperparameter ranges per stage); a pure-code MCTS engine then searches that
space — structure **and** hyperparameters — over a fixed evaluation budget. The
LLM never does the search; it only proposes the space (**O(1) LLM calls per
task**, plus at most one call between search slices for Option 3 — never inside
the inner search loop).

_Last updated: 2026-07-11. Pre-ship quality upgrades (2026-07-10): whole-plan
extending injection replaces stage-targeted Option 3, focused-refinement bonus
phase replaces the HP-only one, structure-aware subsampling + seed-averaged
rewards fix the noisy imbalanced-task proxy, `_SanitizeColumns` makes injected
boosters fit through skrub's special-character column names. 2026-07-11 session:
robustness + perf + second provider — parser rejects truncated JSON fragments;
numeric grids include each dim's default (root reachable by `expand`);
sparse/tuple params coerced; ensemble deduped on effective state; malformed/
single-option stages surfaced (`dropped_sections`/`single_option_stages`);
`all_aggregates` chains multi-table joins; rollout wall-clock cap 45→60s;
**CV fold-parallelism `n_jobs=6` (default, booster-safe — the segfault is the
estimator's own OpenMP, not joblib forking); `--n-jobs` CLI + `NJOBS` make var**;
**two-provider env scheme (`PROVIDER=google|school`) live-validated on GWDG's
qwen3.5-397b** (16k LiteLlm max_tokens for reasoning models, search hints
stripped off-Gemini, `make probe-school`). Offline suite: **pytest tests/ green**
(292 passed, 1 skipped — the gated live Gemini smoke; ~3.5 min with n_jobs=6).
Option-3 proposer is now provider-agnostic (Gemini or school/OpenAI-compat, same
env switch as the agents) — PROVIDER=school has full Option-3 parity, validated
live on qwen3.5-397b (injected KMeans/QuantileTransformer/Spline). A malformed
proposed HP (log scale + 0.0 low) that used to raise out of `resolve_spec` and
silently drop the whole injection is fixed (`_build_choice` linear-fallback +
per-param guard in `_make`); a merge that still won't resolve is recorded in
`result["proposal_injection_error"]` and surfaced in summary.md, so a failed
injection is distinguishable from "proposer added nothing". Friend-facing usage
guide at `docs/USAGE.md`; `make probe-school`, `PROVIDER=`/`NJOBS=`/`DRIVER=`
flags. Owed: `llm_calls` miscounts offline replay proposers as real calls.
Historical bugs (symptom → cause → fix → test) are catalogued in
[BUG_LEDGER.md](BUG_LEDGER.md)._

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

1. **load_task** → dataframe + target + task_type + metric (parses `task_description.txt`).
2. **make_data_summary** → compact EDA digest (the only thing the LLM sees of the data).
3. **ADK agents** → `data_analyst` (web search) → `plan_author` → JSON plan in state.
4. **resolve_spec** → JSON names/HP-ranges → seeded estimators + `choose_*` nodes (allowed-list, no `eval`).
5. **build_staged_plan** → skrub DataOps plan; **get_action_space** → the MCTS search space; **get_choice_gating** → model→HP gate.
6. **run_search_loop** → persisted-tree MCTS over `budget` rollouts (score cache + gated HPs); `outer_steps>1` runs the budget as fixed-size slices on one tree, re-mining the tree between slices to re-target a stage (Option 1, `retarget=`) and, with a proposer, asking for a whole *extended plan* before each subsequent slice (Option 3 — `search → propose → search → …`; the extension is merged strictly additively via `_merge_raw_plans`, re-resolved, and rebuilt, so injected operators arrive with their HP ranges). After the budget, a **focused-refinement bonus phase** (`ceil(total/4)` rollouts, no LLM) descends from the incumbent node and explores ALL of its single-edit neighbors — structure and HPs alike (`hp_refine=`, dims reported as `refined_dims`).
7. **report** → score the incumbent on the competition metric (RMSE, etc.); write `runs/<task>_<ts>/{result.json,summary.md}`.

## Module map

| File | Role |
|---|---|
| `mcts.py` | MCTS engine (UCT select/expand/backprop, **persistent tree, score cache, `gating`/`target_key` in `expand`, `canonicalize`**, `prior_fn` hook, DOT/ASCII viz) |
| `skrub_ops.py` | skrub glue: `build_staged_plan` (incl. relational `assemble` + a fixed `_SanitizeColumns` rename before the model so boosters accept skrub's special-character feature names), `get_action_space`, **`get_choice_gating`**, `apply_state`, seeded rollouts (configurable `scoring`; profile-aware subsample sizing `_profile_subsample_n` — imbalance/high-cardinality grow it, capped at 2000; optional seed-averaged rewards `n_subsample_seeds`; per-rollout 60s wall-clock cap via `_time_limit`/`timeout_s` → slow config scores 0.0; CV folds parallel via `n_jobs`, default 6), `run_ablation`, `pick_target_node` |
| `search_loop.py` | **outer loop**: persisted MCTS as fixed-budget slices; `tree_action_values` (tree-mined ablation), Option 1 targeting re-run between slices (`retarget=`) + non-target locking, Option 3 **whole-plan extending injection** between slices (`propose(plan_json, context) -> extended plan`, merged strictly additively via `_merge_raw_plans`, re-resolved + rebuilt — injected operators arrive tuned), `make_llm_proposer` (≤ one provider call between slices — native Gemini or OpenAI-compatible, same env switch as the agents); post-budget **focused-refinement bonus phase** (`ceil(total/4)` rollouts from the incumbent node over ALL its single-edit neighbors — `hp_refine=`, edited dims in `refined_dims`) |
| `spec_resolver.py` | LLM JSON → seeded estimators **+ HP `choose_*`**; **import** allow-list (roots `sklearn`/`skrub`/`lightgbm`/`xgboost`) is the safety envelope; free-form HP ranges (`_build_free_choice`) unless curated in `REGISTRY` (then clipped); per-param safety nets (`_accepts_param`, `_RNG_PARAMS`) drop unknown/RNG params without dropping the operator; `assemble` passthrough |
| `data_summary.py` | `make_data_summary` — EDA digest for the analyst |
| `adk_agent.py` | ADK graph `data_analyst → plan_author`; `build_root_agent` factory; `_resolve_model` env-switches native Gemini vs `LiteLlm` (OpenAI/compatible); `google_search` attached only on the Gemini path |
| `metrics.py` | search-reward scorer (per task; adopts a bounded task metric like roc_auc) vs report metric (competition) |
| `ensemble.py` | **top-k read-off**: `top_k_states` over the score cache + `evaluate_top_k` (seeded holdout, soft-vote/average) |
| `pipeline.py` | end-to-end driver + CLI (`--budget`, `--outer-steps`, `--refine`, `--top-k`); multi-table `load_task` (`aux_*.csv`) |
| `scripts/stage_credit_fraud.py` | stages the relational credit-fraud task under `tasks/` (`make stage-credit-fraud`) |
| `run_logging.py` | sanity log of prompts+outputs to JSONL (`log_dir`) |
| `agent.py`, `sub_agents/`, `eval/` | **deprecated** standalone OpenAI `ManagerAgent` template (OpenAI now runs *through* ADK via `LiteLlm`; decoupled, **off the MCTS path**) |
| `probe_gemini.py`, `probe_school.py` | standalone model/quota probes (Gemini; GWDG school endpoint) |

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
  This is the *between-slice, stage-level* targeting; the *post-budget* local search
  around the incumbent is the **focused-refinement bonus phase** (see Pre-ship upgrades
  item 2; `expand` accepts a *set* of target keys, and `mcts_search` a `start_node`).
- ✅ **Option 3 — LLM option injection between slices** — with a proposer, one call per
  outer step extends the plan mid-search; the LLM never enters the inner loop
  (≤ `outer_steps - 1` calls total; `make_llm_proposer`). **A run keeps a pipeline
  containing an option not in the original plan** (`test_run_keeps_an_option_not_in_the_plan`).
  Originally per-stage path injection (`_inject`); replaced 2026-07-10 by the
  **whole-plan extending injection** (Pre-ship upgrades item 1) — the proposer now
  returns a whole extended plan with HP ranges, merged additively via `_merge_raw_plans`.

**Logic + agent layers (pre-existing, still green)**
- ✅ **skrub layer** — staged plan build, action space, `apply_state`, seeded rollouts, `run_ablation`.
- ✅ **Spec resolver** — allowed-list operators **+ hyperparameter search** (clipped ranges, seeded, no `eval`).
- ✅ **ADK agent graph** — one stack, provider switched by `ROOT_AGENT_MODEL` (native Gemini with `google_search`, or OpenAI/compatible via `LiteLlm` with web search off); rich JSON plan authored by the LLM (no hand-written menu).
- ✅ **End-to-end driver** with search-vs-report scoring split; run artifacts (`result.json` + `summary.md`) written per run.
- ✅ **Offline tests for every layer** (agents mocked via `FakeLlm`) + 2 gated live smoke tests.
- ✅ Python pinned to 3.13; `gemini-2.5-flash` default model.

**Datasets & fixtures**
- ✅ Task dirs present: `california-housing-prices`, `employee-salaries`, `midwest-survey`, `open-payments`.
- ✅ Offline agent-I/O fixtures: `california_agent_io` (regression), `open_payments_agent_io` (classification).

**Week 2 — second axis + amplifiers (landed 2026-07-05, all offline-tested)**
- ✅ **Relational assemble, end-to-end** — `load_task` discovers `aux_*.csv` tables;
  `make_data_summary` digests each aux table **with join-key candidates**; `plan_author`
  proposes `AggJoiner` configs (exact schema in its prompt); `resolve_spec` validates them
  against the real table schemas (tables/keys/operations/cols — hallucinations dropped);
  `run_search_loop(aux_tables=…)` threads aux through builds, rollouts and the Option-3
  rebuild. Credit-fraud staged under `tasks/credit-fraud` (15k baskets, 1.25% fraud).
  (`tests/test_relational_pipeline.py`.)
- ✅ **Scope stage (reopened) — per-column searchable encoders** — `scoped_encodings`
  groups become `scope_<name>` choices applied via `.skb.apply(cols=…)` before the
  TableVectorizer; column names validated at resolve+build time, runtime selector is a
  missing-tolerant regex union (`_scope_selector`); Options 1/3 can target/inject into
  groups, and an Option-3 scoped proposal (`{"name", "cols"}`) creates a NEW group
  mid-search. (`tests/test_scope_stage.py`.)
- ✅ **prior_fn warm-start (free-form)** — options may carry `"prior": 0.0-1.0` in the
  plan (zero extra LLM calls); `resolve_spec` collects `spec["priors"]`; `_make_prior_fn`
  seeds EVERY fresh child (neutral 0.5 default — seeding only rated children would invert
  the prior under UCT's inf-for-unvisited rule). (`tests/test_priors_and_proposer.py`.)
- ✅ **Richer Option-3 proposer** — the one-call-per-step prompt now carries the full
  cross-stage tree ledger, the incumbent config+score, the data digest and real column
  names; plan_author is instructed to give generous (registry-clipped) HP ranges up front.
- ✅ **Top-k ensemble (thin read-off)** — `ensemble.top_k_states` ranks the persisted
  score cache; `evaluate_top_k` fits each config on a seeded split and soft-votes /
  averages on the holdout; `--top-k` / `TOP_K=` reports ensemble vs incumbent.
  (`tests/test_ensemble.py`.)
- ✅ **Search reward can adopt a bounded task metric** — `search_scorer(task_type, metric)`
  returns roc_auc/f1/etc. when the task metric is already bounded higher-is-better
  (accuracy is blind on 1.25%-positive credit-fraud).
- ✅ **Stratified CV for imbalanced classification (bug found during the live credit-fraud
  smoke)** — skrub's `cross_validate` calls `check_cv(cv)` without `y`/`classifier`, so it
  always defaults to plain `KFold`, even for classifiers. On a subsampled ~1%-positive
  target a random fold can land on zero positives; sklearn's scorer then silently NaNs
  that fold (no exception) and the NaN reward would poison the score cache. Fixed via
  `skrub_ops._cv_kwarg`/`stratify=` threaded through `make_rollout_fn`/`evaluate_full`/
  `run_search_loop`/`pipeline.py`, auto-enabled for `task_type == "classification"`.
  (`tests/test_relational_pipeline.py::test_stratified_rollout_avoids_nan_on_rare_target`.)
- ✅ **`get_default_state` root-state bug (found during the SAME live smoke, more serious)**
  — when a list-based stage (`encoder_options`, `clean_options`, `stages`) has an HP-tuned
  entry (a nested choice), skrub's own `describe_defaults()` abbreviates it to
  `"ClassName(...)"`, which never matches `get_action_space`'s full-repr label. The old
  reconciliation silently stored that unappliable abbreviated string as the search ROOT
  state, so `apply_state` raised on it and **every rollout in the run scored 0.0** —
  observed live once the plan_author prompt (Item 3) started asking for generous HP
  ranges on non-model operators too, e.g. `encoder_options`. Fixed: discrete defaults now
  come from `get_action_space(plan)[name][0]` (skrub's `Choice.default` is always
  `outcomes[0]`, verified against `_choosing.py`), never from the describe_defaults()
  string. (`tests/test_scope_stage.py::test_default_state_is_appliable_when_encoder_options_have_tuned_hps`.)

**Week 2 — follow-ups (landed 2026-07-07, all offline-tested)**
- ✅ **HP-refinement bonus phase** *(superseded 2026-07-10 by the focused-refinement
  phase — Pre-ship upgrades item 2 — which explores ALL single-edit neighbors of the
  incumbent, not just its HPs, and reports `refined_dims` instead of `hp_refined`)* —
  after the main budget, `run_search_loop`
  spends `ceil(total/4)` extra rollouts starting selection at the incumbent
  node (`mcts.start_node` local descent; UCT + backprop unchanged) and
  restricting expansion to the incumbent model's *active, still-untested*
  gated HPs (`_incumbent_hp_targets`; `mcts.expand` now accepts a set of
  target keys). Gating guarantees it only ever tunes the incumbent's own
  model, and it is a strict no-op when no untested HP space remains (bare,
  non-tuned plans), so it never disturbs the existing bare-spec invariants.
  This is the "biased HP exploration around the best config" the targeting
  logic was originally conceived for. Off switch: `hp_refine=False`; refined
  dims surface as `result["hp_refined"]`.
  (`tests/test_search_loop.py::test_hp_bonus_phase_refines_incumbent_after_main_budget`,
  `::test_hp_bonus_phase_is_noop_and_off_switch`.)
- ✅ **Scoped groups v2: position + additive** — `scoped_encodings` entries take
  `"position": "pre_encode"|"post_encode"` (post-encode groups apply after the
  TableVectorizer — numeric passthrough names only) and `"additive": true`
  (the operator runs on a selected copy and its output — suffixed
  `__<group>` — is concatenated back by row index, so originals survive; skip
  is `SelectCols([])`, an empty frame, since `None`-passthrough would duplicate
  columns). The LLM declares *intent* (in-place vs additive) in the plan;
  the concat/rename structure is code-owned — names never enter the LLM
  contract. Resolver validates both flags; Option-3 scoped proposals may carry
  them, so an injected group can be additive/post-encode. `skrub.DatetimeEncoder`
  added to the registry (the "extract date parts additively" case).
  (`tests/test_scope_stage.py` — v2 section.)
- ✅ **Interleaved propose between fixed-budget slices** — `run_search_loop` now
  re-mines the tree and re-targets after *every* slice (`retarget=False` keeps
  the first pick, A/B-able in sweeps), and with a proposer fires one call
  before each slice after the first: `outer_steps=4, budget_per_step=15` =
  `search 15 → propose → 15 → propose → 15 → propose → 15` on one persisted
  tree (≤ `outer_steps-1` LLM calls; the invariant holds). CLI sugar
  `--n-proposes N` / `make run-live N_PROPOSES=N` = `outer_steps N+1` + refine.
  (`tests/test_search_loop.py` — interleave section.)

Run all: `uv run python -m pytest tests/ -q`  (or `make test`)
Run live pipeline: `make run-live BUDGET=20` (small) / `BUDGET=80` (large) · with targeting+injection: `make run-refine`

---

## What's left to implement — prioritized, with timeline

We are in **Week 3 (evaluation + writeup)**; deadline **Tue Jul 15, 18:00**. Weeks 1–2
landed on schedule (feature freeze passed 2026-07-09; pre-ship upgrades 2026-07-10), so
what remains is the evaluation, figures and writeup below. Judged by the invariant: *reduces rollouts / adds a real flexibility
axis without moving the LLM into the inner loop.*

### Week 2 — second axis + amplifiers + experimentation (Jul 3 – Jul 9)

| Item | Priority | Status | Est. |
|---|---|---|---|
| **Relational assemble — end-to-end auto-config** | **High** (differentiator) | ✅ **done Jul 5** (AggTarget stretch not taken) | Mon–Tue (Jul 3–4) |
| **Scope stage — per-column encoders** (reopened by decision) | High | ✅ **done Jul 5** (`scoped_encodings`) | — |
| **`prior_fn` warm-start (free form)** | Medium | ✅ **done Jul 5** | Wed (Jul 5) |
| **Top-k ensemble** | Medium | ✅ **done Jul 5** (`ensemble.py`) | Thu (Jul 6) |
| **Richer Option-3 proposer** (evidence context + scoped proposals) | Medium | ✅ **done Jul 5** | — |
| **Experimentation harness + sweeps** | **High** (A-grade) | ✅ **done Jul 7** (`sweep.py`, JSON specs, spec reuse) | Fri + weekend (Jul 7–9) |
| **Imbalance-safe rollouts** (stratified subsample + minority floor) | High (fixes noisy credit-fraud rewards) | ✅ **done Jul 7** | — |

Remaining notes:
- **AggTarget** (target-based aggregation as a guarded assemble option) was the explicit
  stretch and was *not* taken; skrub's `AggTarget` takes `y` at fit, so inside skrub CV it
  is fold-safe by construction — cheap to add later if evaluation wants it.
- ~~**Live validation still owed**~~ ✅ done 2026-07-05: `make run-refine TASK=credit-fraud
  BUDGET=15 TOP_K=3` completed cleanly — best search score (roc_auc) 0.5850, held-out
  report 0.5781, top-3 ensemble 0.6032 (beats the single incumbent). Assemble stayed
  `"skip"` in the final incumbent this run (explored but not converged on within
  budget 15×3) — a longer/bigger live run is the natural next check, not a bug.
- ~~**Known residual issue:** some CV folds raise `ValueError: y should be a 1d
  array, got shape (100, 2)` from the `roc_auc` scorer.~~ ✅ Root-caused and fixed
  2026-07-08: skrub's learner reports `_estimator_type == "transformer"`, so sklearn's
  builtin scorer never reduced a binary `predict_proba` to the positive column —
  proba-only classifiers (RF/LGBM/XGB) silently scored 0.0 on roc_auc tasks. Fixed in
  `skrub_ops._resolve_scoring` (positive-column callable). See
  [BUG_LEDGER.md](BUG_LEDGER.md).

4. ~~**Experimentation harness + sweeps (the A-grade differentiator).**~~ ✅ done Jul 7:
   `machine_learning_engineering/sweep.py` runs a JSON sweep spec (`sweeps/example.json`;
   `defaults` + `runs`, list values cartesian-expand per entry, `n_proposes` sugar) via
   `make sweep SWEEP=<file>` → `runs/sweep_<ts>/{sweep.csv,sweep.md,<slug>/}`. Key design:
   **agents run once per task** — the raw spec is captured and reused across every point
   (`run_pipeline(spec_raw=...)`), so LLM calls stay O(tasks) not O(runs) (the checked-in
   example: 24 runs, 14 calls — free tier is ~250/day; wall-clock CV is the binding
   constraint). `--spec-cache` shares fetches across sweep invocations; quota errors get
   one retry, other failures become `status="failed"` rows and the sweep continues.
   `c`, `retarget`, `seed` are now also plumbed through `run_pipeline` and the pipeline
   CLI (`--c`, `--no-retarget`, `--seed`). *Still owed for Week 3:* the 2–3 sweep
   figures with one-sentence findings, generated from `sweep.csv`.

5. **Imbalance-safe rollouts (done Jul 7).** The live credit-fraud scores (~0.585) were
   near-noise: a plain 500-row subsample of a 1.25%-positive table holds ~6 positives, so
   StratifiedKFold(5) tested on 1–2 per fold and bad draws degenerated folds
   (roc_auc → NaN). `make_rollout_fn(target=...)` now uses `_stratified_subsample`
   (per-class seeded sampling with a minority floor of 10 real rows, never duplicated —
   duplication would leak across folds), shrinks `n_splits` when the subsampled minority
   is still smaller than the fold count, and averages folds with `_fold_mean`
   (NaN-skipping, all-NaN → 0.0). The ensemble holdout is likewise stratified for
   classification. Re-run credit-fraud live to re-baseline before the Week-3 figures.

> **Hard feature freeze: end of Week 2 (Sun Jul 9).** After this — evaluation, debugging, writing only.

### Pre-ship quality upgrades (decided 2026-07-09, ✅ landed 2026-07-10)

Surfaced by the budget-80 full-suite re-run, where **more budget regressed
credit-fraud/midwest** vs budget-20 — evidence the search *mechanics* needed
work, not more rollouts. All three stayed invariant-safe (≤ O(outer steps) LLM
calls). All offline-tested; re-run the live/offline eval before the Week-3
figures.

1. ✅ **Whole-plan extending injection (replaced stage-targeted Option 3).**
   The proposer now sees the whole current raw plan + search evidence and
   returns a NEW plan that *extends* it (any/multiple stages, **with HP
   ranges**). `_merge_raw_plans` unions it strictly additively (existing
   entries are never modified — tree states stay appliable), the merged plan
   re-resolves through `resolve_spec` (same import-allow-list envelope), the
   plan is rebuilt, and the action-space diff lands in `injected_options`.
   Injected LGBM/XGB now enter TUNED, competing with tuned HGB (the old
   param-less `_inject` couldn't). New contract `propose(plan_json, context)
   -> dict|None`; `make_llm_proposer` + `scripts/claude_agents.
   make_replay_proposer` both rewritten; prerequisite `_SanitizeColumns`
   (booster-safe column renames, invisible to the action space) added to
   `build_staged_plan`. (`tests/test_search_loop.py` merge/inject section,
   `tests/test_sanitize_columns.py`.)
2. ✅ **Focused refinement search, not HP-targeted.** The bonus phase keeps
   `start_node=best_node` local descent but explores ALL single-edit
   neighbors of the incumbent (encoder/scale/assemble/HPs alike);
   `_incumbent_hp_targets` deleted; result key renamed `hp_refined` →
   `refined_dims` (the dims actually edited during the phase). The old
   HP-only phase no-op'd once the HP grid was exhausted (`hp_refined=[]` on
   every task at budget 80). (`tests/test_search_loop.py` bonus-phase tests.)
3. ✅ **Structure-aware subsampling + seed-averaged rewards.**
   `skrub_ops._profile_subsample_n` sizes the rollout subsample from the data
   profile: imbalance grows it toward `min(minority, 40) / prevalence`
   (capped at 2000 rows for wall-clock), high-cardinality text bumps it to
   ≥1000; the stratified minority floor rose from 10 to up to 40 real rows.
   `make_rollout_fn(n_subsample_seeds=)` averages the reward over several
   seeded subsamples (deterministic, cache-exact; timeout scales);
   `pipeline._auto_subsample_seeds` auto-enables 3 seeds for imbalanced
   classification (minority < 5%); `--subsample-seeds` on the CLI and
   `subsample_seeds` in sweep specs. (`tests/test_profile_subsample.py`.)

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
- ~~Acquire/stage the relational dataset (`fetch_credit_fraud`) under `tasks/`.~~ ✅ done
  (`make stage-credit-fraud` → `tasks/credit-fraud/`).
- Stable live-eval runs once Gemini quota is steady.

### Explicitly cut (do not reopen)

- Full MLE-STAR iterative ensembler (the thin top-k read-off in `ensemble.py` replaces it).
- ArchPilot-style restart, mid-search GEN/progressive-widening — wrong fit (our scorer is fixed), wrong cost class.
- `post-process` skrub stage, per-model training loss as a searchable HP — future work.
  (`scope` was on this list but was **reopened by decision and shipped 2026-07-05** as
  `scoped_encodings`.)
- Per-expansion or per-rollout LLM calls — the invariant; rejected regardless of per-call cost.

### Fallback

If relational assemble isn't stable end-to-end by midweek 2, ship **Option 1 + Option 3 +
top-k ensemble** as the evaluated result and present relational as designed/partially-built
future work. **Protect Week-3 evaluation time above any single feature.**

---

## Key design decisions (for slides / Q&A)

- **Structured JSON plan, not code-gen** — no `eval` of model output, central
  seeding (determinism MCTS needs), schema-validated, MCTS-friendly named choices.
- **Import-level allow-list** (not dynamic import) — the safety envelope is the
  import root (`sklearn`/`skrub`/`lightgbm`/`xgboost`), so a hallucinated/unlisted operator is dropped,
  never executed. HP ranges are free-form (used as given) unless the param is
  curated in `REGISTRY` (then clipped); a param the class can't accept, or an
  RNG-identity param, is dropped individually without dropping the operator.
  Option 3's injected paths go through the same import gate.
- **Two scorers** — bounded higher-is-better reward for *search*; the competition
  metric only for the *final report* (they must not be conflated).
- **Persist the tree, don't restart** — our scorer is fixed, so a config's reward never
  changes when the target moves; the tree is a running ablation we mine for free.
- **LLM-call complexity is the cost model** — O(1) per task, ≤ O(outer steps) with Option 3;
  never O(expansions) or O(rollouts). The single real cost is CV rollouts.
- **One ADK stack, env-switched provider, shared logic** — `ROOT_AGENT_MODEL`
  selects native Gemini or OpenAI/compatible (`LiteLlm`) with no code change;
  the MCTS/skrub core is client-agnostic and import-clean.
</content>
</invoke>
