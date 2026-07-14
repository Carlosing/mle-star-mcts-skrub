# Project state & roadmap

**Thesis:** adapt MLE-STAR into an **MCTS search over skrub DataOps pipelines**.
LLM agents read the data and author a *rich JSON plan* (a menu of operators +
hyperparameter ranges per stage); a pure-code MCTS engine then searches that
space — structure **and** hyperparameters — over a fixed evaluation budget. The
LLM never does the search; it only proposes the space (**O(1) LLM calls per
task**, plus at most one call between search slices for Option 3 — never inside
the inner search loop).

_Last updated: **2026-07-14 — project hand-off.** The code is feature-complete and
the offline suite is green. The **shipped results are knowingly biased in our own
favour** and must be reported as such: they predate the on-disk train/holdout split
(commit `517f954`), so the search selected its configs with the eval rows visible
while AutoGluon's did not. The fix is in the code but there was no time to re-run
the benchmark. Read
[Hand-off — the published numbers are biased](#hand-off--the-published-numbers-are-biased)
before you write a single number down._

_History: the search core (MCTS + skrub + the two-agent plan) froze on 2026-07-09;
pre-ship quality upgrades landed 07-10 (whole-plan extending injection,
focused-refinement bonus phase, structure-aware subsampling, `_SanitizeColumns`);
robustness + a second provider on 07-11 (truncated-JSON rejection, default-in-grid
numeric options, CV fold-parallelism `n_jobs=6`, `PROVIDER=google|school`
live-validated on GWDG's qwen3.5-397b). Weeks of 07-12→07-14 were **evaluation
plumbing and evaluation-integrity fixes**, not new search features: always-on
skrub backbones, the Caruana ensemble, the benchmark harness (AutoGluon + a
revived MLE-STAR), and the shared on-disk bench — see
[Landed after the freeze](#landed-after-the-freeze-jul-10--jul-14). Historical
bugs (symptom → cause → fix → test) are catalogued in
[BUG_LEDGER.md](BUG_LEDGER.md); post-freeze ideas that were deliberately NOT
implemented are in [FUTURE_WORK.md](FUTURE_WORK.md)._

## Hand-off — the published numbers are biased

**Decision (2026-07-14): we ship the existing results and disclose the bias,
rather than re-run the benchmark.** There was no time to re-run before the
deadline. This section is the record of *exactly* what the bias is, so the
writeup can state it precisely instead of hand-waving.

### What is and isn't wrong with the archived results

They are **not** garbage, and they are **not** incomparable. The two arms are on a
common bench:

- Both the extension and AutoGluon scored the **identical rows** — both called
  `ensemble.holdout_split(train.csv, target, task_type, seed=42, frac=0.25)`, the
  same function with the same seed, and `train.csv` was byte-stable across every
  archived run (it last changed 2026-07-10; the earliest archived run is 07-12).
- Both **fit on the same 75%** and scored the same 25%. The final-fit protocol is
  symmetric.

The asymmetry is in **model selection**, and it is one-directional:

> `pipeline.run_pipeline` handed the **full** `train.csv` to `run_search_loop`, so
> every MCTS rollout cross-validated over rows that were also in the 25% holdout.
> The extension therefore *chose* its pipeline, model and hyperparameters with the
> eval rows visible. AutoGluon chose its models having only ever seen the 75%.

So the archived extension score is **optimistically biased**; AutoGluon's is
clean. The reported extension−AutoGluon delta is an **upper bound** on the true
delta. (Full write-up: BUG_LEDGER #25–#28.)

### How to report it

- ✅ **Quote `result["holdout"]["score"]`** — the *incumbent* on the shared 25%.
  This is already what `scripts/make_figures.py` plots, and it carries **only**
  the selection bias above.
- ❌ **Do NOT quote `ensemble_score`** from the 28 Caruana-era archived runs
  (anything after `4490d2e`, 07-13). There, `_caruana_select` greedily *maximised*
  the holdout score and we then published that maximum — a **second, larger** bias
  stacked on the first, and precisely the ensemble-overfitting flaw our own report
  criticises MLE-STAR for (§8.3). The 6 pre-Caruana runs used an unweighted top-k
  mean and carry only the selection bias, but the simplest safe rule is: **report
  the incumbent, not the ensemble.**
- State the bias **direction and cause**, and that its **magnitude is unmeasured**.
  It is honest to write "optimistic, unquantified, upper bound"; it is not honest
  to write "small" or "negligible" — nobody measured it.
- **MLE-STAR has 0 runs.** It was never executed end-to-end. Do not put a number
  in its column; the harness (caps + scorer) is tested but unused.

### The cheap way out, if any time appears

Re-running the extension does **not** cost API quota: `scripts/replay_from_run.py`
/ `run_pipeline(spec_raw=…)` reuse a run's captured plan, so a replay on the fixed
split is pure CPU and zero LLM calls (~3 min/arm on country-happiness). Re-running
AutoGluon costs nothing but CPU either. Doing **three fast tasks** (country-happiness,
california-housing-prices, toxicity) on the fixed split would turn "unquantified
bias" into "bias measured at X on a 3-task sample" — a materially stronger claim
for a few hours of compute and no quota. Protocol in
[USAGE.md § Benchmark comparison](USAGE.md#benchmark-comparison-extension-vs-autogluon-vs-mle-star).

### Also still true

- **Any *fresh* run is clean** — the code is fixed. On a new run, check
  `ensemble.selection == "oof_3fold"` (not `legacy_holdout`) before quoting it.
- **Wide high-cardinality tasks remain intractable at a real budget**
  (traffic-violations projected ~15h; GapEncoder forfeits on the 60s rollout cap).
  See [BUG_LEDGER § Open / flagged](BUG_LEDGER.md#open--flagged-not-fixed).

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
6. **run_search_loop** → persisted-tree MCTS over `budget` rollouts (score cache + gated HPs); `outer_steps>1` runs the budget as fixed-size slices on one tree, re-mining the tree between slices to pick a focus stage — a proposer hint only, expansion is never locked (Option 1, `retarget=`) — and, with a proposer, asking for a whole *extended plan* before each subsequent slice (Option 3 — `search → propose → search → …`; the extension is merged strictly additively via `_merge_raw_plans`, re-resolved, and rebuilt, so injected operators arrive with their HP ranges). After the budget, a **focused-refinement bonus phase** (`ceil(total/4)` rollouts, no LLM) descends from the incumbent node and explores ALL of its single-edit neighbors — structure and HPs alike (`refinement_phase=`, dims reported as `refined_dims`).
7. **ensemble** (`top_k > 1`) → Caruana greedy selection over a candidate pool of the
   top `max(2*top_k, ensemble_pool)` distinct cache configs. Members are **selected**
   on 3-fold out-of-fold predictions spanning all of train, then **refit on all of
   train and scored on the shared holdout** — selection rows and reported rows are
   never the same rows (`ensemble.selection` records which logic ran).
8. **report** → score the incumbent on the competition metric (RMSE, etc.) by CV
   (`report`) *and* on the shared on-disk holdout (`holdout` — the cross-method
   comparable number); write `runs/<task>_<ts>/{result.json,summary.md,ensemble.pkl}`.

## Module map

| File | Role |
|---|---|
| `mcts.py` | MCTS engine (UCT select/expand/backprop, **persistent tree, score cache, `gating`/`target_key` in `expand`, `canonicalize`**, `prior_fn` hook) |
| `skrub_ops.py` | skrub glue: `build_staged_plan` (incl. relational `assemble` + a fixed `_SanitizeColumns` rename before the model so boosters accept skrub's special-character feature names), `get_action_space`, **`get_choice_gating`**, `apply_state`, seeded rollouts (configurable `scoring`; profile-aware subsample sizing `_profile_subsample_n` — imbalance/high-cardinality grow it, capped at 2000; optional seed-averaged rewards `n_subsample_seeds`; per-rollout 60s wall-clock cap via `_time_limit`/`timeout_s` → slow config scores 0.0; CV folds parallel via `n_jobs`, default 6), `run_ablation`, `pick_target_node` |
| `search_loop.py` | **outer loop**: persisted MCTS as fixed-budget slices; `tree_action_values` (tree-mined ablation), Option 1 focus pick re-run between slices (`retarget=`; a proposer hint, never an expansion lock), Option 3 **whole-plan extending injection** between slices (`propose(plan_json, context) -> extended plan`, merged strictly additively via `_merge_raw_plans`, re-resolved + rebuilt — injected operators arrive tuned), `make_llm_proposer` (≤ one provider call between slices — native Gemini or OpenAI-compatible, same env switch as the agents); post-budget **focused-refinement bonus phase** (`ceil(total/4)` rollouts from the incumbent node over ALL its single-edit neighbors — `refinement_phase=`, edited dims in `refined_dims`) |
| `spec_resolver.py` | LLM JSON → seeded estimators **+ HP `choose_*`**; **import** allow-list (roots `sklearn`/`skrub`/`lightgbm`/`xgboost`) is the safety envelope; free-form HP ranges (`_build_free_choice`) unless curated in `REGISTRY` (then clipped); per-param safety nets (`_accepts_param`, `_RNG_PARAMS`) drop unknown/RNG params without dropping the operator; `assemble` passthrough |
| `data_summary.py` | `make_data_summary` — EDA digest for the analyst |
| `adk_agent.py` | ADK graph `data_analyst → plan_author`; `build_root_agent` factory; `_resolve_model` env-switches native Gemini vs `LiteLlm` (OpenAI/compatible); `google_search` attached only on the Gemini path |
| `metrics.py` | search-reward scorer (per task; adopts a bounded task metric like roc_auc) vs report metric (competition) |
| `ensemble.py` | **Caruana ensemble read-off** over the persisted score cache (no new search): `top_k_states` ranks the cache (deduped on the defaults-filled *effective* state), `_caruana_select` greedily builds a **weighted** ensemble with replacement + early stop, `evaluate_top_k` keeps **selection** (3-fold OOF over all of train) and **reporting** (refit on train, score the shared holdout) on disjoint rows and stamps `selection`; `holdout_split` is the seeded split helper; `EnsemblePredictor` is the fitted, picklable result (`ensemble.pkl`) |
| `pipeline.py` | end-to-end driver + CLI (`--budget`, `--outer-steps`, `--refine`, `--top-k`, `--n-proposes`, `--n-jobs`, `--time-budget-s`, `--legacy-ensemble`); multi-table `load_task` (`aux_*.csv`, reads **only `train.csv`**); `load_holdout` (the shared on-disk bench); `save_run_artifacts` (`result.json` + `summary.md` + `ensemble.pkl`) |
| `runner.py` | minimal OpenAI-compatible runner used by the **revived MLE-STAR baseline** in place of the ADK runtime; `llm_call` is the single chokepoint the benchmark harness wraps to impose call/token/time caps |
| `run_logging.py` | sanity log of prompts+outputs to JSONL (`log_dir`), incl. per-call `tokens` |
| `agent.py`, `sub_agents/`, `shared_libraries/`, `eval/` | the **upstream MLE-STAR agent, revived as the benchmark baseline** (driven by `scripts/run_mlestar.py` through `runner.py`, hard-capped). It is *not* on the MCTS path and no MCTS feature belongs here — but it is no longer dead code: it is the thing we compare against. `shared_libraries/web_search_util.py` (DuckDuckGo) backs its model-retrieval step |
| `probe_gemini.py`, `probe_school.py` | standalone model/quota probes (Gemini; GWDG school endpoint) |

### Benchmark + evaluation scripts

| File | Role |
|---|---|
| `scripts/stage_tasks.py` | **Draws the one and only train/holdout split, on disk** (`make stage-tasks`): `train.csv` / `test.csv` / `test_answer.csv` per task, 80/20 seeded, stratified for classification. Also encodes the per-dataset knowledge that can't be inferred (leaky columns, subsample size, aux joins) |
| `scripts/stage_credit_fraud.py` | downloads + stages the relational credit-fraud task (`make stage-credit-fraud`, online, once) |
| `scripts/run_autogluon.py` | AutoGluon baseline on the **same** holdout + time budget (`make bench-autogluon`). Flat-table only — it never sees `aux_*.csv`, which is exactly the extension's relational advantage. `NUM_CPUS=1` required on macOS-ARM (libomp) |
| `scripts/run_mlestar.py` | the revived MLE-STAR under hard caps (`make bench-mlestar`) — max LLM calls, per-call token bound, wall-clock deadline. Its token cost is otherwise *unbounded*; treat its result as one data point, not a curve |
| `scripts/claude_agents.py`, `scripts/run_claude_pipeline.py` | offline Claude-authored plans + replay proposer → full runs at **zero API quota** (`make run-claude` / `make sweep-claude`) |
| `scripts/replay_from_run.py` | re-runs the search from a stored run's captured plan (`spec_raw`), for A/Bs at zero agent cost |
| `scripts/collect_results.py` | copies `result.json` artifacts from `runs/` into the small git-shareable `results/` mirror (leaves AutoGluon's multi-GB `ag_models/` behind) |
| `scripts/make_figures.py` | renders `figures/`: `quality_at_cost.png`, `time_scaling.png`, `mechanism_table.md`, `comparison.csv`. All three methods emit the same `result.json` schema (`method`, `holdout`, `tokens`, `llm_calls`, `wall_clock_s`), so it reads them uniformly |

---

## Current state — done & tested ✅

**The Week-1 spine is complete and green.** Everything below has offline tests.

**Search-quality core (Week 1)**
- ✅ **MCTS engine** — UCT select/expand/backprop, persistent tree.
- ✅ **Score cache** — `mcts.score_cache` memoizes `state_key → reward`; deterministic
  rollouts make it exact, so each distinct config is evaluated at most once
  (`test_score_cache_one_call_per_distinct_state`, and asserted across outer steps).
- ✅ **Conditional (model-gated) HP nesting (CASH fix)** — `get_choice_gating` reads
  skrub's conditional-children graph; `expand` only edits an HP when its parent model
  is selected, and `canonicalize` drops inactive HPs so the cache/dedup don't split on
  them (`test_gating_skips_inactive_hp_and_canonicalizes`,
  `test_run_states_are_model_gated_canonical`).
- ✅ **Option 1 — tree-mined ablation → proposer hint** — `search_loop.tree_action_values`
  mines per-stage deltas from the persisted tree (no fresh rollouts), `pick_target_node`
  chooses the highest-variance stage, and the pick is forwarded to the Option-3 proposer
  as its `target_stage` hint (`test_targeting_picks_an_operator_stage`,
  `test_targeting_is_a_proposer_hint_not_an_expansion_lock`). Expansion is *never*
  locked to the pick (changed 2026-07-13): replay A/Bs on country-happiness and
  california-housing showed the variance ledger elects `model` on essentially every
  pick — the retarget on/off axis was bit-identical — while the lock starved off-target
  injected options until the bonus phase (on country-happiness the *winning* edit, an
  injected QuantileTransformer, was exactly such a rescue). The `mcts.expand` lock
  capability itself remains (`target_key`, single name or set —
  `test_target_key_restricts_expansion`), just unused by the outer loop. The
  *post-budget* local search around the incumbent is the **focused-refinement bonus
  phase** (see Pre-ship upgrades item 2; `mcts_search` takes a `start_node`).
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
- ✅ **Top-k ensemble (thin read-off)** *(superseded 2026-07-13/14 — the unweighted
  top-k mean is now a **Caruana** greedy weighted selection over a wider pool, and the
  members are selected on out-of-fold rows rather than the reported holdout; see
  [Landed after the freeze](#landed-after-the-freeze-jul-10--jul-14). The old combiner
  still runs on the same fits and is returned as `ensemble_score_mean`, so every run is
  a free A/B.)* — `ensemble.top_k_states` ranks the persisted score cache;
  `evaluate_top_k` fits the pool and scores the holdout; `--top-k` / `TOP_K=` reports
  ensemble vs incumbent. (`tests/test_ensemble.py`.)
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
  — when a list-based stage (a `vectorizer`/`cleaner` backbone slot, `stages`) has an
  HP-tuned entry (a nested choice), skrub's own `describe_defaults()` abbreviates it to
  `"ClassName(...)"`, which never matches `get_action_space`'s full-repr label. The old
  reconciliation silently stored that unappliable abbreviated string as the search ROOT
  state, so `apply_state` raised on it and **every rollout in the run scored 0.0** —
  observed live once the plan_author prompt (Item 3) started asking for generous HP
  ranges on non-model operators too. Fixed: discrete defaults now come from
  `get_action_space(plan)[name][0]` (skrub's `Choice.default` is always
  `outcomes[0]`, verified against `_choosing.py`), never from the describe_defaults()
  string. (`tests/test_scope_stage.py::test_default_state_is_appliable_when_slot_options_have_tuned_hps`.)

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
  logic was originally conceived for. Off switch: `hp_refine=False` (kwarg
  renamed `refinement_phase=` on 2026-07-13); refined dims surfaced as
  `result["hp_refined"]`.
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
  re-mines the tree and re-picks the proposer's focus-stage hint after *every*
  slice (`retarget=False` keeps the first pick — though the replay A/Bs of
  2026-07-13 found the pick is `model` on essentially every task, making that
  axis a null comparison), and with a proposer fires one call
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
| **Experimentation harness + sweeps** | **High** (A-grade) | ✅ done Jul 7 → **removed Jul 13** (unused: budget/N_PROPOSES are chosen by time + LLM-call count, not swept) | Fri + weekend (Jul 7–9) |
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

4. ~~**Experimentation harness + sweeps.**~~ ✅ done Jul 7, **removed Jul 13**
   (`sweep.py`, `sweeps/`, `make sweep`/`sweep-live`, `tests/test_sweep.py`): in practice
   budget and `N_PROPOSES` are chosen by wall-clock time and LLM-call count, not by
   sweeping the search's own hyperparameters — and the one axis the harness was built to
   A/B (`retarget`) turned out to be a null comparison (see Option 1 above). What
   survives: `run_pipeline(spec_raw=...)` spec reuse (used by the replay/Claude drivers),
   the `--c`/`--no-retarget`/`--seed` CLI plumbing, and the time-scaling figure —
   `scripts/make_figures.py` now reads it from extension `result.json` artifacts at
   different budgets instead of a `sweep.csv`.

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
   classification (minority < 5%); `--subsample-seeds` on the CLI.
   (`tests/test_profile_subsample.py`.)

### Landed after the freeze (Jul 10 – Jul 14)

Everything here is **evaluation plumbing or an integrity fix** — no new search
features, and the LLM-call invariant was never touched.

| Landed | What | Why it matters |
|---|---|---|
| **Jul 12** | **Always-on skrub backbones** (`8cc66f9`) — `clean`/`encode` are now always a bare `Cleaner()` + `TableVectorizer()` (skrub's own defaults = the robust root). The register-operator default menus were deleted; their knobs are now LLM-authored like any HP (`cleaner`/`vectorizer` spec keys). Scalar `choice` lists are reordered **default-first** so the root reproduces stock behaviour (trap: `skrub.choose_bool()` defaults to `True`) | The old code-owned fallback menus were arbitrary; the space is now entirely LLM-proposed, code-validated |
| **Jul 12** | **TextEncoder backbone cache + JSON parse safety nets** (`3c00e95`) — the e5-small-v2 backbone is cached per `model_name` (1 load per run, not per rollout); `parse_spec_json` repairs Python literals (`None`/`True`) and balanced-but-broken JSON, still rejecting truncation | qwen emitted Python `None` and silently fell back to a bare plan; the fix recovered the rich plan (traffic-violations holdout 0.882 vs 0.837) |
| **Jul 13** | **Retarget lock removed** (`42536c6`) — Option 1's stage pick is now a **proposer hint only**; expansion is never locked to it | Replay A/Bs showed the variance ledger elects `model` on essentially every pick (a null A/B axis), while the lock *starved* off-target injections until the bonus phase — on country-happiness the winning edit was exactly such a rescue (−753.87 → −716.95). `mcts.expand`'s `target_key` capability remains, just unused |
| **Jul 13** | **Sweep harness removed** (`42536c6`, `ed932c4`) — `sweep.py`, `sweeps/`, `make sweep*`, `tests/test_sweep.py` | Budget and `N_PROPOSES` are chosen by wall-clock and LLM-call count, not by sweeping the search's own HPs — and the one axis it existed to A/B (`retarget`) was null. What survives: `run_pipeline(spec_raw=…)` spec reuse, the `--c`/`--seed` CLI plumbing, and the time-scaling figure (now read from `result.json` artifacts at different budgets) |
| **Jul 13** | **Caruana ensemble** (`4490d2e`) — greedy weighted selection with replacement + early stop over a **pool** of the top `max(2*top_k, 10)` distinct configs, replacing the unweighted top-k mean. Plus `_ImputeNumeric` (post-aggregation NaNs were silently failing models) | The pool reaches *past* the incumbent's near-duplicate cousins for real model-family diversity; the early stop collapses a duplicate pool to one member, so it is never worse than the incumbent. `ensemble_score_mean` returns the old combiner on the same fits — a free A/B every run |
| **Jul 14** | **Shared on-disk train/holdout split** (`517f954`) — the split is drawn by `stage_tasks.py` *before any method sees the data*; `load_task` reads only `train.csv` | **The integrity fix.** Previously the search CV'd over rows that later became its own eval set, so every extension-vs-AutoGluon delta was overstated. See BUG_LEDGER #25–#28 |
| **Jul 14** | **OOF ensemble selection** (`dbc0f77`) — Caruana selects on 3-fold out-of-fold predictions over *all* of train, then refits and scores the untouched holdout; `selection` stamped on every artifact | Selecting on the reported rows made `ensemble_score` a greedy maximum over the published metric — the exact flaw the MLE-STAR report criticises MLE-STAR for. Corollary: the ensemble is **no longer guaranteed** to beat the best pool member; that guarantee *was* the bias |
| **Jul 14** | **`--legacy-ensemble` flag, pickled `EnsemblePredictor`, root `figures/`** (`eb94f55`) | `LEGACY_ENSEMBLE=1` keeps pre-fix results reproducible and makes the two paths A/B-able on an identical split; `ensemble.pkl` is the fitted final model, built free from fits the reporting pass already did |
| **Jul 11–14** | **Benchmark harness** — `run_autogluon.py`, `run_mlestar.py` (revived MLE-STAR under hard caps), `collect_results.py`, `make_figures.py`, token instrumentation (`result["tokens"]`), `--time-budget-s` | The three-way comparison at the heart of the writeup. All arms emit one uniform `result.json` schema |

### What's next (in order)

1. **Write up the results with the bias disclosed.** Follow
   [How to report it](#how-to-report-it) to the letter: quote the incumbent
   `holdout` score, never the Caruana-era `ensemble_score`; state the bias as
   optimistic / unquantified / an upper bound on the delta; leave MLE-STAR's
   column empty.
2. **The headline comparisons.** (a) **relational lift** — assemble vs flat on
   credit-fraud / country-happiness, where AutoGluon is *structurally* blind to
   `aux_*.csv`. Note this one is **not** undermined by the bias: it is an
   architectural capability gap, not a score delta. (b) **time-scaling at constant
   LLM cost** — several budgets per task; also robust, since it is a within-method
   trend, and both ends carry the same bias. (c) **flexibility lift** — fixed space
   vs Option 3, likewise within-method. *The bias hurts the head-to-head
   quality-vs-AutoGluon number most, and the structural claims least — lead with
   the structural claims.*
3. **Writeup + slides:** the MLE-STAR comparison table (mechanism / debug cost /
   token cost / leakage handling / adaptivity) + the figures. The token-cost axis
   (constant vs unbounded) is measured and unaffected by the split bug.
4. **If time appears:** the zero-quota partial re-run described in
   [The cheap way out](#the-cheap-way-out-if-any-time-appears).

### Ops / housekeeping

- `uv lock` reconcile when online (`make sync`); local/Docker version parity.
- Benchmark deps are an extra, deliberately out of the core: `uv sync --extra bench`.
- The `results/` mirror is git-shareable by construction (`make collect-results`
  leaves AutoGluon's multi-GB `ag_models/` behind).

### Explicitly cut (do not reopen)

- Full MLE-STAR iterative ensembler (the Caruana read-off in `ensemble.py` replaces it —
  it reuses the search's own score cache, so it costs no new rollouts).
- ArchPilot-style restart, mid-search GEN/progressive-widening — wrong fit (our scorer is fixed), wrong cost class.
- `post-process` skrub stage, per-model training loss as a searchable HP — future work.
  (`scope` was on this list but was **reopened by decision and shipped 2026-07-05** as
  `scoped_encodings`.)
- Per-expansion or per-rollout LLM calls — the invariant; rejected regardless of per-call cost.
- The sweep harness (removed 2026-07-13) — do not rebuild it to A/B `retarget`; that
  axis was measured and is null.

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
- **The bench is drawn on disk, before any method runs** — not by convention inside
  each method. A method cannot be trusted-but-verified into honouring a holdout; the
  rows are simply absent from every file it can read. The same principle applies one
  level down: the ensemble **selects** on out-of-fold rows and **reports** on the
  holdout, because a number you fitted to is not a number you may publish.
- **Persist the tree, don't restart** — our scorer is fixed, so a config's reward never
  changes when the target moves; the tree is a running ablation we mine for free.
- **LLM-call complexity is the cost model** — O(1) per task, ≤ O(outer steps) with Option 3;
  never O(expansions) or O(rollouts). The single real cost is CV rollouts.
- **One ADK stack, env-switched provider, shared logic** — `ROOT_AGENT_MODEL`
  selects native Gemini or OpenAI/compatible (`LiteLlm`) with no code change;
  the MCTS/skrub core is client-agnostic and import-clean.
</content>
</invoke>
