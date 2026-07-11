# Bug ledger — historical bugs, where they lived, and the fix that shipped

A reference of the real bugs found during development (most surfaced on *live*
runs, not the offline suite), kept here so the code doesn't need bug-history
comments. Each entry: symptom → root cause → fix location → regression test.

Almost every entry shares one shape: **a failure inside skrub's CV is swallowed
by the rollout `try/except` (or NaN-ed by the scorer) → the config silently
scores 0.0 → search returns garbage without crashing.** When adding an operator
family or scorer, probe that boundary first (build a tiny plan, call the
rollout, assert reward > 0), then pin it in
[test_silent_zero_regressions.py](../tests/test_silent_zero_regressions.py).

## Rollout / scoring boundary (silent 0.0s)

| # | Bug | Root cause | Fix | Test |
|---|---|---|---|---|
| 1 | Every proba-only classifier (RF/LGBM/XGB) scored 0.0 on roc_auc tasks — incl. flagship credit-fraud | skrub's DataOps learner reports `_estimator_type == "transformer"`, so sklearn's builtin scorer never reduces a binary `predict_proba` (n,2) to the positive column; `roc_auc_score` raises per fold, skrub NaNs it | `skrub_ops._resolve_scoring` swaps proba-ranking metric *names* for a positive-column callable | `test_skrub_ops.py::test_roc_auc_scores_a_proba_only_classifier` |
| 2 | decision_function-only classifiers (LinearSVC, SGD-hinge) zeroed on roc_auc | the #1 shim *forced* `predict_proba` | the callable falls back to `decision_function` | `test_silent_zero_regressions.py` |
| 3 | Classification folds silently NaN on rare targets | skrub's `cross_validate` calls `check_cv(cv)` without `y`/`classifier` → always plain `KFold`; a subsampled fold can hold zero positives | `skrub_ops._cv_kwarg` passes an explicit seeded `StratifiedKFold`, auto-on for classification | `test_relational_pipeline.py::test_stratified_rollout_avoids_nan_on_rare_target` |
| 4 | Every rollout in a run scored 0.0 once plans HP-tuned non-model operators | `describe_defaults()` abbreviates an HP-tuned list entry to `"ClassName(...)"`, which never matches `get_action_space`'s full-repr label → unappliable root state | `skrub_ops.get_default_state` reads discrete defaults from `get_action_space(plan)[name][0]` (skrub's `Choice.default` is always `outcomes[0]`), never from the describe string | `test_scope_stage.py::test_default_state_is_appliable_when_encoder_options_have_tuned_hps` |
| 5 | Rewards near-noise on imbalanced data (credit-fraud ~0.585 ≈ chance); budget-80 *worse* than budget-20 | 500-row plain subsample of a 1.25%-positive table ≈ 6 positives → 1–2 per fold | `skrub_ops._stratified_subsample` (per-class seeded, minority floor, never duplicated — duplication leaks across folds), `_profile_subsample_n` (grow n toward floor/prevalence, cap 2000), `make_rollout_fn(n_subsample_seeds=)` seed-averaged rewards, adaptive `n_splits`, `_fold_mean` NaN-skipping | `test_profile_subsample.py` |
| 6 | Negative r2 leaked into backprop (violating the [0,1] reward invariant); the interim clamp flattened hard-regression landscapes (flight-delays, movielens) so UCT chose at random | r2 is unbounded below; clamping to 0 destroys ordering | `skrub_ops._bounded_reward` monotone squash `1/(2 − s)` for unbounded-below scorers; `_UNIT_SCORERS` pass through raw; 0.0 stays reserved for failures (all-NaN folds checked *before* the squash) | `test_silent_zero_regressions.py` |
| 7 | NaN in the target crashed plan construction / zeroed regression rollouts | NaN rows reach fit/score | dropped in `pipeline.load_task` and defensively in `skrub_ops.make_rollout_fn` | `test_silent_zero_regressions.py` |
| 8 | Assemble's `mode` aggregation option dead (0.0 vs 0.64+ for siblings) — credit-fraud's shipped `basket_mode` had never worked | on a modal tie `AggJoiner(operations="mode")` writes the tied *array* into the cell; the object column kills skrub's own TableVectorizer in `CleanNullStrings` | fixed `_ScalarizeAggregates` step right after the assemble stage in `skrub_ops.build_staged_plan` (invisible to the action space; ties broken by first element — pandas mode returns sorted) | `test_silent_zero_regressions.py` |
| 9 | Injected LightGBM always scored 0.0 | LightGBM rejects skrub's special-character feature names outright | fixed `_SanitizeColumns` rename step just before the model in `build_staged_plan` | `test_sanitize_columns.py` |

## Hard crashes (worse than 0.0 — they kill the whole run)

| # | Bug | Root cause | Fix | Test |
|---|---|---|---|---|
| 10 | LightGBM/XGBoost segfault (exit 139, uncatchable) inside skrub CV on macOS-ARM | a multi-threaded booster fit loads a second OpenMP runtime next to sklearn's bundled `libomp` | `n_jobs=1` in the `spec_resolver.REGISTRY` defaults for all four booster entries — **load-bearing, not a perf knob**. CV *fold* parallelism (`cross_validate(n_jobs=)`) is a different axis (inter-process forking) and is safe with boosters | `test_skrub_ops.py::test_booster_rollout_is_identical_and_crashfree_across_n_jobs` |
| 11 | Full test suite segfaulted while files passed standalone | `import torch` anywhere in the process (was on the import chain `adk_agent → run_logging → common_util`) loads a second OpenMP runtime; the *next* xgboost fit crashes. Order-dependent, and estimator `n_jobs=1` does not protect | torch made lazy inside `shared_libraries/common_util.set_random_seed` (only the legacy sub_agents path calls it). Never add torch (or another OpenMP-loading import) to any module reachable from `pipeline.py` | `test_silent_zero_regressions.py::test_pipeline_import_does_not_load_torch` |
| 12 | A single sparse-emitting transformer (`OneHotEncoder` default `sparse_output=True`) aborted the whole run at `build_staged_plan` | skrub DataOps carry pandas frames, which can't hold sparse matrices; skrub's eager preview raises at construction | `spec_resolver._SPARSE_PARAMS` forced False via `_names_param` — a *strict* signature check, because `_accepts_param` returns True for any `**kwargs` ctor and would have injected `sparse_output` into the boosters | `test_silent_zero_regressions.py` |
| 13 | XGBClassifier crashed `build_staged_plan` as default model, and zeroed every rollout on string-label tasks (open-payments, midwest-survey) | xgboost ≥ 1.6 dropped internal label encoding — `fit` requires integer `y` | `spec_resolver._xgb_classifier_shim()` — subclass *named* `XGBClassifier` (action-space labels and gating unchanged) that label-encodes in sorted order; original `classes_` exposed only after `super().fit` | `test_silent_zero_regressions.py` |

## Search mechanics

| # | Bug | Root cause | Fix | Test |
|---|---|---|---|---|
| 14 | The incumbent was stuck at the root (every HP at default) on credit-fraud and employee-salaries | numeric grids from linspace/geomspace hit the endpoints and skip the centre, but the root state seeds each dim at skrub's default = the midpoint → the root was unreachable by `expand` | each dim's default is inserted into its option grid in `skrub_ops.get_action_space`; guarded with `numbers.Number`, not `isinstance(x, (int, float))` — skrub returns `np.int64`, which is not a Python `int` | `test_skrub_ops.py` |
| 15 | Top-k "ensemble" fitted the same model k times (`ensemble_score == individual_scores[0]` exactly) | cache keys differ (`{"model": "LGBM"}` vs `{"model": "LGBM", hp: default}`) but `apply_state` resets omitted choices to defaults — one effective pipeline | `ensemble.top_k_states` fingerprints on the defaults-filled *effective* state | `test_ensemble.py` |
| 16 | Tasks with a numeric low-cardinality target (movielens' 10-value rating) searched `accuracy` over classifiers while reporting RMSE | `data_summary.infer_task_type`'s `nunique > 20` heuristic ignored the declared metric | `metrics.metric_task_type` overrules the heuristic when the dtype allows | `test_silent_zero_regressions.py` |
| 17 | HP-only refinement phase no-op'd (`hp_refined=[]` on every task at budget 80) | once the incumbent's HP grid was exhausted there was nothing left to expand | replaced by the **focused-refinement** phase — all single-edit neighbors of the incumbent, structure and HPs alike (`refined_dims`) | `test_search_loop.py` bonus-phase tests |

## LLM output parsing / resolution

| # | Bug | Root cause | Fix | Test |
|---|---|---|---|---|
| 18 | Option 3 silently no-op'd (`injected_options: []`, no error) | proposer responses truncated at `max_output_tokens` mid-object; the leftover text still holds valid JSON *fragments* and a brace-scan returned one (`{"float": [0.7, 1.0]}` became "the extended plan"). A tolerant parser is itself a silent-zero vector | `spec_resolver.parse_spec_json` requires plan-shape (`_PLAN_KEYS`) on every candidate and raises on fragments; proposer token cap raised to 16384 | `test_spec_resolver.py` |
| 19 | Tuple-typed params (`ngram_range`, `quantile_range`, `hidden_layer_sizes`) always lost by forfeit | JSON has no tuple type; sklearn validates these as *tuple* and raises `InvalidParameterError` on the list | `spec_resolver._json_options` (list→tuple) in both `_build_choice` and `_build_free_choice` | `test_spec_resolver.py` |
| 20 | One malformed proposed HP (log scale + 0.0 low) silently dropped a whole Option-3 injection | `skrub.choose_*` raises on log scale at low ≤ 0, and the resolve failure skipped the entire merged plan | `_build_choice` falls back to a linear scale; per-param guard in `_make`; an injection that still won't resolve is recorded in `result["proposal_injection_error"]` | `test_spec_resolver.py` / `test_search_loop.py` |
| 21 | School (GWDG) reasoning models returned empty content → fallback spec | reasoning models burn output tokens thinking; a low `max_tokens` ends them mid-thought | LiteLlm path sets `max_tokens=16384` (`LITELLM_MAX_TOKENS`); search-instruction prompt fragments stripped when `google_search` is off | live-validated (qwen3.5-397b) |
| 22 | Rollouts silently 0.0 at CV `n_jobs>1` when `skrub_ops` was loaded via `importlib.util.spec_from_file_location` under its bare name (legacy test boilerplate) | joblib's spawned workers must unpickle plan steps (`_SanitizeColumns` etc.) by re-importing module `"skrub_ops"`, which isn't importable in a fresh process → `BrokenProcessPool` → caught → 0.0 (and whether it failed depended on cloudpickle's by-value fallback, so it was plan-dependent) | tests import `machine_learning_engineering.skrub_ops` normally (the boilerplate predated the side-effect-free package `__init__`); never register logic modules under bare names | suite green at `make_rollout_fn`'s `n_jobs=6` default |

## Open / flagged (not fixed)

- **GapEncoder loses by forfeit on high-cardinality text tasks.** On
  midwest-survey one GapEncoder rollout takes ~45s+ and hits the wall-clock cap
  → 0.0, while StringEncoder squeaks under. The cap that bounds pathological
  free-form HPs also forfeits a legitimate encoder; wide high-card tasks are
  intractable at budget 40 (traffic-violations projected ~15h). Options if
  wide-task eval is needed: shrink the subsample cap for high-card tasks, or
  give expensive encoders their own per-rollout budget.
- **`llm_calls` miscounts offline replay proposers as real calls** (bookkeeping
  only).

## Practices these bugs taught

- **Live-smoke every search-affecting feature** — the offline suite was green
  through nearly all of the above; only real runs on real (imbalanced,
  relational, string-labeled) tasks surfaced them.
- **Vet any native-lib operator by process exit code** on a real rollout
  through `build_staged_plan` + `make_rollout_fn` before trusting it (a
  segfault is not an exception).
- **Timeouts must be `BaseException`** — sklearn's `cross_validate` catches
  `Exception` per fold, so an `Exception`-based timeout would NaN one fold
  instead of aborting the rollout (`skrub_ops._RolloutTimeout`).
