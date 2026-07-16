# Bug ledger — historical bugs, where they lived, and the fix that shipped

A reference of the real bugs found during development (most surfaced on *live*
runs, not the offline suite), kept here so the code doesn't need bug-history
comments. Each entry: symptom → root cause → fix location → regression test.

The entries fall into **two shapes, and neither one crashes**:

1. **The silent 0.0** (#1–#25). A failure inside skrub's CV is swallowed by the
   rollout `try/except` (or NaN-ed by the scorer) → the config silently scores
   0.0 → search returns garbage without crashing. When adding an operator family
   or scorer, probe that boundary first (build a tiny plan, call the rollout,
   assert reward > 0), then pin it in
   [test_silent_zero_regressions.py](../tests/test_silent_zero_regressions.py).
2. **The flattering number** (#26–#29). Selection and reporting quietly share
   rows, so the published score is a maximum over the metric rather than an
   out-of-sample estimate. There is no failing artifact at all — the benchmark
   just looks good. These were found by tracing the data flow, not by a test.

Both classes are invisible to a green suite. That is the point of this file.

## Rollout / scoring boundary (silent 0.0s)

| # | Bug | Root cause | Fix | Test |
|---|---|---|---|---|
| 1 | Every proba-only classifier (RF/LGBM/XGB) scored 0.0 on roc_auc tasks — incl. flagship credit-fraud | skrub's DataOps learner reports `_estimator_type == "transformer"`, so sklearn's builtin scorer never reduces a binary `predict_proba` (n,2) to the positive column; `roc_auc_score` raises per fold, skrub NaNs it | `skrub_ops._resolve_scoring` swaps proba-ranking metric *names* for a positive-column callable | `test_skrub_ops.py::test_roc_auc_scores_a_proba_only_classifier` |
| 2 | decision_function-only classifiers (LinearSVC, SGD-hinge) zeroed on roc_auc | the #1 shim *forced* `predict_proba` | the callable falls back to `decision_function` | `test_silent_zero_regressions.py` |
| 3 | Classification folds silently NaN on rare targets | skrub's `cross_validate` calls `check_cv(cv)` without `y`/`classifier` → always plain `KFold`; a subsampled fold can hold zero positives | `skrub_ops._cv_kwarg` passes an explicit seeded `StratifiedKFold`, auto-on for classification | `test_relational_pipeline.py::test_stratified_rollout_avoids_nan_on_rare_target` |
| 4 | Every rollout in a run scored 0.0 once plans HP-tuned non-model operators | `describe_defaults()` abbreviates an HP-tuned list entry to `"ClassName(...)"`, which never matches `get_action_space`'s full-repr label → unappliable root state | `skrub_ops.get_default_state` reads discrete defaults from `get_action_space(plan)[name][0]` (skrub's `Choice.default` is always `outcomes[0]`), never from the describe string | `test_scope_stage.py::test_default_state_is_appliable_when_slot_options_have_tuned_hps` |
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
| 18 | Extended Feature 3 proposer's tuned re-proposal of an existing operator was silently dropped (a live toxicity run's `skrub.TextEncoder` gaining an `n_components` range vanished on merge, costing tokens for zero effect) | `_merge_option_lists` deduped by operator PATH only, so a same-path entry carrying new params matched the bare one and was skipped | dedup now by full signature (path + tuned params) on **repr-labeled** stages (clean/encoder/scoped), so bare and tuned coexist as siblings; a numeric-range re-tune of an already-tuned param is stripped to avoid a duplicate `<stage>__<Class>__<param>` choose-node; the class-name-labeled model list keeps path-only dedup (`repr_labeled=False`) since same-class models collapse at resolve | `test_search_loop.py::test_merge_raw_plans_adds_tuned_reproposal_as_sibling` + `_blocks_numeric_retune_collision` + `_drops_same_class_model_reproposal` |

## LLM output parsing / resolution

| # | Bug | Root cause | Fix | Test |
|---|---|---|---|---|
| 19 | Extended Feature 3 silently no-op'd (`injected_options: []`, no error) | proposer responses truncated at `max_output_tokens` mid-object; the leftover text still holds valid JSON *fragments* and a brace-scan returned one (`{"float": [0.7, 1.0]}` became "the extended plan"). A tolerant parser is itself a silent-zero vector | `spec_resolver.parse_spec_json` requires plan-shape (`_PLAN_KEYS`) on every candidate and raises on fragments; proposer token cap raised to 16384 | `test_spec_resolver.py` |
| 20 | Tuple-typed params (`ngram_range`, `quantile_range`, `hidden_layer_sizes`) always lost by forfeit | JSON has no tuple type; sklearn validates these as *tuple* and raises `InvalidParameterError` on the list | `spec_resolver._json_options` (list→tuple) in both `_build_choice` and `_build_free_choice` | `test_spec_resolver.py` |
| 21 | One malformed proposed HP (log scale + 0.0 low) silently dropped a whole Extended Feature 3 injection | `skrub.choose_*` raises on log scale at low ≤ 0, and the resolve failure skipped the entire merged plan | `_build_choice` falls back to a linear scale; per-param guard in `_make`; an injection that still won't resolve is recorded in `result["proposal_injection_error"]` | `test_spec_resolver.py` / `test_search_loop.py` |
| 22 | School (GWDG) reasoning models returned empty content → fallback spec | reasoning models burn output tokens thinking; a low `max_tokens` ends them mid-thought | LiteLlm path sets `max_tokens=16384` (`LITELLM_MAX_TOKENS`); search-instruction prompt fragments stripped when `google_search` is off | live-validated (qwen3.5-397b) |
| 23 | Rollouts silently 0.0 at CV `n_jobs>1` when `skrub_ops` was loaded via `importlib.util.spec_from_file_location` under its bare name (legacy test boilerplate) | joblib's spawned workers must unpickle plan steps (`_SanitizeColumns` etc.) by re-importing module `"skrub_ops"`, which isn't importable in a fresh process → `BrokenProcessPool` → caught → 0.0 (and whether it failed depended on cloudpickle's by-value fallback, so it was plan-dependent) | tests import `machine_learning_engineering.skrub_ops` normally (the boilerplate predated the side-effect-free package `__init__`); never register logic modules under bare names | suite green at `make_rollout_fn`'s `n_jobs=6` default |
| 24 | A live toxicity run crashed at plan-BUILD: `ValueError: max_df corresponds to < documents than min_df` (`build_staged_plan` → `node.skb.apply(vectorizer)`) — not a silent 0.0, a hard run-killer outside the rollout net | the plan_author put a raw `sklearn...TfidfVectorizer` with `min_df:{int:[1,5]}` in the encoder slot. Two compounding faults: (a) skrub eagerly PREVIEWS every `.skb.apply` on a **single row**, and `min_df≥2` is impossible on one doc (skrub previews a `choose_int` at its midpoint → 3); (b) even past that, sklearn's text vectorizers return a **scipy-sparse** matrix skrub's pandas DataOps can't carry — and, unlike OneHotEncoder, they have no dense flag. The existing per-param nets (`_accepts_param`/`_RNG_PARAMS`/`_SPARSE_PARAMS`) gate on *legality*, not value-survivability, so a legal `min_df` waved through | `_DOC_FREQ_PARAMS` drops `min_df`/`max_df` (like `_SPARSE_PARAMS`); `_emits_dataframe` screens each **sklearn-rooted** transformer (bare-constructor probe on a text sample — output container type is a class property, so no tuned-param/`choose_*` quirks; skrub encoders trusted, so no `TextEncoder` model download) and `_make` drops sparse emitters; `format_allowed_for_prompt` steers the LLM to skrub.StringEncoder/TextEncoder/GapEncoder/MinHashEncoder for text | `test_spec_resolver.py::test_sklearn_text_vectorizers_are_dropped_as_sparse` |
| 25 | The relational **assemble** stage silently vanished (`dropped_sections: ["assemble"]`) on country-happiness — the join differentiator never entered the search on the one task where it *is* the task | it is a 1-to-1 lookup (one GDP/life-exp row per country), so the planner correctly emitted `operations: []` — but `_resolve_assemble` required ≥1 aggregation op and dropped every join. All three agents (analyst/planner/proposer) asked for joins; the resolver killed them each time | empty operations default to `["mean"]` (identity on a single matched row, still valid for 1-to-many); GIVEN-but-all-invalid ops still drop (hallucination). Multi-table already composes via `_ChainedAggJoiner`'s `all_aggregates` | `test_relational_pipeline.py::test_assemble_empty_operations_default_to_mean_for_one_to_one` |

## Evaluation integrity (a flattering number, not a crash)

The 2026-07-14 class. None of these threw, none showed up in the suite, and none
would ever have shown up in a *score* — a biased benchmark just looks like a good
result.

All four are **fixed in code**, but the fix landed too late to re-run the
benchmark: the shipped results were produced *before* it and are optimistically
biased in our own favour. That bias is disclosed rather than removed — see
[PROJECT_STATE § Hand-off](PROJECT_STATE.md#hand-off--the-published-numbers-are-biased)
for exactly what it does and does not compromise, and how to report it.

| # | Bug | Root cause | Fix | Test |
|---|---|---|---|---|
| 26 | The extension's `holdout` score — the *cross-method comparable* number — was optimistically biased. Nothing crashed; the figure was simply flattering | `pipeline.run_pipeline` passed the FULL `train.csv` frame to `run_search_loop`, and `ensemble.evaluate_top_k` then carved its 25% holdout out of that *same* frame. So every MCTS rollout cross-validated over rows that later became the extension's own eval set: the config was selected with knowledge of the rows we report on. AutoGluon, by contrast, fit only the 75% and never saw the holdout — so every extension-vs-AutoGluon delta was overstated | the split is now drawn **on disk before any method sees the data** (`scripts/stage_tasks.py` writes `train.csv` / `test.csv` / `test_answer.csv`). `load_task` reads only `train.csv`, so the holdout rows are physically absent from the search's frame; `pipeline.load_holdout` supplies the shared bench to the scorers | `test_shared_holdout.py::test_holdout_rows_are_absent_from_train` |
| 27 | The Caruana ensemble reported a score it had been fitted to — the exact ensemble-overfitting flaw the MLE-STAR benchmark report criticises MLE-STAR for (§8.3) | `_caruana_select` greedily maximised the score **on the holdout**, and `evaluate_top_k` then returned that maximised value as `ensemble_score`. Selection and reporting were the same rows, so the number was a greedy maximum over the published metric, not an out-of-sample score | `evaluate_top_k(holdout=)` splits the two: members are **selected** on 3-fold out-of-fold predictions spanning all of train (`_oof_pool`, one shared fold assignment so the OOF columns are combinable), then the pool is refitted on all of train and **scored** on the untouched holdout. `ensemble_score` is consequently no longer guaranteed to beat the best pool member — that guarantee was the bias. `selection` is stamped on every artifact; the old path survives behind `--legacy-ensemble` so pre-fix runs stay reproducible | `test_ensemble.py::test_caruana_selects_on_oof_predictions_not_on_the_reported_holdout`, `::test_legacy_selection_flag_restores_the_biased_path` |
| 28 | `run_mlestar._score_submission` would have published a meaningless number for MLE-STAR, silently — the benchmark's headline comparison | MLE-STAR predicts the task's `test.csv`; the scorer aligned those predictions **by row order** against a holdout carved out of `train.csv`, guarding on nothing but `len(sub) == len(holdout)`. Those lengths collide **exactly on 8 of the 13 tasks** (test.csv is 20% of source, the old holdout was 25% of the 80% train — the same count), so the guard passed and predictions for one set of rows were scored against targets from a completely different set. Never fired only because MLE-STAR had not yet been run end-to-end | `test.csv` IS the shared bench now, so MLE-STAR's submission is the right rows by construction; the scorer reads `test_answer.csv` and returns `{"error": ...}` on any mismatch instead of a silent `None`-or-plausible-number | `test_shared_holdout.py::test_mlestar_scorer_refuses_a_mismatched_submission` |
| 29 | Every relational task's holdout rows joined to **nothing** — all aux-derived features NaN at predict time, on the very tasks the skrub extension exists to win | `scripts/stage_tasks.py` filtered each aux table to the keys of `train` (`aux[aux[aux_key].isin(train[main_key])]`), so `test.csv`'s keys were absent: country-happiness holdout coverage was **0/29**. Latent while each method carved its holdout out of `train.csv` (whose keys *were* covered); it would have surfaced the moment `test.csv` became the shared bench | filter by the keys of the **whole** table (`df`), not `train`. Not leakage: aux tables carry features, never the target — and a holdout row you cannot join is a holdout row you cannot predict. Coverage is now 79–100% (country-happiness's 79% is countries genuinely missing from the World-Bank source, matching the train-side rate) | `test_shared_holdout.py` (aux coverage asserted via the relational task fixtures) |

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
- **A bad *number* is worse than a bad *crash*, and the eval path is where they
  hide** (#26–#29). Every one of those was a silent overstatement in our own
  favour, found by reading the data flow rather than by any test failing. When
  you touch scoring, splitting or ensembling, ask the one question the tests
  can't: *are the rows I am selecting on the rows I am reporting on?*
