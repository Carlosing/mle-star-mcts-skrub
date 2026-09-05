# CONTRIBUTIONS.md

This file records who wrote which part of the project.

> **Note on attribution.** The team worked collaboratively and did not track authorship at the
> commit or line level during development. Rather than reconstruct that after the fact, we
> attribute by **functional area** — each area is a coherent, self-contained slice of the
> system with a clear owner. The file/area pointers below are accurate; the per-line split
> within a shared file is approximate.
>
> All members contributed to design discussions, code review, debugging (see
> `docs/BUG_LEDGER.md`), the offline test suite, and the written reports.

The system is divided into four areas, matching its layer contract
(`AGENT LAYER → spec_resolver → skrub_ops → search_loop → mcts`, wired by `pipeline.py`):

| Area | Owner | Core files |
|---|---|---|
| Agent development | **Carlos Alberto Escobedo Lopez** | `adk_agent.py`, `data_summary.py`, `prompt.py`, + revived MLE-STAR baseline |
| MCTS search | **Bolkar Eren** | `mcts.py`, `search_loop.py` |
| skrub conversion | **Murat Mert Ali Sekerci** | `skrub_ops.py`, `spec_resolver.py` |
| Pipeline & ensemble | **Setenay Konukseven** | `pipeline.py`, `ensemble.py`, `metrics.py`, + benchmark/eval scripts |

---

## Area 1 — Agent development — **Carlos Alberto Escobedo Lopez**

The LLM-facing layer: the two-agent ADK graph that reads the data and authors the rich JSON
plan, the EDA digest it sees, the provider switch, and the revived MLE-STAR baseline used for
comparison.

**Files / pointers**

- `machine_learning_engineering/adk_agent.py` — the ADK graph `data_analyst → plan_author`;
  `build_root_agent`; `_resolve_model` (env-switched native Gemini vs OpenAI-compatible via
  `LiteLlm`; `google_search` attached on the Gemini path only).
- `machine_learning_engineering/data_summary.py` — `make_data_summary`, the compact EDA digest
  that is the *only* view of the data the LLM gets.
- `machine_learning_engineering/prompt.py` — the agent prompts (analyst + plan-author schema,
  including the relational `AggJoiner` and `scoped_encodings` contracts).
- `machine_learning_engineering/run_logging.py` — per-call prompt/response + token logging.
- `machine_learning_engineering/__init__.py` — the `PROVIDER` selection glue.
- `probe_gemini.py`, `probe_school.py` — the model/quota probes.
- **Revived MLE-STAR baseline** (off the MCTS path, used only as the comparison arm):
  `machine_learning_engineering/agent.py`, `sub_agents/` (`initialization`, `refinement`,
  `ensemble`, `submission`), `shared_libraries/` (incl. `web_search_util.py`,
  `check_leakage_util.py`, `code_util.py`, `debug_util.py`), `runner.py`, `eval/`.
- Baseline drivers: `scripts/run_mlestar.py`, `scripts/convert_mlestar_final_state.py`.

**Tests:** `tests/test_agent_orchestration.py`, `tests/test_data_summary.py`,
`tests/test_web_search_util.py`, `tests/test_initialization_agent.py`,
`tests/test_mlestar_caps.py`, `tests/test_check_leakage_util.py`, `tests/test_code_util.py`.

---

## Area 2 — MCTS search — **Bolkar Eren**

The pure-code search engine and the outer loop that drives it. No skrub, no LLM client, no I/O
— fully offline-testable.

**Files / pointers**

- `machine_learning_engineering/mcts.py` — the MCTS engine: UCT select/expand/backprop, the
  **persistent tree**, the **score cache**, model-gated expansion (`gating`/`target_key`),
  `canonicalize`, the `prior_fn` hook, and `start_node` incumbent-local descent.
- `machine_learning_engineering/search_loop.py` — the outer loop: persisted-tree MCTS as
  fixed-budget slices; `tree_action_values` (tree-mined ablation) and `pick_target_node`
  (**Optional Feature 1** focus-stage hint); `make_llm_proposer` + `_merge_raw_plans` (**Optional Feature 3**
  whole-plan injection); the post-budget **focused-refinement bonus phase** (`refined_dims`).

**Tests:** `tests/test_mcts.py`, `tests/test_search_loop.py`, `tests/test_priors_and_proposer.py`.

---

## Area 3 — skrub conversion — **Murat Mert Ali Sekerci**

The bridge from the LLM's JSON plan to an executable, seeded skrub DataOps pipeline, and the
search space derived from it. This is where *all* skrub access is isolated.

**Files / pointers**

- `machine_learning_engineering/skrub_ops.py` — `build_staged_plan` (assemble + scoped
  encodings + always-on `Cleaner`/`TableVectorizer` backbones + the `_SanitizeColumns` rename),
  `get_action_space`, `get_choice_gating`, `apply_state`, the seeded rollouts (profile-aware
  subsampling, the 60 s per-rollout wall-clock cap, parallel CV folds, stratified CV, the
  proba-scorer shim), `run_ablation`, `pick_target_node`.
- `machine_learning_engineering/spec_resolver.py` — LLM JSON → seeded estimators + `choose_*`
  nodes; the **import allow-list** safety envelope; free-form vs `REGISTRY`-clipped HP ranges;
  the per-param safety nets (`_accepts_param`, `_RNG_PARAMS`); the relational `assemble`
  passthrough.

**Tests:** `tests/test_skrub_ops.py`, `tests/test_spec_resolver.py`, `tests/test_staged.py`,
`tests/test_scope_stage.py`, `tests/test_relational_pipeline.py`,
`tests/test_sanitize_columns.py`, `tests/test_rollout_timeout.py`,
`tests/test_profile_subsample.py`, `tests/test_arbitrary_hp.py`,
`tests/test_silent_zero_regressions.py`.

---

## Area 4 — Pipeline & ensemble — **Setenay Konukseven**

The end-to-end driver that wires all layers together, the Caruana ensemble read-off, the
scorers, and the whole benchmark/evaluation harness (staging, baselines, figures).

**Files / pointers**

- `machine_learning_engineering/pipeline.py` — the end-to-end driver + CLI; `load_task` (reads
  **only** `train.csv`; discovers `aux_*.csv`), `load_holdout` (the shared on-disk bench),
  `save_run_artifacts` (`result.json` + `summary.md` + `ensemble.pkl`).
- `machine_learning_engineering/ensemble.py` — the **Caruana** greedy weighted read-off over
  the score cache; `holdout_split`; OOF selection (`evaluate_top_k`, `selection` stamp);
  `EnsemblePredictor` (the picklable fitted result).
- `machine_learning_engineering/metrics.py` — the two scorers (bounded search reward vs the
  competition report metric).
- **Benchmark / evaluation harness:** `scripts/stage_tasks.py`, `scripts/stage_credit_fraud.py`,
  `scripts/run_autogluon.py`, `scripts/run_benchmark.py`, `scripts/collect_results.py`,
  `scripts/make_figures.py`, `scripts/replay_from_run.py`, `scripts/claude_agents.py`,
  `scripts/run_claude_pipeline.py`, `scripts/recover_answers.py`.

**Tests:** `tests/test_pipeline.py`, `tests/test_ensemble.py`, `tests/test_shared_holdout.py`,
`tests/test_staged_tasks.py`, `tests/test_integration.py`,
`tests/test_initialization_pipeline.py`.

---

## Shared / cross-cutting

Some artifacts were genuinely joint work and are not owned by a single area:

- **The offline test suite** (`tests/`, 427 tests) — each area owns its own tests (listed
  above), but the fixtures, `conftest.py`, and the `FakeLlm` agent mocks were built jointly.
- **Documentation** — `README.md`, `CLAUDE.md`, `docs/` (PROJECT_STATE, USAGE, BUG_LEDGER,
  architecture notes) and the three submission docs (`EXPERIMENTS.md`,
  `EXPERIMENTAL_RESULTS.md`, this file) were written collaboratively.
- **The build/config** — `Makefile`, `pyproject.toml`, `uv.lock`, `.env.example`, `Dockerfile`.
