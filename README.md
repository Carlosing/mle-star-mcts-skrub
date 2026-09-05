# MLE-STAR × skrub × MCTS

> An adaptation of **MLE-STAR** into an **MCTS search over [skrub](https://skrub-data.org)
> DataOps pipelines**. LLM agents read a dataset and author a *rich JSON plan* — a
> per-stage menu of operators plus hyperparameter ranges. A pure-code MCTS engine then
> searches that space (structure **and** hyperparameters) over a fixed evaluation budget.
> **The LLM proposes the space; it never runs the search.**

A fork of Google's [MLE-STAR](https://github.com/google/adk-samples). The original
code-writing agent is retained as a benchmark baseline; the project is the MCTS
extension documented here.

Reproduction and per-run provenance: [`EXPERIMENTS.md`](EXPERIMENTS.md). Results and
their validity: [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md). Commands and
options: [`docs/USAGE.md`](docs/USAGE.md).

---

## Architecture

```
  AGENT LAYER (LLM via ADK)                 LOGIC LAYER (pure Python, no LLM client)
  data_analyst ─► plan_author ──JSON──►     spec_resolver ─► skrub_ops ─► search_loop ─► mcts
  (reads a data   (rich plan:               (names + HP     (build plan,  (persist tree, (UCT
   digest, web     operators +               ranges ->       action space,  ablation,      search)
   search)         HP ranges)                seeded          rollouts)      Optional Feature 1/3)
                                             estimators)
        └──────────────── pipeline.py (driver) wires the whole thing ──────────────┘
```

**The load-bearing invariant:** *code owns structure and math; the LLM owns knowledge
and language.* The LLM is called **O(1) times per task** — two agent calls, plus at
most one mid-search call to extend the plan — and **never** inside the inner search
loop. The only cost that scales with search size is CV rollouts, which are pure
Python. Quality is bought with compute at a token cost fixed in advance,
rather than with an unbounded cascade of code-and-debug calls.

---

## How it works (`pipeline.run_pipeline`)

0. **The split already exists on disk.** Each task ships as `train.csv` /
   `test.csv` / `test_answer.csv`, split before any method runs (80/20 seeded,
   stratified for classification). `load_task` reads **only** `train.csv`; the
   holdout goes to the scorers alone, so the search cannot see it.
1. **`load_task`** → dataframe + target + task type + metric, plus any relational
   `aux_*.csv` tables.
2. **`make_data_summary`** → a compact EDA digest, the only view of the data the LLM
   gets.
3. **Agents** (`data_analyst` → `plan_author`) → a rich JSON plan: a per-stage menu
   of operators (by dotted import path) with hyperparameter ranges.
4. **`resolve_spec`** → JSON to seeded estimators and `choose_*` nodes. Operators
   pass an **import allow-list** (`sklearn`, `skrub`, `lightgbm`, `xgboost`); nothing
   is `eval`'d, hallucinated paths are dropped.
5. **`build_staged_plan` / `get_action_space` / `get_choice_gating`** → assembles the
   skrub pipeline, derives the search space, exposes the model→HP conditional gate.
6. **`run_search_loop`** → **MCTS** over `budget` CV rollouts. The tree and the cache
   of already-scored configurations are carried across the whole run, and
   hyperparameters only enter the search when their model is selected. The run is
   divided into *slices* — stretches of search between two proposal calls — and three
   **Optional Features** sit on top:
   - **Optional Feature 1 — Ablation.** The search tree doubles as a free
     ablation: every node differs from its parent by one edit, so grouping nodes by
     the stage they edited gives a per-stage spread of rewards. Between slices, the
     stage whose choices moved the score most is passed to the proposer as a hint
     about where the plan is worth extending. It is only a hint — expansion is never
     restricted to that stage, and no extra rollouts are spent producing it.
   - **Optional Feature 2 — Prior Targeting.** Plan options may carry a
     `"prior": 0.0–1.0`; `prior_fn` seeds fresh children's Q/N with it, at zero extra
     LLM calls.
   - **Optional Feature 3 — Plan Injection.** One LLM call between slices returns a
     whole extended plan, merged additively and rebuilt.
   - Plus a **focused-refinement bonus phase**: `ceil(total/4)` extra rollouts over
     all single-edit neighbours of the incumbent, no LLM.
7. **Ensemble** (`--top-k > 1`) → a **Caruana** greedy weighted read-off over the
   configurations already scored during the search, so it costs no new evaluations.
   Members are selected on 3-fold out-of-fold predictions over all of train, then
   refit on all of train and scored on the untouched holdout; selection rows and
   reported rows are never the same rows.
8. **Report** → the incumbent on the competition metric (`report`) and on the on-disk
   holdout (`holdout`, the cross-method-comparable number). Writes
   `runs/<task>_<ts>/{result.json, summary.md, ensemble.pkl}`.

---

## Modules

All on the MCTS path, under `machine_learning_engineering/`.

| Module | Role |
|---|---|
| `mcts.py` | The MCTS engine: UCT select/expand/backprop, persistent tree, score cache, model-gated expansion, `canonicalize`, `prior_fn`, incumbent-local descent. No skrub, no LLM, no I/O. |
| `skrub_ops.py` | All skrub access. `build_staged_plan`, `get_action_space`, `get_choice_gating`, `apply_state`, seeded rollouts (profile-aware subsampling, 60 s per-rollout cap, parallel CV folds), `run_ablation`. |
| `spec_resolver.py` | LLM JSON → seeded estimators + `choose_*` nodes. The import allow-list is the safety envelope; HP ranges are free-form unless curated in `REGISTRY`. |
| `search_loop.py` | Outer loop: fixed-budget MCTS slices, tree-mined ablation, whole-plan injection (`make_llm_proposer`), focused-refinement bonus phase. |
| `ensemble.py` | Caruana read-off over the score cache. `holdout_split`, OOF selection, `EnsemblePredictor` → `ensemble.pkl`. |
| `data_summary.py` | `make_data_summary` — the EDA digest the LLM sees. |
| `adk_agent.py` | The ADK graph `data_analyst → plan_author`; provider switched by `ROOT_AGENT_MODEL`. |
| `metrics.py` | The two scorers — bounded search reward vs competition report metric. |
| `pipeline.py` | End-to-end driver + CLI. |

Benchmark drivers live in `scripts/`; the revived MLE-STAR baseline is `agent.py` +
`sub_agents/` + `shared_libraries/` + `runner.py` + `eval/`, deliberately off the
MCTS path.

---

## Design decisions

- **Structured JSON plan, not code-gen.** No `eval` of model output. The plan is
  schema-validated, centrally seeded, and made of named choices the search can reason
  about.
- **Import-level allow-list** is the safety envelope — a hallucinated operator is
  dropped, never executed.
- **Two separate scorers** — a bounded, higher-is-better reward drives the search;
  the competition metric is used only for the final report.
- **The train/test split is drawn on disk before any method runs.** The holdout rows
  are in no file the search can read, so the boundary holds by construction. The
  ensemble follows the same rule one level down: it selects members on out-of-fold
  predictions and reports on the holdout.
- **One search tree, carried across the whole run.** The reward function is fixed, so
  a configuration's score stays valid for the entire run and the tree accumulates into
  a reusable ablation.
- **LLM calls are the cost model** — two per task, plus one per proposal step, fixed
  before the run starts.
- **Relational tables are a first-class, searchable stage** — `AggJoiner` over
  `aux_*.csv`, a structural capability the flat-table baselines lack.
- **One ADK stack, env-switched provider** — `ROOT_AGENT_MODEL` selects native Gemini
  or any OpenAI-compatible endpoint via `LiteLlm`. The logic layer imports no LLM
  client.

---

## Quick start

```bash
make sync          # build the venv (needs internet, run once)
```

```bash
make test          # offline suite — mocked agents, real skrub CV (~5 min, 427 tests)
```

```bash
make run-live TASK=california-housing-prices BUDGET=40 PROVIDER=school
```

Each run writes `runs/<task>_<ts>/` with `result.json`, `summary.md` and
`ensemble.pkl`.

**Provider.** Live runs need `PROVIDER=school` (GWDG, OpenAI-compatible —
`openai/qwen3.5-397b-a17b`); it is the only endpoint with the quota for a full
full benchmark run, and `make benchmark-extension` / `make benchmark-mlestar` take the
same flag. The `google` (Gemini) provider was used during development, which is why
the switch exists; because `google_search` is Gemini-native, the analyst's web search
is off on `school`.

---

## Repository layout

```
machine_learning_engineering/   # the MCTS extension + revived MLE-STAR baseline
  └── tasks/                    # the 13 staged tasks — train/test/test_answer + aux
scripts/                        # benchmark drivers, replay, figure pipeline
tests/                          # offline suite (make test) — agents mocked with FakeLlm
results/                        # git-shareable mirror of run artifacts (whole run dirs)
figures/                        # rendered comparison figures + comparison.csv
docs/                           # USAGE, PROJECT_STATE, BUG_LEDGER, architecture notes
CLAUDE.md                       # the invariant-level design contract
CONTRIBUTIONS.md                # per-area authorship
```

The staged tasks are committed, so nothing needs downloading or staging after
cloning.
