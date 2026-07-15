# MLE-STAR × skrub × MCTS

> An adaptation of **MLE-STAR** into an **MCTS search over [skrub](https://skrub-data.org)
> DataOps pipelines**. LLM agents read a dataset and author a *rich JSON plan* — a
> per-stage menu of operators plus hyperparameter ranges. A pure-code MCTS engine then
> searches that space (structure **and** hyperparameters) over a fixed evaluation budget.
> **The LLM proposes the space; it never runs the search.**

This repository is a fork of Google's [MLE-STAR](https://github.com/google/adk-samples)
agent. The original agent (which writes and iteratively debugs Python code) is still
present, but *revived as a benchmark baseline* — the project itself is the MCTS extension
described below. The upstream agent's design is off the critical path; this README
documents **our** system.

- **New here?** Read this file, then [`docs/USAGE.md`](docs/USAGE.md) to run it.
- **Grading / structure?** See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md),
  [`EXPERIMENTS.md`](EXPERIMENTS.md), [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md).
- **Design deep-dive / status:** [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) is the
  canonical roadmap; [`CLAUDE.md`](CLAUDE.md) is the invariant-level design contract.

---

## The idea in one picture

```
  AGENT LAYER (LLM via ADK)                 LOGIC LAYER (pure Python, no LLM client)
  data_analyst ─► plan_author ──JSON──►     spec_resolver ─► skrub_ops ─► search_loop ─► mcts
  (reads a data   (rich plan:               (names + HP     (build plan,  (persist tree, (UCT
   digest, web     operators +               ranges ->       action space,  ablation,      search)
   search)         HP ranges)                seeded          rollouts)      Extended Feature 1/3)
                                             estimators)
        └──────────────── pipeline.py (driver) wires the whole thing ──────────────┘
```

**The load-bearing invariant:** *code owns structure and math; the LLM owns knowledge and
language.* The LLM is called **O(1) times per task** (two agent calls: `data_analyst` +
`plan_author`, plus at most one call between search slices for mid-search injection). It is
**never** called inside the inner search loop. The only cost that scales with search size is
CV rollouts — pure Python.

This is the whole point of the design: **you buy quality with compute/time at a fixed,
known token cost**, instead of with an unbounded cascade of code-and-debug LLM calls.

---

## How it works, end to end (`pipeline.run_pipeline`)

0. **The train/holdout split already exists on disk.** `scripts/stage_tasks.py` wrote
   `train.csv` / `test.csv` / `test_answer.csv` per task *before any method runs* (80/20
   seeded, stratified for classification). `load_task` reads **only** `train.csv`; the
   holdout is handed to the *scorers* alone. The search physically cannot see the holdout.
1. **`load_task`** → dataframe + target + task type + metric (from `task_description.txt`),
   plus any relational `aux_*.csv` tables.
2. **`make_data_summary`** → a compact EDA digest — the *only* view of the data the LLM gets.
3. **Agents** (`data_analyst` → `plan_author`) → a rich JSON plan: a per-stage menu of
   operators (by dotted import path) with hyperparameter ranges. `data_analyst` may use web
   search (Gemini only).
4. **`resolve_spec`** → turns the JSON (names + HP ranges) into seeded estimators and
   `choose_*` nodes. Operators pass through an **import allow-list** (roots `sklearn`,
   `skrub`, `lightgbm`, `xgboost`); nothing is `eval`'d, hallucinated paths are dropped.
5. **`build_staged_plan` / `get_action_space` / `get_choice_gating`** → assembles the skrub
   DataOps pipeline, derives the MCTS search space, and exposes the model→HP conditional gate.
6. **`run_search_loop`** → persisted-tree **MCTS** over `budget` CV rollouts (with a score
   cache and model-gated HPs). Three optional **Extended Features** amplify the base search
   (older commit messages call 1 and 3 "Option 1/3"):
   - **Extended Feature 1 — ablation targeting.** Between search slices, a tree-mined ablation
     picks a focus stage (a *proposer hint*, never an expansion lock).
   - **Extended Feature 2 — prior warm-start.** Plan options may carry a `"prior": 0.0–1.0`
     rating; `prior_fn` seeds fresh children's Q/N with it (the AlphaZero "policy prior + UCT"
     pattern, at zero extra LLM calls).
   - **Extended Feature 3 — mid-search plan injection.** With a proposer, one LLM call between
     slices returns a *whole extended plan* (merged strictly additively, re-resolved, rebuilt —
     injected operators arrive already tuned).
   - Plus a **focused-refinement bonus phase** — after the budget, `ceil(total/4)` extra
     rollouts explore *all* single-edit neighbours of the incumbent (structure and HPs), no LLM.
7. **Ensemble** (`--top-k > 1`) → a **Caruana** greedy weighted read-off over the persisted
   score cache (no new search). Members are **selected** on 3-fold out-of-fold predictions
   over all of train, then **refit on all of train and scored on the untouched holdout** —
   selection rows and reported rows are never the same rows.
8. **Report** → score the incumbent on the competition metric (via CV, `report`) *and* on
   the shared on-disk holdout (`holdout` — the cross-method-comparable number). Write
   `runs/<task>_<ts>/{result.json, summary.md, ensemble.pkl}`.

---

## Main code components

Everything on the MCTS path lives in `machine_learning_engineering/`.

| Module | Role |
|---|---|
| [`mcts.py`](machine_learning_engineering/mcts.py) | The MCTS engine: UCT select/expand/backprop, a **persistent tree**, a **score cache**, model-gated expansion, `canonicalize`, `prior_fn` hook, incumbent-local descent. **No skrub, no LLM, no I/O.** |
| [`skrub_ops.py`](machine_learning_engineering/skrub_ops.py) | **All skrub access.** `build_staged_plan` (assemble + scoped encodings + always-on `Cleaner`/`TableVectorizer` backbones), `get_action_space`, `get_choice_gating`, `apply_state`, seeded rollouts (profile-aware subsampling, per-rollout 60 s wall-clock cap, parallel CV folds), `run_ablation`. |
| [`spec_resolver.py`](machine_learning_engineering/spec_resolver.py) | LLM JSON → seeded estimators + `choose_*` nodes. The **import allow-list** is the safety envelope; HP ranges are free-form unless curated in `REGISTRY` (then clipped). |
| [`search_loop.py`](machine_learning_engineering/search_loop.py) | The outer loop: fixed-budget MCTS slices, tree-mined ablation (Extended Feature 1), whole-plan injection (Extended Feature 3, `make_llm_proposer`), and the focused-refinement bonus phase. |
| [`ensemble.py`](machine_learning_engineering/ensemble.py) | **Caruana ensemble read-off** over the score cache. `holdout_split`, OOF selection, `EnsemblePredictor` (the picklable fitted result → `ensemble.pkl`). |
| [`data_summary.py`](machine_learning_engineering/data_summary.py) | `make_data_summary` — the EDA digest the LLM sees. |
| [`adk_agent.py`](machine_learning_engineering/adk_agent.py) | The ADK graph `data_analyst → plan_author`; provider switched by `ROOT_AGENT_MODEL` (native Gemini vs OpenAI-compatible via `LiteLlm`). |
| [`metrics.py`](machine_learning_engineering/metrics.py) | The two scorers — a bounded search reward vs the competition report metric. |
| [`pipeline.py`](machine_learning_engineering/pipeline.py) | End-to-end driver + CLI. |

**Benchmark / evaluation scripts** live in `scripts/` — see [`EXPERIMENTS.md`](EXPERIMENTS.md)
for the full map (`stage_tasks.py`, `run_autogluon.py`, `run_mlestar.py`,
`run_claude_pipeline.py`, `replay_from_run.py`, `collect_results.py`, `make_figures.py`).

The **revived MLE-STAR baseline** is `agent.py` + `sub_agents/` + `shared_libraries/` +
`runner.py` + `eval/`, driven by `scripts/run_mlestar.py`. It is deliberately *off* the MCTS
path — no extension feature belongs there — but it is no longer dead code: it is the arm we
compare against.

---

## Key features & design decisions

- **Structured JSON plan, not code-gen.** No `eval` of model output. The plan is
  schema-validated, centrally seeded (determinism is what MCTS needs), and made of named
  choices the search can reason about.
- **Import-level allow-list** is the safety envelope — a hallucinated operator is *dropped*,
  never executed. HP ranges are free-form (used as given) unless curated in `REGISTRY`.
- **Two scorers, never conflated** — a bounded, higher-is-better reward for the *search*; the
  competition metric only for the *final report*.
- **The bench is drawn on disk, before any method runs** — not honoured by convention inside
  each method, but enforced by *absence*: the holdout rows are simply not in any file the
  search can read. The same rule one level down: the ensemble selects on out-of-fold rows and
  reports on the holdout.
- **Persist the tree; never restart.** The scorer is fixed, so a config's reward never changes
  when the target moves — the tree is a running ablation we mine for free.
- **LLM-call complexity is the cost model** — O(1) per task, ≤ O(outer steps) with Extended Feature 3;
  never O(expansions) or O(rollouts).
- **Relational tables are a first-class, searchable stage** — `AggJoiner` over `aux_*.csv`.
  This is a *structural* capability the flat-table baselines lack.
- **One ADK stack, env-switched provider** — `ROOT_AGENT_MODEL` selects native Gemini (with
  `google_search`) or any OpenAI-compatible endpoint via `LiteLlm`. The logic layer imports no
  LLM client.

---

## Quick start

```bash
make sync          # build the venv (needs internet, run once)
make test          # offline test suite — mocked agents, real skrub CV (~5 min, 427 tests)

# a real run against a live LLM API (needs a key in .env — see docs/USAGE.md)
make run-live TASK=california-housing-prices BUDGET=40

# or fully offline, zero API quota (plans come from pre-written files):
make run-claude TASK=credit-fraud BUDGET=40 CLAUDE_TOP_K=3
```

Each run writes `runs/<task>_<ts>/` with `result.json`, `summary.md`, and `ensemble.pkl`.
Full command reference, provider setup, and output-field documentation:
[`docs/USAGE.md`](docs/USAGE.md).

---

## Results in one line

On a 10-task benchmark against AutoGluon and the revived MLE-STAR, the extension is
**competitive on quality at a small, fixed LLM-token cost** (2 calls/task, constant in the
search budget), and holds a **structural advantage on relational data** that flat-table
AutoML cannot match. **Important caveat:** the shipped extension numbers are *optimistically
biased* (they predate the on-disk split fix) and the three arms are scored on three different
bases — read [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md) before quoting any number.

---

## Repository layout

```
machine_learning_engineering/   # the MCTS extension (the project) + revived MLE-STAR baseline
scripts/                        # staging, benchmark drivers, figure pipeline (see EXPERIMENTS.md)
tests/                          # offline test suite (make test) — agents mocked with FakeLlm
data/                           # raw datasets (13)
results/                        # git-shareable mirror of run artifacts (result.json per run)
figures/                        # rendered comparison figures + comparison.csv
docs/                           # USAGE, PROJECT_STATE, BUG_LEDGER, architecture, per-stage notes
CLAUDE.md                       # the invariant-level design contract
```

> **Note:** the staged tasks (`machine_learning_engineering/tasks/` — the shared
> train/holdout split, drawn once by `scripts/stage_tasks.py`) ship with the
> repository, so no staging step is needed after cloning.
