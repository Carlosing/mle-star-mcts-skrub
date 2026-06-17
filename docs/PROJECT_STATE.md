# Project state & roadmap

**Thesis:** adapt MLE-STAR into an **MCTS search over skrub DataOps pipelines**.
LLM agents read the data and author a *rich JSON plan* (a menu of operators +
hyperparameter ranges per stage); a pure-code MCTS engine then searches that
space — structure **and** hyperparameters — over a fixed evaluation budget. The
LLM never does the search; it only proposes the space.

## Architecture — three layers

```
  AGENT LAYER (Gemini via ADK)            LOGIC LAYER (pure Python, no LLM)
  data_analyst ─► plan_author ──JSON──►   spec_resolver ─► skrub_ops ─► mcts
  (reads digest, (rich plan: ops +        (names+HP ->     (build plan,  (UCT
   web search)    HP ranges)               seeded instances) action space, search)
                                                            rollouts)
        └──────────── pipeline.py (driver) wires the whole thing ───────────┘
```

Design rule: **clients may be provider-native, but the logic layer imports no
LLM client.** See [agent-architecture.md](agent-architecture.md).

## End-to-end flow (`pipeline.run_pipeline`)

1. **load_task** → dataframe + target + task_type + metric (parses `task_description.txt`).
2. **make_data_summary** → compact EDA digest (the only thing the LLM sees of the data).
3. **ADK agents** → `data_analyst` (web search) → `plan_author` → JSON plan in state.
4. **resolve_spec** → JSON names/HP-ranges → seeded estimators + `choose_*` nodes (allowed-list, no `eval`).
5. **build_staged_plan** → skrub DataOps plan; **get_action_space** → the MCTS search space.
6. **mcts_search** → best config over `budget` rollouts (reward = bounded search scorer).
7. **report** → score the incumbent on the competition metric (RMSE, etc.).

## Module map

| File | Role |
|---|---|
| `mcts.py` | MCTS engine (UCT select/expand/backprop, persistent tree, `prior_fn` hook, DOT/ASCII viz) |
| `skrub_ops.py` | skrub glue: `build_staged_plan`, `get_action_space`, `apply_state`, seeded rollouts (configurable `scoring`), ablation |
| `spec_resolver.py` | LLM JSON → seeded estimators **+ HP `choose_*`**; curated allowed-list registry; clips HP ranges |
| `data_summary.py` | `make_data_summary` — EDA digest for the analyst |
| `adk_agent.py` | ADK graph `data_analyst → plan_author`; `google_search`; `build_root_agent` factory |
| `run_logging.py` | sanity log of prompts+outputs to JSONL (`log_dir`) |
| `metrics.py` | search-reward scorer (per task) vs report metric (competition) |
| `pipeline.py` | end-to-end driver + CLI |
| `agent.py` | legacy hand-rolled OpenAI `ManagerAgent` (teammates'; decoupled, kept for merge) |
| `probe_gemini.py` | standalone model/quota probe |

## Current state — done & tested

- ✅ MCTS engine; skrub layer (build/search/rollout/ablation)
- ✅ ADK agent graph on **native Gemini** (free AI Studio key); per-stack web search
- ✅ Data digest → LLM; **rich JSON plan** authored by the LLM (no hand-written menu)
- ✅ Spec resolver: allowed-list operators **+ hyperparameter search** (clipped ranges, seeded)
- ✅ End-to-end driver with search-vs-report scoring split
- ✅ Offline tests for every layer (agents mocked via `FakeLlm`); 1 gated live smoke test
- ✅ Python pinned to 3.13; `gemini-2.5-flash` as the default model

Run all: `uv run --no-sync python -m pytest tests/ -q`
Run live pipeline: `uv run --no-sync python -m machine_learning_engineering.pipeline --budget 30`

## What's left to add (roadmap)

**Search quality**
- **Conditional (model-gated) HP nesting** — today a non-selected model's HPs are
  inactive search dims (CASH); make them active only under their model.
- **LLM `prior_fn`** — warm-start child Q/N from a policy prior (AlphaZero-style).
- **Ablation / targeted-refinement outer loop** — wire `run_ablation` +
  `pick_target_node`, persist the tree across outer steps.
- **MCTS score cache** — memoize rollouts by state so over-budget / exhausted
  spaces don't re-evaluate (currently harmless but wasteful).

**Coverage**
- **Relational assemble auto-config** — LLM proposes `AggJoiner` configs for
  multi-table tasks (e.g. `fetch_credit_fraud`).
- **Ensemble / submission** generation (the MLE-STAR back half we cut).
- **Per-model training loss** as a searchable HP (e.g. NN criterion).
- More datasets + fixtures (`employee_salaries` for the encoder-choice story).

**Ops / housekeeping**
- `uv lock` reconcile when online (pyproject `requires-python` was tightened).
- Local/Docker version parity; stable live evaluation runs once quota is steady.

## Key design decisions (for slides / Q&A)

- **Structured JSON plan, not code-gen** — no `eval` of model output, central
  seeding (determinism MCTS needs), schema-validated, MCTS-friendly named choices.
- **Allowed-list registry** (not dynamic import) — novel ops/HPs are dropped or
  clipped, never executed.
- **Two scorers** — bounded higher-is-better reward for *search*; the competition
  metric only for the *final report* (they must not be conflated).
- **Provider-native clients, shared logic** — Gemini (ADK) and OpenAI stacks stay
  separate; the MCTS/skrub core is client-agnostic and import-clean.
