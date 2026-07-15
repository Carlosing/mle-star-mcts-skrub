# Demo & dataset guide

A quick orientation for the team: where the code is, what the demo shows, and
which (small, tabular) datasets best showcase it. Favor datasets that load in
one call and run in seconds. For the full status + roadmap see
[PROJECT_STATE.md](PROJECT_STATE.md).

## Where the project is right now

**Working & tested** (`make test`):

- **MCTS engine** ([mcts.py](../machine_learning_engineering/mcts.py)) — UCT
  select / expand / backprop, persistent tree, score cache, model-gated HPs,
  optional LLM-prior hook. Pure Python, no skrub/LLM needed.
- **skrub layer** ([skrub_ops.py](../machine_learning_engineering/skrub_ops.py))
  — action space, apply-config, seeded profile-aware rollouts, ablation.
- **Constructive staged pipeline** (`build_staged_plan`) — searches the
  *construction* of a pipeline across stages:
  **assemble (relational join) → clean → scope → encode → scale →
  feature-eng → model → hyperparameters**.
  See [docs/pipeline-stages.md](pipeline-stages.md).
- **Agent layer (ADK, two providers)** — `data_analyst` → `plan_author`
  ([adk_agent.py](../machine_learning_engineering/adk_agent.py)) author a rich
  JSON plan on native Gemini or an OpenAI-compatible endpoint
  (`PROVIDER=google|school`, see [agent-architecture.md](agent-architecture.md));
  `spec_resolver` resolves it to seeded estimators **with hyperparameter
  choices** (import allow-list only, no `eval`).
- **End-to-end driver** ([pipeline.py](../machine_learning_engineering/pipeline.py))
  — `run_pipeline`: task → data digest → agents → resolve → MCTS search
  (Extended Feature 1 focus hint, Extended Feature 3 injection, focused refinement) → Caruana
  ensemble → report on the shared on-disk holdout.
- **Benchmark harness** — AutoGluon and a revived, hard-capped MLE-STAR run on
  the *same* task, holdout and time budget, all three emitting one uniform
  `result.json` (`make bench-autogluon` / `make bench-mlestar` / `make figures`).

[PROJECT_STATE.md](PROJECT_STATE.md) is the canonical status. **The code is
complete and green, but the shipped results are knowingly biased in our favour** —
they predate the on-disk train/holdout split (2026-07-14), so the search chose its
configs with the eval rows visible while AutoGluon's did not. Read
[Hand-off](PROJECT_STATE.md#hand-off--the-published-numbers-are-biased) before
quoting any number.

## What the demo shows

1. **It's not just a tuner** — MCTS searches *structure* (which preprocessing /
   feature-engineering steps to add), not only hyperparameters.
2. **Reward climbs with enrichment** — a model alone is weak; adding the right
   encoder / feature step lifts the score (our `test_reward_climbs` story).
3. **Relational assemble** — aggregating an auxiliary table (skrub `AggJoiner`)
   surfaces signal a flat-table AutoML (AutoGluon) can't reach. This is our
   strongest differentiator.
4. **Encoder choice matters** — GapEncoder vs MinHashEncoder on dirty,
   high-cardinality categorical columns (skrub's specialty).

## What makes a good demo dataset

Pick datasets that **exercise the capabilities above** and stay small:

- ✅ **Small**: hundreds–few thousand rows (we subsample anyway; CV must be fast).
- ✅ **Mixed types**: numeric + categorical, so the encoder choice is meaningful.
- ✅ **High-cardinality / messy categoricals**: to show GapEncoder/MinHashEncoder.
- ✅ **Non-linear structure / interactions**: so feature-engineering and model
  choice visibly change the score (the enrichment story).
- ✅ **At least one multi-table** dataset: to demo the assemble/`AggJoiner` stage.
- ⚠️ Avoid: huge datasets, image/audio, anything needing heavy text embedding
  (TextEncoder pulls in torch — slower, save for later).

## Recommended datasets

All 13 demo datasets are already **staged as task dirs** under
`machine_learning_engineering/tasks/<name>/` (`train.csv`, `test.csv`,
optional `aux_*.csv`, `task_description.txt`) by
[scripts/stage_tasks.py](../scripts/stage_tasks.py) /
`make stage-credit-fraud` — run them directly with
`make run-live TASK=<name>` or offline with `make run-claude TASK=<name>`
(see [USAGE.md](USAGE.md) for the full task list and commands). Note that
three relational tasks (`country-happiness`, `flight-delays`, `movielens`)
are machinery stress tests, not performance benchmarks — on the latter two
every config scores r2 < 0.

The originals are skrub built-ins, one call to load:

```python
from skrub.datasets import fetch_employee_salaries, fetch_credit_fraud  # etc.
```

| Loader | Task | ~Size | Why it's good for us |
|---|---|---|---|
| `fetch_employee_salaries` | regression | ~9k | **Dirty high-cardinality categorical** (job titles) → the encoder-choice demo. skrub's flagship example |
| `fetch_credit_fraud` | classification | multi-table | **Relational** (baskets + products) → the `AggJoiner` assemble demo. Our differentiator |
| `fetch_midwest_survey` | classification | ~2.7k | Small, messy categoricals → clean + encode + model flow |
| `fetch_california_housing` | regression | ~20k (subsample) | All-numeric → feature-engineering + model-choice (enrichment) demo; already our default task |
| `fetch_bike_sharing` | regression | ~17k (subsample) | Datetime features → DatetimeEncoder |
| `fetch_country_happiness` | regression | tiny | Fast smoke tests; staged as relational (all signal via aggregate joins) |

**Suggested split:** `fetch_employee_salaries` (encoder choice) +
`fetch_credit_fraud` (relational assemble) cover our two headline capabilities;
add `fetch_california_housing` for a clean numeric enrichment-climbing curve.

Kaggle is fine too if you find something small and tabular — but the skrub
built-ins are zero-friction and chosen to highlight exactly our strengths.

## How to try one (the pattern)

```python
from skrub.datasets import fetch_employee_salaries
import machine_learning_engineering.skrub_ops as so   # or importlib-load it

bunch = fetch_employee_salaries()
df = bunch.employee_salaries     # inspect the bunch; attribute name varies per dataset
# (rename the label column to "target", or pass target=... to build_staged_plan)

spec = {                          # the hand-written stage menu (LLM does this later)
    "vectorizer": {"slots": {"high_cardinality": ["skrub.GapEncoder", "skrub.MinHashEncoder"]}},
    "stages": [{"name": "scale", "options": [None, StandardScaler()]}],
    "model": {"HGB": HistGradientBoostingRegressor(), "RF": RandomForestRegressor()},
}
plan = so.build_staged_plan(spec, df, target="current_annual_salary")
rollout = so.make_rollout_fn(plan, df)        # seeded, subsampled, fast
print(so.get_action_space(plan))              # the search space MCTS explores
print(rollout({"model": "HGB"}))              # score one configuration
```

For relational (`fetch_credit_fraud`), pass the auxiliary table:
`build_staged_plan(spec_with_assemble, main_df, aux_tables={"products": products_df})`.

Notes:
- `HistGradientBoosting*` is the fast default model (uses all cores via OpenMP);
  prefer it over plain `GradientBoosting`.
- Keep it small: rollouts subsample to ~500–2000 rows (profile-aware — imbalance
  and high-cardinality text grow it) and are seeded, so scores are reproducible.
