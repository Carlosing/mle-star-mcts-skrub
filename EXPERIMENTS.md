# EXPERIMENTS.md

How to reproduce the experiments, and where the log files are. Every experiment is a
scripted `make` target; nothing here needs a raw Python invocation. For a knob-by-knob
reference of the options mentioned below, see [`docs/USAGE.md`](docs/USAGE.md).

> **Reproducibility note.** The benchmark is a three-way comparison — our **extension** vs
> **AutoGluon** vs the revived **MLE-STAR** — on a **shared, on-disk train/holdout split**
> drawn once by `scripts/stage_tasks.py` (seed 42) *before any method runs*. Every arm trains
> on `train.csv`, predicts the rows of `test.csv`, and is scored against `test_answer.csv`.
> That shared split is what makes the three numbers comparable. Heads-up: the shipped
> extension numbers predate the split fix (small optimistic bias, measured), the three arms
> are not all scored on the same basis, and the LLM's plan is the one non-deterministic input
> — see the heads-up section of [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md).

---

## The 10 benchmark tasks

The comparison is run over these 10 tasks (the intersection of what all three arms could run):

```
bike-sharing        country-happiness   credit-fraud        flight-delays       medical-charge
movielens           open-payments       toxicity            traffic-violations  videogame-sales
```

Four tasks are **relational** — they ship `aux_*.csv` tables the extension can aggregate-join
via its `AggJoiner` stage, and which the flat-table baselines never see:

| Task | Aux table(s) | Flat-table (main) columns |
|---|---|---|
| `country-happiness` | `aux_gdp`, `aux_life_expectancy`, `aux_legal_rights` | `Country, happiness_score` — string + target only |
| `credit-fraud` | `aux_products` | `ID, fraud_flag` — ID + target only |
| `movielens` | `aux_movies` | `userId, movieId, rating` — pure IDs |
| `flight-delays` | `aux_airports` | rich (carrier, times, origin/dest, distance) |

The predictive payoff of the aggregation depends on how much signal the flat table retains
without it — from *none* (country-happiness, where AutoGluon **cannot run the task at all**) to
*most of it* (flight-delays). See [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md) for the
full discussion.

Three further staged datasets (`california-housing-prices`, `employee-salaries`,
`midwest-survey`) are used for development/replay smoke runs but are outside the benchmark
allow-list and do not appear in the comparison figures.

---

## Step 0 — one-time setup

```bash
make sync                 # build the venv from the lockfile (online, once)
uv sync --extra bench     # add AutoGluon + matplotlib (kept out of the core deps)
```

**The staged tasks ship with the repository** — every task dir
(`machine_learning_engineering/tasks/<task>/{train,test,test_answer}.csv` +
`task_description.txt`, aux tables included) is committed, so no staging step is needed after
cloning. The split's *provenance* is `scripts/stage_tasks.py` (seeded, drawn once before any
method ran); `make stage-tasks` exists only to re-derive a task dir from `data/` and is a safe
no-op on dirs that already exist. (`scripts/stage_tasks.py --force` restages, which invalidates
every archived number for that task — don't, unless you mean to.)

To run the **live** extension or MLE-STAR you also need an LLM key in `.env` — see
[`docs/USAGE.md § Choosing a provider`](docs/USAGE.md#choosing-a-provider-google-vs-school).
The **offline** paths (`make test`, `make run-claude`) need no key.

---

## Step 1 — the extension (our method)

**All 10 tasks, sequentially, with the exact archived configurations:**

```bash
make benchmark-extension          # PROVIDER=school by default (what every archived run used)
```

This drives [`scripts/run_benchmark.py`](scripts/run_benchmark.py), where each task's
configuration is **pinned from its most recent archived `result.json`** — reproducing the
shipped experiment set, not a generic sweep. The actual configs (all runs: `n_proposes=2`,
provider `school` / `openai/qwen3.5-397b-a17b`):

| Task | budget | top_k | n_jobs | Source run |
|---|---|---|---|---|
| bike-sharing | 60 | 1 | **1** | `bike-sharing_20260714-0809` |
| country-happiness | 60 | 3 | 6 | `country-happiness_20260713-0014` |
| credit-fraud | 60 | 3 | 6 | `credit-fraud_20260712-1806` |
| flight-delays | 60 | 1 | **1** | `flight-delays_20260714-0932` |
| medical-charge | 60 | 2 | 6 | `medical-charge_20260714-0854` |
| movielens | 100 | 3 | 6 | `movielens_20260713-1224` |
| open-payments | 100 | 1 | 6 | `open-payments_20260713-1300` |
| toxicity | 100 | 2 | 6 | `toxicity_20260713-0041` |
| traffic-violations | 40 | 3 | 6 | `traffic-violations_20260712-2301` |
| videogame-sales | 100 | 1 | 6 | `videogame-sales_20260713-1351` |

> **Heads-up — `n_jobs=1` on bike-sharing and flight-delays.** These two (the largest frames)
> overloaded memory with the default 6 fold-parallel workers; their archived runs used a single
> worker, and the script pins that. Everything else runs at the default `n_jobs=6`.
>
> **Heads-up — plan variance.** The configuration is reproducible; the LLM's plan is not
> (EXPERIMENTAL_RESULTS.md heads-up 1). A re-run authors a fresh plan and can land a different
> score at the same config. For a bit-for-bit reproduction of an *archived* result, use the
> replay path below instead.

**One task at a time** (the generic form, any knobs):

```bash
make run-live TASK=<task> BUDGET=60 TOP_K=3 N_PROPOSES=2 PROVIDER=school
```

**Offline / zero-quota alternative** — same search engine, plans come from pre-written files
(`scripts/claude_agents.py`), so it costs no API quota and is fully deterministic:

```bash
make run-claude TASK=<task> BUDGET=60 CLAUDE_PROPOSES=2 CLAUDE_TOP_K=3   # one task
make sweep-claude BUDGET=60                                              # every authored task
```

**Extended Feature 3 scaling** (quality vs number of mid-search proposer calls) is produced by replaying a
captured plan at N=0,1,2 injections — zero API quota:

```bash
uv run python scripts/replay_from_run.py --run runs/<task>_<ts> --n-proposes 0
uv run python scripts/replay_from_run.py --run runs/<task>_<ts> --n-proposes 1
uv run python scripts/replay_from_run.py --run runs/<task>_<ts> --n-proposes 2
```

→ **Log files:** each run writes `runs/<task>_<ts>/` containing
`result.json` (all scores + the searched space), `summary.md` (human-readable report),
`ensemble.pkl` (the fitted final model), and the raw agent I/O
(`data_analyst_*.json`, `plan_author_*.json`, `proposer_*.json`, each with per-call `tokens`).
The git-shareable mirror of these is under `results/` (see Step 4).

---

## Step 2 — AutoGluon baseline

Same holdout, flat-table only (it never sees `aux_*.csv`).

**All 10 tasks, sequentially, with the archived configuration:**

```bash
make benchmark-autogluon
```

> **Heads-up — every archived AutoGluon run used the 1-hour mark** (`time_budget_s=3600`,
> `presets=best_quality`, `num_cpus=1`), and the script pins exactly that. Budget roughly
> **10 CPU-hours** for the full arm. Most tasks fill the hour (bagging + stacking); a few
> finish early (toxicity ~7 min).
>
> **Heads-up — country-happiness is expected to fail** ("No models were trained"): the flat
> table is a string ID + target, so there is nothing to fit. The failure artifact *is* the
> archived result — don't debug it.

- `num_cpus=1` is **required on Apple-Silicon Macs** — LightGBM/XGBoost otherwise segfault on a
  duplicate `libomp`. Raise it on Linux for full parallelism.
- One task at a time: `make bench-autogluon TASK=<task> TIME_BUDGET=3600`.

→ **Log files:** `runs/autogluon_<task>_<ts>/result.json` (the uniform schema) plus AutoGluon's
own multi-GB `ag_models/` (left behind by the `results/` mirror on purpose).

---

## Step 3 — MLE-STAR baseline (revived upstream agent)

The original code-writing agent, revived and **hard-capped** (its token cost is otherwise
unbounded).

**All 10 tasks, sequentially, under the clean protocol:**

```bash
make benchmark-mlestar            # MAX_CALLS=60, 1h cap per task, PROVIDER=school
```

One task at a time:

```bash
make bench-mlestar TASK=<task> MAX_CALLS=60 TIME_BUDGET=3600 PROVIDER=school
```

- `MAX_CALLS` aborts the run after that many LLM calls; there is also a per-call token bound and
  the wall-clock cap. Treat its result as **one data point**, not a swept curve — and a task can
  legitimately end with **no runnable script** (videogame-sales did in the archived set); that
  failure is itself a reportable result.

> **Caveat on the shipped MLE-STAR numbers.** The 13 archived MLE-STAR runs under
> `results/mle-star-*/` were produced by a teammate's earlier runs and ingested with
> `scripts/convert_mlestar_final_state.py`. Their score is MLE-STAR's **own internal
> validation**, *not* the shared holdout — a self-reported basis. `run_mlestar.py` *does* score
> against the shared `test.csv`/`test_answer.csv`, so a fresh `make bench-mlestar` run is the
> clean way to put MLE-STAR on the shared bench. See
> [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md).

→ **Log files:** `runs/mlestar_<task>_<ts>/result.json`; the ingested runs are under
`results/mle-star-<task>/` (`result.json` + `final_state.json`).

---

## Step 4 — collect results and render figures

```bash
make collect-results        # runs/ → the small git-shareable results/ mirror
make figures RUNS=results   # results/ → figures/
```

→ **Outputs** (`figures/`):

| File | What it shows |
|---|---|
| `quality_at_cost.png` | One panel per task (5×2 grid); each method's shared-holdout score with its token spend annotated. |
| `token_cost.png` | Every task on one axes — cumulative real-LLM tokens vs Extended Feature 3 call count (the "cost stays a small constant" story). |
| `proposal_scaling.png` | Extension quality vs Extended Feature 3 call count, from the Step-1 replays. |
| `mechanism_table.md` | The qualitative mechanism comparison (search mechanism, LLM-call count, leakage handling, …). |
| `comparison.csv` | The flat table behind all the figures — one row per (task, method): `score`, `tokens`, `llm_calls`, `wall_clock_s`, `relational`. |

---

## The offline test suite (not a benchmark, but the correctness gate)

```bash
make test        # 427 tests, ~5 min — agents mocked with FakeLlm, but real skrub CV
```

Every layer (MCTS, skrub glue, spec resolver, search loop, ensemble, agents) has offline tests;
the agents are scripted with `FakeLlm` so the suite makes **zero network calls**. A green suite
is the precondition for trusting any run.

---

## Script map (what each script is for)

| Script | Role |
|---|---|
| `scripts/run_benchmark.py` | **The benchmark reproducer** (`make benchmark-<method>`): runs one arm over all 10 tasks in order, with each task's config pinned from its most recent archived `result.json`. |
| `scripts/stage_tasks.py` | Draws the shared on-disk split; encodes per-dataset knowledge (leaky columns, subsample size, aux joins). |
| `scripts/stage_credit_fraud.py` | Downloads + stages the relational credit-fraud task. |
| `scripts/run_autogluon.py` | AutoGluon arm (`make bench-autogluon`). |
| `scripts/run_mlestar.py` | Revived MLE-STAR arm under hard caps (`make bench-mlestar`). |
| `scripts/claude_agents.py` | Pre-written offline plans + replay proposer (the zero-quota agent stand-in). |
| `scripts/run_claude_pipeline.py` | Offline extension driver (`make run-claude` / `make sweep-claude`). |
| `scripts/replay_from_run.py` | Re-runs the search from a captured plan (Extended Feature 3 scaling, A/Bs) at zero agent cost. |
| `scripts/convert_mlestar_final_state.py` | Ingests a teammate's MLE-STAR `final_state.json` into the uniform `result.json` schema. |
| `scripts/collect_results.py` | `runs/` → the `results/` mirror. |
| `scripts/make_figures.py` | `results/` → `figures/`. |
