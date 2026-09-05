# EXPERIMENTS.md

Reproduction procedure, artifact locations, and — in
[Provenance](#provenance-of-the-archived-runs) — the per-run record of which
archived run produced which number and whether it is still valid. Every experiment
is a scripted `make` target. Knob-by-knob option reference:
[`docs/USAGE.md`](docs/USAGE.md).

> **Validity of the archived numbers.** Every archived benchmark run predates the
> on-disk train/holdout split (commit `517f954`, 2026-07-14 17:13), the three methods
> are scored on three different bases, and two tasks were measured on a training
> frame that no longer exists. The
> [provenance table](#provenance-of-the-archived-runs) marks each run accordingly;
> the full analysis is in
> [`EXPERIMENTAL_RESULTS.md § Status of the numbers`](EXPERIMENTAL_RESULTS.md#status-of-the-numbers).

---

## The 10 benchmark tasks

```
bike-sharing        country-happiness   credit-fraud        flight-delays       medical-charge
movielens           open-payments       toxicity            traffic-violations  videogame-sales
```

Four are **relational** — they ship `aux_*.csv` tables the extension can
aggregate-join via its `AggJoiner` stage, and which the flat-table baselines never
see:

| Task | Aux table(s) | Flat-table (main) columns |
|---|---|---|
| `country-happiness` | `aux_gdp`, `aux_life_expectancy`, `aux_legal_rights` | `Country, happiness_score` — string + target only |
| `credit-fraud` | `aux_products` | `ID, fraud_flag` — ID + target only |
| `movielens` | `aux_movies` | `userId, movieId, rating` — pure IDs |
| `flight-delays` | `aux_airports` | rich (carrier, times, origin/dest, distance) |

How much the aggregation is worth depends on how much signal the flat table retains
without it — from *none* (country-happiness, where AutoGluon **cannot run the task
at all**) to *most of it* (flight-delays). See
[`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md).

Three further staged datasets (`california-housing-prices`, `employee-salaries`,
`midwest-survey`) are used for development and replay smoke runs. They are outside
the benchmark allow-list and do not appear in the comparison figures.

### Task sizes (as staged today)

| Task | train | test | aux | type | metric |
|---|---|---|---|---|---|
| bike-sharing | 13 903 | 3 476 | — | regression | RMSE |
| country-happiness | 117 | 29 | 3 tables | regression | RMSE |
| credit-fraud | 12 000 | 3 000 | `aux_products` | classification | roc_auc |
| flight-delays | 12 000 | 3 000 | `aux_airports` | regression | RMSE |
| medical-charge | 16 000 | 4 000 | — | regression | RMSE |
| movielens | 11 999 | 3 000 | `aux_movies` | regression | RMSE |
| open-payments | 5 120 | 1 280 | — | classification | accuracy |
| toxicity | 800 | 200 | — | classification | accuracy |
| traffic-violations | 12 001 | 2 999 | — | classification | accuracy |
| videogame-sales | 13 258 | 3 314 | — | regression | RMSE |

---

## Step 0 — one-time setup

```bash
make sync
```

```bash
uv sync --extra bench
```

The first builds the venv from the lockfile; the second adds AutoGluon and
matplotlib, kept out of the core deps.

**The staged tasks ship with the repository** — every task dir
(`machine_learning_engineering/tasks/<task>/{train,test,test_answer}.csv` +
`task_description.txt`, aux tables included) is committed. Nothing needs
downloading or staging: clone or pull the current revision and the shared bench is
already in place.

To run the **live** extension or MLE-STAR you also need an LLM key in `.env` and
`PROVIDER=school` on the command — see
[`docs/USAGE.md § Choosing a provider`](docs/USAGE.md#choosing-a-provider-google-vs-school).
The `google` (Gemini) provider was used during development; that is why the switch
exists.

---

## Step 1 — the extension

**All 10 tasks, sequentially:**

```bash
make benchmark-extension PROVIDER=school
```

This drives [`scripts/run_benchmark.py`](scripts/run_benchmark.py), which holds the
per-task configs in its `EXTENSION` table — **the Makefile's `BUDGET`, `TOP_K` and
`NJOBS` do not reach this path** (they only feed the single-task `run-live` target),
so passing them to `make benchmark-extension` has no effect. Edit the table to
change a config.

The archived runs were configured as:

| Task | budget | n_jobs | Source run |
|---|---|---|---|
| bike-sharing | 60 | **1** | `bike-sharing_20260714-0809` |
| country-happiness | 60 | 6 | `country-happiness_20260713-0014` |
| credit-fraud | 60 | 6 | `credit-fraud_20260712-1806` |
| flight-delays | 60 | **1** | `flight-delays_20260714-0932` |
| medical-charge | 60 | 6 | `medical-charge_20260714-0854` |
| movielens | 100 | 6 | `movielens_20260713-1224` |
| open-payments | 100 | 6 | `open-payments_20260713-1300` |
| toxicity | 100 | 6 | `toxicity_20260713-0041` |
| traffic-violations | 40 | 6 | `traffic-violations_20260712-2301` |
| videogame-sales | 100 | 6 | `videogame-sales_20260713-1351` |

All runs used `top_k=3` and `n_proposes=2`.

> **`n_jobs=1` on bike-sharing and flight-delays** was forced: these two (the
> largest frames) overloaded memory with the default 6 fold-parallel workers.
>
> **Plan variance.** The configuration is reproducible; the LLM's plan is not. A
> re-run authors a fresh plan and can land a different score at the same config.
> For a bit-for-bit reproduction of an archived result, use the replay path below.

**One task at a time:**

```bash
make run-live TASK=<task> BUDGET=60 TOP_K=3 N_PROPOSES=2 PROVIDER=school
```

**Replay a captured plan on the current split** — this is how the four valid
numbers in `EXPERIMENTAL_RESULTS.md` §1 were produced. Zero API quota, and fully
deterministic: the plan and the proposals are read back from the source run's
artifacts, so nothing is re-authored.

`--source` takes a run directory name, resolved under `results/` — which is
committed, so any archived run replays from a fresh clone:

```bash
uv run python scripts/replay_from_run.py --source <task>_<ts> --n-proposes 2
```

→ **Log files:** a **live** run writes `runs/<task>_<ts>/` containing `result.json`
(all scores + the searched space), `summary.md`, `ensemble.pkl` (the fitted final
model), and the raw agent I/O (`data_analyst_*.json`, `plan_author_*.json`,
`proposer_*.json`, each with per-call `tokens`); Step 4 mirrors it into `results/`.
A **replay** makes no API calls, so it has no agent I/O to record and writes its
`result.json` straight into `results/replay_<task>_np<n>_<ts>/`.

---

## Step 2 — AutoGluon baseline

Flat-table only — it never sees `aux_*.csv`.

```bash
make benchmark-autogluon
```

One task at a time:

```bash
make bench-autogluon TASK=<task> TIME_BUDGET=3600
```

- Every archived AutoGluon run used `time_budget_s=3600`, `presets=best_quality`,
  `num_cpus=1`, and the script pins that. Budget roughly **10 CPU-hours** for the
  full sweep. Most tasks fill the hour; toxicity finishes in ~7 minutes.
- `num_cpus=1` is **required on Apple-Silicon Macs** — LightGBM/XGBoost otherwise
  segfault on a duplicate `libomp`. Raise it on Linux.
- **country-happiness is expected to fail** ("No models were trained"): the flat
  table is a string ID + target, so there is nothing to fit. The failure artifact
  *is* the archived result, not a bug to fix.

> **The archived AutoGluon runs are not on the current bench.** They predate
> `517f954` and carved their own 25% holdout out of `train.csv` via
> `ensemble.holdout_split(seed=42)`. A fresh run reads the on-disk `test.csv`
> instead. This is the cheapest method to re-run cleanly — CPU only, no API cost.

→ **Log files:** `runs/autogluon_<task>_<ts>/result.json` plus AutoGluon's own
multi-GB `ag_models/` (deliberately left out of the `results/` mirror).

---

## Step 3 — MLE-STAR baseline (revived upstream agent)

> **⚠ The shipped MLE-STAR numbers were not produced by these commands.**
>
> All 13 archived MLE-STAR results under `results/mle-star-*/` come from a
> teammate's earlier runs, ingested by
> [`scripts/convert_mlestar_final_state.py`](scripts/convert_mlestar_final_state.py).
> That script's own docstring is explicit: it writes *"MLE-STAR's own internal score
> (`submission_code_exec_result.score`), **not** a re-score against the shared
> holdout."*
>
> Concretely, in every one of those files:
> - **the score is a validation split MLE-STAR carved out of `train.csv` itself.**
>   Its generated scripts are instructed to split train/validation, print
>   `Final Validation Performance:`, and *not* to use `test.csv` — see the
>   teammate's report,
>   [`docs/mle_star_implementation_report.pdf`](docs/mle_star_implementation_report.pdf)
>   §2.7 and §5.4.
> - **`llm_calls: 0` and `wall_clock_s: 0.0` are hardcoded**, not measured — there
>   is no call count or wall clock for this method.
> - **`time_budget_s: 300` is `exec_timeout`** — the timeout on a single generated
>   script — not a run budget. It is not comparable to AutoGluon's `3600`.
>
> `scripts/run_mlestar.py` *does* score against the shared `test.csv` /
> `test_answer.csv` and *does* record calls and wall clock, but **has never been run
> end-to-end**. It is the one step of a clean three-way comparison that costs real
> API budget.

To run it properly:

```bash
make benchmark-mlestar PROVIDER=school
```

One task at a time:

```bash
make bench-mlestar TASK=<task> MAX_CALLS=60 TIME_BUDGET=3600 PROVIDER=school
```

`MAX_CALLS` aborts the run after that many LLM calls; there is also a per-call
token bound and a wall-clock cap. Treat its result as **one data point**, not a
swept curve — and a task can legitimately end with **no runnable script**
(videogame-sales did in the archived set). That failure is itself a reportable
result.

→ **Log files:** `runs/mlestar_<task>_<ts>/result.json`; the ingested runs are under
`results/mle-star-<task>/` (`result.json` + `final_state.json`).

---

## Step 4 — collect results and render figures

```bash
make collect-results
```

```bash
make figures RUNS=results
```

`collect-results` mirrors each run directory from `runs/` into `results/` whole —
`result.json`, `summary.md`, `ensemble.pkl` and the raw agent I/O — so an archived
run stays replayable. AutoGluon's multi-GB `ag_models/` is excluded, and
`MAX_FILE_MB` (default 25) caps any single file.

→ **Outputs** (`figures/`):

| File | What it shows |
|---|---|
| `quality_at_cost.png` | One panel per task (5×2 grid). Bars are keyed by **(method, bench)**: a hatched `extension (clean)` bar appears wherever a current-bench run exists, never averaged into the plain pre-fix bar. Each bar is the **mean of the runs in its group**, with min–max error bars — so a bar with `n>1` will not equal the single value tabled in `EXPERIMENTAL_RESULTS.md`. |
| `token_cost.png` | Cumulative real-LLM tokens vs Optional Feature 3 call count, every task on one axes. |
| `proposal_scaling.png` | Extension quality vs proposal count, from the Step-1 replays. Pre-fix and current-bench runs are drawn as **separate series**, never averaged — they are scored on different holdouts. |
| `mechanism_table.md` | The qualitative mechanism comparison. |
| `comparison.csv` | The flat table behind the figures — one row per run: `score`, `tokens`, `llm_calls`, `wall_clock_s`, `time_budget_s`, `relational`. |

Two non-obvious properties of `comparison.csv`:

- **A run whose `holdout` is null is silently dropped.** The "cannot run" and "no
  runnable script" outcomes are *absences* from the CSV, not rows.
- **The `relational` column is per-method, not per-task.** It reads `True` for
  every MLE-STAR row and `False` for every other row. It does not mark which tasks
  ship aux tables.

The committed figures are current: they were regenerated from `results/` as it
stands, after the budget-sweep runs were removed.

---

## Provenance of the archived runs

The cut line is commit **`517f954`, 2026-07-14 17:13** — the on-disk split.
`dbc0f77` (17:56) added out-of-fold ensemble selection and the `selection` stamp.

**How to tell a clean run from a biased one:**

| `ensemble.selection` | Meaning |
|---|---|
| `"oof_3fold"` | **Clean** — post-fix, selection and reporting on disjoint rows |
| *key absent* | **Pre-fix** — biased |
| `"legacy_holdout"` | Producible only by `--legacy-ensemble` on a *fresh* run |

### Extension

| Run | Task | budget | holdout | Status |
|---|---|---|---|---|
| `bike-sharing_20260714-0809` | bike-sharing | 60 | −38.00 | pre-fix |
| `country-happiness_20260712-2337` | country-happiness | 60 | *(none)* | pre-fix, failed |
| `country-happiness_20260713-0014` | country-happiness | 60 | −753.87 | pre-fix |
| `credit-fraud_20260712-1806` | credit-fraud | 60 | 0.826 | pre-fix, **void** ⚠ |
| `flight-delays_20260714-0932` | flight-delays | 60 | −36.31 | pre-fix |
| `medical-charge_20260714-0854` | medical-charge | 60 | −2003.1 | pre-fix |
| `movielens_20260713-1224` | movielens | 100 | −1.006 | pre-fix |
| `open-payments_20260713-1300` | open-payments | 100 | 0.941 | pre-fix, **void** ⚠ |
| `toxicity_20260712-2242` | toxicity | 100 | 0.845 | pre-fix |
| `toxicity_20260713-0041` | toxicity | 100 | 0.945 | pre-fix |
| `traffic-violations_20260712-2020` | traffic-violations | 60 | 0.837 | pre-fix |
| `traffic-violations_20260712-2301` | traffic-violations | 40 | 0.885 | pre-fix |
| `videogame-sales_20260713-1351` | videogame-sales | 100 | −1.278 | pre-fix |

Note the two toxicity runs (0.845 / 0.945) and the two traffic-violations runs
(0.837 / 0.885). `EXPERIMENTAL_RESULTS.md`'s archived table quotes the better of
each pair.

### Extension — replays (zero API cost)

| Run | Task | N | holdout | Status |
|---|---|---|---|---|
| `replay_credit-fraud_np0_20260713-2234` | credit-fraud | 0 | 0.809 | pre-fix, void ⚠ |
| `replay_credit-fraud_np1_20260713-2114` | credit-fraud | 1 | 0.828 | pre-fix, void ⚠ |
| `replay_credit-fraud_np2_20260713-2337` | credit-fraud | 2 | 0.824 | pre-fix, void ⚠ |
| **`replay_credit-fraud_np2_20260715-1325`** | credit-fraud | 2 | **0.807** | **CLEAN** (`oof_3fold`) |
| `replay_toxicity_np{0,1,2}_20260713-*` | toxicity | 0,1,2 | 0.940 / 0.950 / 0.950 | pre-fix |
| `replay_traffic-violations_np{0,1,2}_2026071{3,4}-*` | traffic-violations | 0,1,2 | 0.885 / 0.883 / 0.884 | pre-fix |
| **`replay_traffic-violations_np2_20260715-1401`** | traffic-violations | 2 | **0.8876** | **CLEAN** (`oof_3fold`) |
| `replay_country-happiness_np{0,1,2}_20260713-*` | country-happiness | 0,1,2 | −753.87 (all) | pre-fix |
| **`replay_country-happiness_np2_20260715-1321`** | country-happiness | 2 | **−568.70** | **CLEAN** (`oof_3fold`) |
| **`replay_bike-sharing_np2_20260715-1358`** | bike-sharing | 2 | **−35.26** | **CLEAN** (by date; no ensemble block, so no stamp) |

### AutoGluon

All 11 runs (2026-07-13 → 07-14 01:39) are **pre-fix**: clean with respect to
selection, but scored on their own 25% carve-out rather than the on-disk `test.csv`.
`autogluon_country-happiness_*` (two attempts) produced no model.
`autogluon_credit-fraud_*` and `autogluon_open-payments_*` are additionally
**void** ⚠ — measured on the pre-shrink `train.csv`.

### MLE-STAR

All 13 ingested runs are **pre-fix and self-reported** (see the Step 3 warning). Ten
are benchmark tasks; three (`california-housing-prices`, `employee-salaries`,
`midwest-survey`) are outside the allow-list. `mle-star-videogame-sales` recorded no
score — no runnable script. `mle-star-credit-fraud` scored exactly 0.500 roc_auc
(chance).

### ⚠ The "void" marker

Commit `517f954` could not recover a labelled holdout for four tasks and carved a
fresh one out of `train.csv`, **shrinking the training frame**:

| Task | train before | after |
|---|---|---|
| credit-fraud | 15 000 | 12 000 |
| open-payments | 6 400 | 5 120 |
| *california-housing-prices* (dev) | 2 400 | 1 920 |
| *employee-salaries* (dev) | 6 400 | 5 120 |

Archived numbers for these tasks — **extension and AutoGluon alike** — were measured
on a dataset that no longer exists in this repository. The commit message states it
plainly: *"results previously measured on them are void."* The remaining eight tasks
re-staged byte-identically.

---

## The offline test suite

```bash
make test
```

427 tests, ~5 minutes. Every layer (MCTS, skrub glue, spec resolver, search loop,
ensemble, agents) has offline tests; the agents are scripted with `FakeLlm`, so the
suite makes **zero network calls** — but the skrub cross-validation is real. A green
suite is the precondition for trusting any run. It is a correctness gate, not a
benchmark.

---

## Script map

| Script | Role |
|---|---|
| `scripts/run_benchmark.py` | The benchmark reproducer (`make benchmark-<method>`): one method over all 10 tasks, `budget` pinned per task. |
| `scripts/run_autogluon.py` | The AutoGluon method (`make bench-autogluon`). |
| `scripts/run_mlestar.py` | The MLE-STAR method under hard caps (`make bench-mlestar`). Scores the shared bench. **Never run end-to-end.** |
| `scripts/convert_mlestar_final_state.py` | Ingests a teammate's MLE-STAR `final_state.json`. Source of all 13 shipped MLE-STAR results; writes a self-reported validation score. |
| `scripts/replay_from_run.py` | Re-runs the search from a captured plan at zero agent cost. **This produced the four valid numbers.** |
| `scripts/collect_results.py` | `runs/` → the `results/` mirror. |
| `scripts/make_figures.py` | `results/` → `figures/`. |
