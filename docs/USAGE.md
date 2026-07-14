# How to run this project

A practical guide to the commands you'll actually use. Everything runs through
`make`, so you rarely type a raw Python command. If you just want to see it work,
skip to [Quick start](#quick-start).

> **What this project does, in one sentence:** LLM agents read a dataset and
> write a *menu* of ML-pipeline options; a pure-code search (MCTS) then tries
> combinations of those options and reports the best one. See the top-level
> `README`/`CLAUDE.md` for the full story.

---

## One-time setup

```bash
make sync          # build the virtual environment (needs internet, run once)
```

Create a `.env` file in the project root for the API keys (only needed for the
*live* commands — the offline ones need nothing). See
[Choosing a provider](#choosing-a-provider-google-vs-school) below.

---

## Quick start

```bash
make test                       # 1. run the offline test suite (no API, ~5 min)
make probe                      # 2. check your Google/Gemini key has quota
make run-live TASK=california-housing-prices BUDGET=40   # 3. a real run
```

Step 1 needs no keys and proves the code works on your machine. Step 3 writes a
folder under `runs/` with the results — see [What you get back](#what-you-get-back).

---

## The commands

### Offline test suite — no API, no keys

```bash
make test          # the full suite; agents are faked, so zero network calls
```

**Expect:** a pytest progress bar ending in `427 passed` with zero skips; takes
~5 minutes (it runs real skrub cross-validation, not mocks — only the *agents*
are faked). If this is green, the code is healthy — run it first whenever you
pull changes.

### Probes — "is my API key working?"

```bash
make probe                     # Google/Gemini: lists models + checks live quota
make probe-school              # School endpoint (GWDG): lists available models
make probe-school SMOKE=1      # + sends a tiny prompt to each, reports behaviour
make probe-school MODEL=qwen3.5-397b-a17b SMOKE=1   # smoke just one model
```

**Expect:** a list of model names. `make probe` also prints `OK — quota
available` per model. `make probe-school SMOKE=1` labels each model
`reasoning`/`instruct` and whether it returned usable text — useful because the
school endpoint's model list changes and some models are temporarily down.
Run a probe before a live run if you're unsure the endpoint is up.

### Live runs — the real thing

```bash
make run-live TASK=california-housing-prices BUDGET=40
make run-live TASK=credit-fraud BUDGET=40 TOP_K=3
make run-live TASK=employee-salaries BUDGET=60 N_PROPOSES=1   # with Option 3
```

This runs the whole pipeline against a real LLM API and writes a results folder.
See [Options](#options-and-their-defaults) for every knob and
[What you get back](#what-you-get-back) for the output.

### Offline "Claude" runs — zero API quota

```bash
make run-claude TASK=credit-fraud BUDGET=40      # one task, no network
make sweep-claude                                 # every task, no network
```

Same search engine, but the plans come from pre-written files instead of a live
LLM — so it costs **no quota** and is fully deterministic. Great for testing the
search itself without spending API calls.

---

## Options and their defaults

Pass these as `NAME=value` after the `make` command, e.g.
`make run-live TASK=credit-fraud BUDGET=60 TOP_K=3`.

| Option | Default | What it does |
|---|---|---|
| `TASK` | (config default) | Which dataset to run. See [Tasks](#available-tasks). |
| `BUDGET` | `20` | How many pipeline configs the search tries per phase. Higher = more thorough, slower. A further `budget/4` refinement rollouts run automatically. |
| `TOP_K` | `1` | Ensemble the best `K` configs at the end (`1` = just report the single best). |
| `N_PROPOSES` | (off) | Ask the LLM to add new options mid-search this many times (**Option 3**). `0` or unset = off. |
| `OUTER_STEPS` | `1` | Split the budget into this many search phases. `>1` enables the between-phase focus-stage pick (a hint passed to the Option-3 proposer). |
| `REFINE` | (off) | `REFINE=1` turns on Option 3 (needs `OUTER_STEPS>1`). `N_PROPOSES` is the simpler way to do this. |
| `PROVIDER` | `google` | Which LLM provider: `google` (Gemini) or `school` (GWDG). See below. |
| `NJOBS` | `6` | How many CPU cores to use per pipeline evaluation. `6` suits an Apple M-series; lower it on a smaller machine. |
| `TIME_BUDGET` | (off) | Wall-clock budget in **seconds** for the whole search (e.g. `3600` = 1 hour). When set, `BUDGET` becomes an upper bound and time is the real cap. The LLM cost stays the same 2 calls — you buy quality with *time*, not tokens. Used for the benchmark protocol. |
| `CLAUDE_PROPOSES` | `2` | Option-3 proposer calls for `make run-claude` (offline). |
| `CLAUDE_TOP_K` | `3` | Top-K ensemble for `make run-claude`. |

**Rules of thumb:**
- `BUDGET=20` is a quick look; `BUDGET=60`–`80` is a serious run.
- Add `TOP_K=3` to see the best few configs, not just one.
- `N_PROPOSES=1` lets the LLM extend the plan during the search (costs 1 extra
  LLM call per proposal).

---

## Choosing a provider (google vs school)

Both providers' keys live in `.env`, prefixed. Pick one per run with `PROVIDER=`.

```dotenv
# Google / Gemini
GOOGLE_API_KEY=...
GOOGLE_ROOT_AGENT_MODEL=gemini-2.5-flash

# School / GWDG (OpenAI-compatible). The model id MUST start with "openai/".
SCHOOL_API_KEY=...
SCHOOL_API_BASE=https://chat-ai.academiccloud.de/v1
SCHOOL_ROOT_AGENT_MODEL=openai/qwen3.5-397b-a17b
```

```bash
make run-live PROVIDER=google TASK=credit-fraud BUDGET=40   # Gemini (default)
make run-live PROVIDER=school TASK=credit-fraud BUDGET=40   # GWDG Qwen
```

Notes:
- `PROVIDER=google` is the default, so you can omit it for Gemini runs.
- The school models are mostly *reasoning* models (they "think" before
  answering) — this is handled automatically. Use `make probe-school` to see
  what's currently available; the list changes over time.
- Web search grounding is Gemini-only; school runs simply run without it.
- **Token cost (measured):** because school models spend *completion* tokens
  thinking, a run costs roughly twice a Gemini one. A measured `toxicity` run on
  GWDG's Qwen was **~12.6k tokens** for the two agent calls (`data_analyst`
  2.9k, `plan_author` 9.7k; ~7.7k of that total was completion/thinking). This
  is the *whole* per-task LLM cost — it does not grow with `BUDGET` or
  `TIME_BUDGET`, only `+1 call` per `N_PROPOSES`.

---

## What you get back

Each live run creates `runs/<task>_<timestamp>/`. The files you'll read:

- **`summary.md`** — the human-readable report: the task, the plan the LLM
  wrote, the best configuration found, its score, and any plan-quality warnings.
- **`result.json`** — the same data as machine-readable JSON (every score, the
  full search space, the ensemble, etc.).
- **`ensemble.pkl`** — the *fitted* final ensemble (the selected members, already
  fit on all of `train.csv`, plus their Caruana weights). Reload and predict:

  ```python
  import pickle
  from machine_learning_engineering import pipeline

  pred = pickle.load(open("runs/<task>_<ts>/ensemble.pkl", "rb"))
  holdout = pipeline.load_holdout("<task>")
  pred.predict(holdout)                      # flat task
  pred.predict(holdout, aux=aux_tables)      # relational task — pass aux back
  ```

  Aux tables are deliberately *not* stored inside the pickle (they can dwarf the
  models), so hand them back at predict time exactly as they were at fit time.

The rest are logs of exactly what each agent said:
`data_analyst_*.json`, `plan_author_*.json`, `proposer_*.json`. Each response log
now also records that call's `tokens`.

**What `result.json` records** (the fields you'll care about):

| Field | Meaning |
|---|---|
| `method` | `"extension"` (AutoGluon / MLE-STAR baselines emit `"autogluon"` / `"mlestar"`). |
| `best_state` | The winning pipeline config (encoder, model, tuned HPs). |
| `best_search_score` | The internal MCTS reward (bounded, higher-is-better). |
| `report` `{scorer, score}` | The incumbent on the competition metric via **CV** (reporting only). |
| `holdout` `{scorer, score}` | The incumbent on the **shared on-disk holdout** (`test.csv` + `test_answer.csv`, drawn by `stage_tasks.py` before any method sees the data) — the number directly comparable across the extension, AutoGluon and MLE-STAR. |
| `ensemble` | Top-`K` ensemble vs the single incumbent, on the same holdout. |
| `ensemble.selection` | **How the members were chosen** — see [below](#ensembleselection--and-the-legacy_ensemble-flag). A result is never ambiguous about its own provenance. |
| `llm_calls` | Exact number of LLM calls made (**`2`** for a plain run: `data_analyst` + `plan_author`; `+1` per `N_PROPOSES`). |
| `tokens` `{prompt, completion, total}` | **Real token cost** of the whole run. |
| `tokens_by_agent` | Per-agent token breakdown (`data_analyst`, `plan_author`, `proposer`), each `{prompt, completion, total, calls}`. |
| `time_budget_s` / `wall_clock_s` | The wall-clock budget (if set) and the measured run time. |
| `action_space`, `injected_options`, `spec_raw`, `analysis`, `data_summary` | The searched space, any Option-3 additions, the raw plan, and the agent I/O. |

The extension's LLM cost is a **fixed constant** — `tokens` does not grow with
`BUDGET`, `TOP_K`, or `TIME_BUDGET`, only with `N_PROPOSES`. That is the whole
point of the design: you scale quality with compute/time at constant token cost.

### `ensemble.selection` — and the `LEGACY_ENSEMBLE` flag

Caruana has to *pick* its members on some rows, and those rows must not be the
ones it reports on — otherwise `ensemble_score` is a greedy maximum over the
published metric rather than an out-of-sample number.

| `selection` | Meaning |
|---|---|
| `oof_3fold` | **Default.** Selected on 3-fold out-of-fold predictions spanning *all* of `train.csv`, then refit on all of train and scored on the untouched holdout. |
| `legacy_holdout` | Selected on the reported holdout itself — **optimistic**. The pre-2026-07-14 logic. |
| `inner_split` | Too few rows to fold; one split of train. |
| `single_member` | A one-config pool; nothing to select. |

Results measured **before 2026-07-14 were produced with `legacy_holdout`** and
carry that optimistic bias. The path is kept so those runs stay reproducible and
so the two can be A/B'd on an identical split:

```bash
make run-live TASK=california-housing-prices LEGACY_ENSEMBLE=1   # the old logic
make run-live TASK=california-housing-prices                     # the honest one
```

On california-housing the legacy path reports **−57128** against the honest
**−57240** — it looks ~112 RMSE better purely from selecting on the rows it
publishes. Note the corollary: with honest selection `ensemble_score` is *no
longer guaranteed* to beat the best pool member. That guarantee **was** the bias.

**Check `selection` on any fresh artifact you intend to quote — it must read
`oof_3fold`.**

**The terminal also prints a short summary at the end:**

```
Task: california-housing-prices  target=median_house_value  (regression)
Search scorer: r2  |  fallback spec: False
Best config:   {'model': 'HistGradientBoostingRegressor', 'learning_rate': ..., ...}
Best search score (r2): 0.7571
Report (neg_root_mean_squared_error): -57040.77
Top-3 ensemble (...): -59822.28 (incumbent -60456.63)
```

How to read it:
- **`fallback spec: False`** — good; the LLM's plan was valid. `True` means the
  LLM output couldn't be used and a minimal default plan was substituted.
- **`Best config`** — the winning pipeline (encoder, model, tuned settings).
- **`Best search score`** — the internal search metric (higher is better, 0–1).
- **`Report`** — the *competition* metric on held-out data (this is the number
  that "counts"; for RMSE it's negative, so closer to 0 is better).
- **`⚠ Plan quality warnings`** in `summary.md` (if present) — stages the LLM
  described but that couldn't be searched (e.g. only one option). Informational,
  not a failure.

---

## Benchmark comparison (extension vs AutoGluon vs MLE-STAR)

A three-way, **time-budgeted** comparison on the **same task and the same seeded
holdout**. Each method gets a fixed wall-clock budget (e.g. 1 hour); we compare
the `holdout` score they reach and the **tokens/LLM calls** each spent to get
there. Needs the benchmark extra once:

```bash
uv sync --extra bench        # installs AutoGluon + matplotlib (kept out of core)
```

**1 — the extension**, filling the budget at constant LLM cost:

```bash
make run-live TASK=toxicity TIME_BUDGET=3600 TOP_K=3 PROVIDER=school
```

**2 — AutoGluon** (the well-known AutoML baseline), same budget, same holdout:

```bash
make bench-autogluon TASK=toxicity TIME_BUDGET=3600
```

- `NUM_CPUS=1` (default) is **required on Apple-Silicon Macs** — LightGBM/XGBoost
  otherwise crash with a silent segfault (the same duplicate-`libomp` issue the
  extension avoids by pinning boosters to one thread). Raise it on Linux.
- AutoGluon is **flat-table only** — on relational tasks (credit-fraud) it never
  sees the `aux_*.csv`; that's exactly the extension's relational advantage.

**3 — MLE-STAR** (the original agent, revived and **hard-capped**), best-effort:

```bash
make bench-mlestar TASK=toxicity MAX_CALLS=60 TIME_BUDGET=3600 PROVIDER=school
```

- MLE-STAR writes and debugs code, so its token cost is **unbounded** — this is
  the whole reason for the caps. `MAX_CALLS` aborts the run once that many LLM
  calls have fired; there is also a per-call token bound and the wall-clock cap.
  Treat its result as **one data point**, not a swept curve.

**Render the comparison** from every `result.json` produced:

```bash
make figures RUNS=runs   # quality-at-cost + mechanism table + time-scaling
```

The time-scaling curve is drawn per task from any *extension* `result.json`
artifacts under `RUNS` whose budget differs (e.g. several
`make run-live BUDGET=...` runs), skipped when no task has two budget points.

This writes `figures/` (project root): `quality_at_cost.png`, `time_scaling.png`,
`mechanism_table.md`, and a flat `comparison.csv`. All three methods emit the
same `result.json` schema (`method`, `holdout`, `tokens`, `llm_calls`,
`wall_clock_s`), so the figure script reads them uniformly.

> **The headline story:** the extension's token cost is a small **constant**
> (2 LLM calls, `+1` per `N_PROPOSES`) no matter how large `BUDGET`/`TIME_BUDGET`
> gets, while MLE-STAR's grows with every code-and-debug step. Plot quality
> against tokens and the extension sits at a fixed, cheap x; MLE-STAR trails off
> to the right; AutoGluon sits at the origin (no LLM).

---

## Troubleshooting

- **`make test` fails** → something is wrong with the code or environment; fix
  this before anything else. Re-run `make sync` if dependencies look off.
- **A live run says `fallback spec: True`** → the LLM's plan couldn't be parsed.
  Check the `plan_author_response.json` in the run folder. Often a transient
  model hiccup — just re-run.
- **Live run errors with a quota / 503 message** → the provider is rate-limited
  or briefly down. Wait and retry, or switch `PROVIDER`. `make probe` /
  `make probe-school` tells you if the endpoint is up.
- **A run is slow** → lower `BUDGET`, or `NJOBS` if you're on a small machine.

---

## Available tasks

`bike-sharing`, `california-housing-prices`, `country-happiness`,
`credit-fraud`, `employee-salaries`, `flight-delays`, `medical-charge`,
`midwest-survey`, `movielens`, `open-payments`, `toxicity`,
`traffic-violations`, `videogame-sales`.

The small, fast ones (`california-housing-prices`, `country-happiness`,
`toxicity`) are good for a first run.
