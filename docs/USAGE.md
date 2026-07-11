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
make test                       # 1. run the offline test suite (no API, ~3.5 min)
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
make test-live     # same, plus one real Gemini smoke test (needs a key)
```

**Expect:** a pytest progress bar ending in `292 passed, 1 skipped`. The 1
skipped is the live Gemini test that only runs under `make test-live`. Takes
~3.5 minutes. If this is green, the code is healthy — run it first whenever you
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

### Sweeps — compare many settings at once

```bash
make sweep SWEEP=sweeps/example.json                  # real API
make sweep SWEEP=sweeps/example.json DRIVER=claude    # offline, no quota
```

Runs a grid of configurations from a JSON spec and writes a `sweep.csv` +
`sweep.md` comparing them. The LLM is called only *once per task* (the plan is
reused across every setting), so a big sweep still costs very little quota.

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
| `OUTER_STEPS` | `1` | Split the budget into this many search phases. `>1` enables between-phase re-targeting. |
| `REFINE` | (off) | `REFINE=1` turns on Option 3 (needs `OUTER_STEPS>1`). `N_PROPOSES` is the simpler way to do this. |
| `PROVIDER` | `google` | Which LLM provider: `google` (Gemini) or `school` (GWDG). See below. |
| `NJOBS` | `6` | How many CPU cores to use per pipeline evaluation. `6` suits an Apple M-series; lower it on a smaller machine. |
| `SWEEP` | `sweeps/example.json` | The sweep spec file for `make sweep`. |
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

---

## What you get back

Each live run creates `runs/<task>_<timestamp>/`. The two files you'll read:

- **`summary.md`** — the human-readable report: the task, the plan the LLM
  wrote, the best configuration found, its score, and any plan-quality warnings.
- **`result.json`** — the same data as machine-readable JSON (every score, the
  full search space, the ensemble, etc.).

The rest are logs of exactly what each agent said:
`data_analyst_*.json`, `plan_author_*.json`, `proposer_*.json`.

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
