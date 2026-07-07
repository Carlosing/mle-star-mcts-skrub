# MLE-STAR x skrub x MCTS — common tasks.
# Uses `uv run` so it runs against the existing venv without
# re-resolving the lockfile. Run `make sync` once (online) to build/reconcile.

# MCTS evaluations per phase; multiples of 4 keep the HP-refinement bonus
# phase (ceil(budget/4)) a round number — 20 for a small search, 80 for a
# large one (e.g. make run-live BUDGET=80).
BUDGET ?= 20
# task name under tasks/ (empty = config default)
TASK ?=
# search phases; >1 enables ablation targeting (Option 1)
OUTER_STEPS ?= 1
# set REFINE=1 to enable LLM per-stage option injection (Option 3; needs OUTER_STEPS>1)
REFINE ?=
# interleave N proposer calls between BUDGET-sized slices (overrides OUTER_STEPS/REFINE)
N_PROPOSES ?=
# ensemble the top-k incumbents from the score cache (1 = off)
TOP_K ?= 1
# JSON sweep spec for `make sweep` (defaults + runs; see sweeps/example.json)
SWEEP ?= sweeps/example.json

.PHONY: help sync test test-live probe run-live run-refine sweep stage-credit-fraud

help:
	@echo "make sync       - reconcile lockfile + build the venv (run once, online)"
	@echo "make test       - run the offline test suite (mocked agents, no API)"
	@echo "make test-live  - run the suite including the gated live Gemini test"
	@echo "make probe      - probe Gemini models + live quota (probe_gemini.py)"
	@echo "make run-live   - full pipeline on the REAL API; writes runs/<task>_<ts>/"
	@echo "                  vars: BUDGET=$(BUDGET) TASK=<name> OUTER_STEPS=$(OUTER_STEPS) REFINE=1 N_PROPOSES=<n>"
	@echo "make run-refine - run-live with Option 1 + Option 3 on (OUTER_STEPS=3, REFINE=1)"
	@echo "make sweep      - run a JSON sweep spec on the REAL API; writes runs/sweep_<ts>/"
	@echo "                  vars: SWEEP=$(SWEEP) (agents run once per task, spec reused)"
	@echo "make stage-credit-fraud - download + stage the relational credit-fraud task (online, once)"

sync:
	uv lock && uv sync

test:
	uv run python -m pytest tests/ -q

test-live:
	RUN_LIVE_TESTS=1 uv run python -m pytest tests/ -q

probe:
	uv run python probe_gemini.py

run-live:
	uv run python -m machine_learning_engineering.pipeline \
		--budget $(BUDGET) --outer-steps $(OUTER_STEPS) --top-k $(TOP_K) \
		$(if $(TASK),--task $(TASK),) $(if $(REFINE),--refine,) \
		$(if $(N_PROPOSES),--n-proposes $(N_PROPOSES),)

run-refine:
	$(MAKE) run-live OUTER_STEPS=3 REFINE=1 BUDGET=$(BUDGET) TASK=$(TASK)

sweep:
	uv run python -m machine_learning_engineering.sweep $(SWEEP)

stage-credit-fraud:
	uv run python scripts/stage_credit_fraud.py
