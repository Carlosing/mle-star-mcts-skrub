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
# --- Claude-driven (offline, zero Gemini quota) --------------------------------
# proposer calls interleaved between BUDGET-sized slices (0 = Option 1 + HP-refine only)
CLAUDE_PROPOSES ?= 2
# top-k ensemble for the Claude driver (1 = off)
CLAUDE_TOP_K ?= 3
# artifact parent dir (empty = runs/claude_<timestamp>)
OUT ?=

.PHONY: help sync test test-live probe run-live run-refine sweep sweep-live \
        run-claude sweep-claude stage-tasks stage-credit-fraud

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
	@echo "make sweep-live - alias for 'make sweep' (explicit: it hits the real Gemini API)"
	@echo ""
	@echo "-- Claude-driven, offline, ZERO Gemini quota (plans+proposals in scripts/claude_agents.py) --"
	@echo "make run-claude   - one task through the Claude driver; writes runs/claude_<ts>/<task>/"
	@echo "                    vars: TASK=<name> BUDGET=$(BUDGET) CLAUDE_PROPOSES=$(CLAUDE_PROPOSES) CLAUDE_TOP_K=$(CLAUDE_TOP_K) OUT=<dir>"
	@echo "make sweep-claude - ALL tasks in scripts/claude_agents.py, same driver (a quota-free sweep)"
	@echo ""
	@echo "make stage-tasks        - stage every data/ dataset missing from tasks/ (offline)"
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

# same thing, named so it is obvious at the call site that this spends quota
sweep-live: sweep

# Claude stands in for the Gemini agent layer: plans + Option-3 proposals come
# from scripts/claude_agents.py, so the search runs with zero network calls.
run-claude:
	uv run python scripts/run_claude_pipeline.py \
		--budget $(BUDGET) --n-proposes $(CLAUDE_PROPOSES) --top-k $(CLAUDE_TOP_K) \
		$(if $(TASK),--task $(TASK),) $(if $(OUT),--out $(OUT),)

# every task claude_agents.py has a plan for, one after another
sweep-claude:
	uv run python scripts/run_claude_pipeline.py \
		--budget $(BUDGET) --n-proposes $(CLAUDE_PROPOSES) --top-k $(CLAUDE_TOP_K) \
		$(if $(OUT),--out $(OUT),)

stage-tasks:
	uv run python scripts/stage_tasks.py

stage-credit-fraud:
	uv run python scripts/stage_credit_fraud.py
