# MLE-STAR x skrub x MCTS — common tasks.
# Uses `uv run` so it runs against the existing venv without
# re-resolving the lockfile. Run `make sync` once (online) to build/reconcile.

# MCTS evaluations per phase (e.g. make run-live BUDGET=50)
BUDGET ?= 15
# task name under tasks/ (empty = config default)
TASK ?=
# search phases; >1 enables ablation targeting (Option 1)
OUTER_STEPS ?= 1
# set REFINE=1 to enable LLM per-stage option injection (Option 3; needs OUTER_STEPS>1)
REFINE ?=

.PHONY: help sync test test-live probe run-live run-refine

help:
	@echo "make sync       - reconcile lockfile + build the venv (run once, online)"
	@echo "make test       - run the offline test suite (mocked agents, no API)"
	@echo "make test-live  - run the suite including the gated live Gemini test"
	@echo "make probe      - probe Gemini models + live quota (probe_gemini.py)"
	@echo "make run-live   - full pipeline on the REAL API; writes runs/<task>_<ts>/"
	@echo "                  vars: BUDGET=$(BUDGET) TASK=<name> OUTER_STEPS=$(OUTER_STEPS) REFINE=1"
	@echo "make run-refine - run-live with Option 1 + Option 3 on (OUTER_STEPS=3, REFINE=1)"

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
		--budget $(BUDGET) --outer-steps $(OUTER_STEPS) \
		$(if $(TASK),--task $(TASK),) $(if $(REFINE),--refine,)

run-refine:
	$(MAKE) run-live OUTER_STEPS=3 REFINE=1 BUDGET=$(BUDGET) TASK=$(TASK)
