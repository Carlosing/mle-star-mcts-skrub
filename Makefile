# MLE-STAR x skrub x MCTS — common tasks.
# Uses `uv run` so it runs against the existing venv without
# re-resolving the lockfile. Run `make sync` once (online) to build/reconcile.

# MCTS evaluations for the live run (e.g. make run-live BUDGET=50)
BUDGET ?= 15
# task name under tasks/ (empty = config default)
TASK ?=

.PHONY: help sync test test-live probe run-live

help:
	@echo "make sync       - reconcile lockfile + build the venv (run once, online)"
	@echo "make test       - run the offline test suite (mocked agents, no API)"
	@echo "make test-live  - run the suite including the gated live Gemini test"
	@echo "make probe      - probe Gemini models + live quota (probe_gemini.py)"
	@echo "make run-live   - full pipeline on the REAL API; writes runs/<task>_<ts>/"
	@echo "                  (vars: BUDGET=$(BUDGET), TASK=<name>)"

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
		--budget $(BUDGET) $(if $(TASK),--task $(TASK),)
