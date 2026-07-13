# MLE-STAR x skrub x MCTS — common tasks.
# Uses `uv run` so it runs against the existing venv without
# re-resolving the lockfile. Run `make sync` once (online) to build/reconcile.

# MCTS evaluations per phase; multiples of 4 keep the HP-refinement bonus
# phase (ceil(budget/4)) a round number — 20 for a small search, 80 for a
# large one (e.g. make run-live BUDGET=80).
BUDGET ?= 20
# task name under tasks/ (empty = config default)
TASK ?=
# search phases; >1 enables the tree-mined focus-stage pick (Option 1, proposer hint)
OUTER_STEPS ?= 1
# set REFINE=1 to enable LLM per-stage option injection (Option 3; needs OUTER_STEPS>1)
REFINE ?=
# interleave N proposer calls between BUDGET-sized slices (overrides OUTER_STEPS/REFINE)
N_PROPOSES ?=
# ensemble the top-k incumbents from the score cache (1 = off)
TOP_K ?= 1
# LLM provider for the live agents: google (native Gemini) or school (OpenAI-
# compat via LiteLlm). Picks {PROVIDER}_ROOT_AGENT_MODEL/_API_KEY/_API_BASE
# from .env (see machine_learning_engineering.__init__._select_provider).
PROVIDER ?= google
# CV fold-parallelism for rollouts (default 6 P-cores; safe with boosters)
NJOBS ?= 6
# wall-clock budget (seconds) for the whole search; empty = pure rollout-count
# budget (BUDGET). Set e.g. TIME_BUDGET=3600 for the 1h benchmark protocol.
TIME_BUDGET ?=
# --- Claude-driven (offline, zero Gemini quota) --------------------------------
# proposer calls interleaved between BUDGET-sized slices (0 = Option 1 + HP-refine only)
CLAUDE_PROPOSES ?= 2
# top-k ensemble for the Claude driver (1 = off)
CLAUDE_TOP_K ?= 3
# artifact parent dir (empty = runs/claude_<timestamp>)
OUT ?=

.PHONY: help sync test probe probe-school run-live run-refine \
        run-claude sweep-claude stage-tasks stage-credit-fraud \
        bench-autogluon bench-mlestar figures

help:
	@echo "make sync       - reconcile lockfile + build the venv (run once, online)"
	@echo "make test       - run the offline test suite (mocked agents, no API)"
	@echo "make probe        - probe Gemini models + live quota (probe_gemini.py)"
	@echo "make probe-school - list school (GWDG) models; SMOKE=1 to health-check each"
	@echo "make run-live   - full pipeline on the REAL API; writes runs/<task>_<ts>/"
	@echo "                  vars: PROVIDER=$(PROVIDER) (google|school) BUDGET=$(BUDGET) TASK=<name> TOP_K=<k> N_PROPOSES=<n> NJOBS=$(NJOBS)"
	@echo "                  full usage guide: docs/USAGE.md"
	@echo "make run-refine - run-live with Option 1 + Option 3 on (OUTER_STEPS=3, REFINE=1)"
	@echo ""
	@echo "-- Claude-driven, offline, ZERO Gemini quota (plans+proposals in scripts/claude_agents.py) --"
	@echo "make run-claude   - one task through the Claude driver; writes runs/claude_<ts>/<task>/"
	@echo "                    vars: TASK=<name> BUDGET=$(BUDGET) CLAUDE_PROPOSES=$(CLAUDE_PROPOSES) CLAUDE_TOP_K=$(CLAUDE_TOP_K) OUT=<dir>"
	@echo "make sweep-claude - ALL tasks in scripts/claude_agents.py, same driver (a quota-free sweep)"
	@echo ""
	@echo "make stage-tasks        - stage every data/ dataset missing from tasks/ (offline)"
	@echo "make stage-credit-fraud - download + stage the relational credit-fraud task (online, once)"
	@echo ""
	@echo "-- benchmark comparison (needs: uv sync --extra bench) --"
	@echo "make bench-autogluon - AutoGluon baseline, same holdout + time budget; TASK=<name> TIME_BUDGET=3600 NUM_CPUS=1 PRESETS=best_quality"
	@echo "make bench-mlestar   - revived MLE-STAR under hard caps (spends LLM budget); TASK=<name> MAX_CALLS=60 PROVIDER=$(PROVIDER)"
	@echo "make figures         - render comparison figures from result.json artifacts; RUNS=runs OUT=<dir>"
	@echo "  (extension side: 'make run-live TIME_BUDGET=3600 ...' to fill the same budget at constant LLM cost)"

sync:
	uv lock && uv sync

test:
	uv run python -m pytest tests/ -q

probe:
	uv run python probe_gemini.py

# list (and with SMOKE=1, health-check) the school OpenAI-compat models
probe-school:
	uv run python probe_school.py $(if $(SMOKE),--smoke,) $(if $(MODEL),--model $(MODEL),)

run-live:
	PROVIDER=$(PROVIDER) uv run python -m machine_learning_engineering.pipeline \
		--budget $(BUDGET) --outer-steps $(OUTER_STEPS) --top-k $(TOP_K) \
		--n-jobs $(NJOBS) \
		$(if $(TIME_BUDGET),--time-budget-s $(TIME_BUDGET),) \
		$(if $(TASK),--task $(TASK),) $(if $(REFINE),--refine,) \
		$(if $(N_PROPOSES),--n-proposes $(N_PROPOSES),)

run-refine:
	$(MAKE) run-live OUTER_STEPS=3 REFINE=1 BUDGET=$(BUDGET) TASK=$(TASK)

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

# --- benchmark comparison (needs the `bench` extra: uv sync --extra bench) -----
# AutoGluon baseline under the same wall-clock budget on the SAME shared holdout.
# NUM_CPUS=1 (default) is REQUIRED on macOS-ARM (LightGBM/XGBoost double-libomp
# segfault); raise on Linux for full AutoGluon parallelism. PRESETS=best_quality
# (default) enables bagging+stacking so the fit actually spends time_limit
# instead of returning in a few seconds regardless of budget.
NUM_CPUS ?= 1
PRESETS ?= best_quality
bench-autogluon:
	OMP_NUM_THREADS=1 uv run python scripts/run_autogluon.py \
		--task $(TASK) --time-budget-s $(if $(TIME_BUDGET),$(TIME_BUDGET),3600) \
		--num-cpus $(NUM_CPUS) --presets $(PRESETS) $(if $(OUT),--out $(OUT),)

# Revived MLE-STAR under hard caps (best-effort; spends real LLM budget). Set
# MAX_CALLS to bound the debug-cascade token cost. Uses PROVIDER for the model.
MAX_CALLS ?= 60
bench-mlestar:
	PROVIDER=$(PROVIDER) OMP_NUM_THREADS=1 uv run python scripts/run_mlestar.py \
		--task $(TASK) --max-calls $(MAX_CALLS) \
		--time-budget-s $(if $(TIME_BUDGET),$(TIME_BUDGET),3600) \
		$(if $(OUT),--out $(OUT),)

# Read every uniform result.json under RUNS and render the comparison figures.
RUNS ?= runs
figures:
	uv run python scripts/make_figures.py --runs $(RUNS) \
		--out $(if $(OUT),$(OUT),runs/figures)
