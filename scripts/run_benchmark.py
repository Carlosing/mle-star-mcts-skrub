"""Re-run the shipped benchmark, method by method, with the archived configs.

Each method's per-task configuration below is pinned from the MOST RECENT
archived `result.json` for that (task, method) pair under `results/` — so
`make benchmark-<method>` reproduces the experiment set as it was actually run,
in order, rather than a generic sweep. Tasks run sequentially; a failure on one
task is recorded and the sequence continues.

Heads-ups baked into the configs (see also EXPERIMENTS.md):
- **AutoGluon used the 1-hour mark on every task** (`time_budget_s=3600`,
  `presets=best_quality`, `num_cpus=1` — required on macOS-ARM, libomp).
  Budget ~10 CPU-hours for the full arm.
- **bike-sharing and flight-delays ran the extension at `n_jobs=1`** — the
  fold-parallel workers overloaded memory on these two (largest frames);
  every other task uses the default `n_jobs=6`.
- **The extension needs a live LLM key** (`PROVIDER=school` was used for every
  archived run: GWDG `openai/qwen3.5-397b-a17b`). Expect plan variance across
  re-runs (EXPERIMENTAL_RESULTS.md heads-up 1) — the config is reproducible,
  the LLM's plan is not.
- **MLE-STAR's archived numbers are ingested self-reported runs** with no local
  config to pin; the arm below is the *clean* protocol (`run_mlestar.py`,
  MAX_CALLS=60, 1h cap) that scores the shared bench.

Example:
    $ uv run python scripts/run_benchmark.py --method autogluon
    [1/10] autogluon bike-sharing ... ok (3583s)
    ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# The 10 benchmark tasks, in the order the arms are run.
TASKS = [
    "bike-sharing", "country-happiness", "credit-fraud", "flight-delays",
    "medical-charge", "movielens", "open-payments", "toxicity",
    "traffic-violations", "videogame-sales",
]

# ---------------------------------------------------------------------------
# Extension — pinned from the latest live run per task (all: n_proposes=2,
# PROVIDER=school / openai/qwen3.5-397b-a17b). `n_jobs=1` where the archived
# run needed it to avoid memory overload (bike-sharing, flight-delays).
# ---------------------------------------------------------------------------
EXTENSION = {
    # task: (budget, top_k, n_jobs)         source run
    "bike-sharing":       (60, 1, 1),   # bike-sharing_20260714-0809 (n_jobs=1: memory)
    "country-happiness":  (60, 3, 6),   # country-happiness_20260713-0014
    "credit-fraud":       (60, 3, 6),   # credit-fraud_20260712-1806 (3 subsample seeds auto)
    "flight-delays":      (60, 1, 1),   # flight-delays_20260714-0932 (n_jobs=1: memory)
    "medical-charge":     (60, 2, 6),   # medical-charge_20260714-0854
    "movielens":          (100, 3, 6),  # movielens_20260713-1224
    "open-payments":      (100, 1, 6),  # open-payments_20260713-1300
    "toxicity":           (100, 2, 6),  # toxicity_20260713-0041
    "traffic-violations": (40, 3, 6),   # traffic-violations_20260712-2301
    "videogame-sales":    (100, 1, 6),  # videogame-sales_20260713-1351
}
EXTENSION_N_PROPOSES = 2  # every archived run: llm_calls=4 = 2 agents + 2 proposals

# ---------------------------------------------------------------------------
# AutoGluon — every archived run: the 1-hour budget, best_quality, num_cpus=1.
# country-happiness is EXPECTED to fail ("No models were trained") — the flat
# table is a string ID + target; the failure artifact is itself the result.
# ---------------------------------------------------------------------------
AUTOGLUON_TIME_BUDGET_S = 3600
AUTOGLUON_PRESETS = "best_quality"
AUTOGLUON_NUM_CPUS = 1  # REQUIRED on macOS-ARM (LightGBM/XGBoost libomp segfault)

# ---------------------------------------------------------------------------
# MLE-STAR — the clean protocol (scores the shared test.csv/test_answer.csv).
# videogame-sales may fail to produce a runnable script — that is a result.
# ---------------------------------------------------------------------------
MLESTAR_MAX_CALLS = 60
MLESTAR_TIME_BUDGET_S = 3600


def _run(cmd: list[str], env: dict | None = None) -> int:
    """Run one task's command, streaming output; return the exit code.

    Example:
        _run([sys.executable, "-m", "machine_learning_engineering.pipeline", ...]) -> 0
    """
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(cmd, env=full_env).returncode


def cmd_extension(task: str) -> tuple[list[str], dict]:
    budget, top_k, n_jobs = EXTENSION[task]
    return (
        [sys.executable, "-m", "machine_learning_engineering.pipeline",
         "--task", task, "--budget", str(budget), "--top-k", str(top_k),
         "--n-proposes", str(EXTENSION_N_PROPOSES), "--n-jobs", str(n_jobs)],
        {"PROVIDER": os.environ.get("PROVIDER", "school")},
    )


def cmd_autogluon(task: str) -> tuple[list[str], dict]:
    return (
        [sys.executable, "scripts/run_autogluon.py",
         "--task", task, "--time-budget-s", str(AUTOGLUON_TIME_BUDGET_S),
         "--num-cpus", str(AUTOGLUON_NUM_CPUS), "--presets", AUTOGLUON_PRESETS],
        {"OMP_NUM_THREADS": "1"},
    )


def cmd_mlestar(task: str) -> tuple[list[str], dict]:
    return (
        [sys.executable, "scripts/run_mlestar.py",
         "--task", task, "--max-calls", str(MLESTAR_MAX_CALLS),
         "--time-budget-s", str(MLESTAR_TIME_BUDGET_S)],
        {"PROVIDER": os.environ.get("PROVIDER", "school"), "OMP_NUM_THREADS": "1"},
    )


METHODS = {"extension": cmd_extension, "autogluon": cmd_autogluon, "mlestar": cmd_mlestar}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(METHODS),
                        help="which arm to run over all 10 tasks, in order")
    parser.add_argument("--tasks", nargs="*", default=TASKS,
                        help="subset/override of tasks (default: all 10, in order)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running them")
    args = parser.parse_args()

    make_cmd = METHODS[args.method]
    failures: list[str] = []
    for i, task in enumerate(args.tasks, 1):
        cmd, env = make_cmd(task)
        print(f"[{i}/{len(args.tasks)}] {args.method} {task}")
        print("   ", " ".join(cmd), {k: v for k, v in env.items()})
        if args.dry_run:
            continue
        t0 = time.time()
        rc = _run(cmd, env)
        dt = time.time() - t0
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"    -> {status} ({dt:.0f}s)")
        if rc != 0:
            failures.append(task)

    if failures:
        print(f"\n{len(failures)} task(s) failed: {', '.join(failures)}")
        print("(For autogluon country-happiness / mlestar videogame-sales a failure "
              "is the expected, reportable outcome — see EXPERIMENTAL_RESULTS.md.)")
        sys.exit(1)
    print("\nall tasks completed")


if __name__ == "__main__":
    main()
