"""Run the full pipeline on every task with Claude standing in for Gemini.

Claude authored the per-task plans and Option-3 proposal menus offline (see
``scripts/claude_agents.py``), reading the same data digest + allowed-operator
vocabulary the real ``data_analyst``/``plan_author`` agents receive. This driver
wires them into the unchanged logic layer:

    load_task -> make_data_summary -> [Claude plan via spec_raw]
      -> resolve_spec -> build_staged_plan -> MCTS (Option 1 + gated HP search)
      -> Option 3 (Claude replay proposer) -> HP-refinement -> top-k ensemble

**Zero Gemini calls** — the search, ablation, HP-refinement and ensemble are
already pure code; the two LLM touchpoints are supplied by Claude via injection.

Run:
    uv run python scripts/run_claude_pipeline.py                 # all tasks
    uv run python scripts/run_claude_pipeline.py --task credit-fraud
    uv run python scripts/run_claude_pipeline.py --budget 40 --n-proposes 2 --top-k 3
"""

import argparse
import os
import time
from datetime import datetime

from machine_learning_engineering import pipeline
from scripts.claude_agents import (
    ALL_TASKS,
    make_replay_proposer,
    proposal_for,
    spec_raw,
)


def run_task(task: str, out_dir: str, budget: int, n_proposes: int, top_k: int,
             seed: int, legacy_ensemble: bool = False) -> dict:
    """Run one task offline (Claude plan + Claude proposer) and return the result."""
    outer_steps = n_proposes + 1
    refine = n_proposes > 0
    # the task's own type selects the plan / proposal when none was authored
    task_type = pipeline.load_task(task)[2]
    propose = (
        make_replay_proposer(proposal_for(task, task_type), log_dir=out_dir)
        if refine
        else None
    )
    return pipeline.run_pipeline(
        task_name=task,
        budget=budget,
        out_dir=out_dir,
        seed=seed,
        outer_steps=outer_steps,
        propose=propose,
        top_k=top_k,
        legacy_ensemble=legacy_ensemble,
        spec_raw=spec_raw(task, task_type),  # <- skips both ADK agents; no Gemini
    )


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None, help="one task (default: all 5)")
    parser.add_argument("--budget", type=int, default=20, help="rollouts per slice")
    parser.add_argument(
        "--n-proposes",
        type=int,
        default=2,
        help="Option-3 proposer calls between slices (0 = off; Option 1 + HP-refine only)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="ensemble top-k (1 = off)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--legacy-ensemble",
        action="store_true",
        help="select the ensemble on the reported holdout (pre-2026-07-14 logic; "
             "optimistic). Stamped ensemble.selection='legacy_holdout'",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="parent dir for run artifacts (default: runs/claude_<timestamp>)",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else ALL_TASKS
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    parent = args.out or os.path.join("runs", f"claude_{stamp}")
    os.makedirs(parent, exist_ok=True)

    rows = []
    for i, task in enumerate(tasks, 1):
        out_dir = os.path.join(parent, task)
        print(f"\n[{i}/{len(tasks)}] {task}  ->  {out_dir}")
        started = time.time()
        try:
            r = run_task(
                task, out_dir, args.budget, args.n_proposes, args.top_k,
                args.seed, args.legacy_ensemble
            )
            rep = r.get("report") or {}
            ens = r.get("ensemble") or {}
            rows.append(
                {
                    "task": task,
                    "type": r["task_type"],
                    "search": _fmt(r["best_search_score"]),
                    "report": f"{rep.get('scorer', '')}={_fmt(rep.get('score'))}"
                    if rep
                    else "-",
                    "ensemble": _fmt(ens.get("ensemble_score")) if ens else "-",
                    "injected": ",".join(r.get("injected_options", [])) or "-",
                    "proposer_calls": r["llm_calls"],  # spec_raw set -> base 0, so == n_proposes
                    "wall_s": round(time.time() - started, 1),
                    "status": "ok",
                }
            )
            print(
                f"    search={_fmt(r['best_search_score'])}  "
                f"report={rows[-1]['report']}  ensemble={rows[-1]['ensemble']}  "
                f"injected={rows[-1]['injected']}  "
                f"({rows[-1]['wall_s']}s, offline_proposer_calls={rows[-1]['proposer_calls']}, gemini_calls=0)"
            )
        except Exception as exc:
            rows.append(
                {"task": task, "status": "FAILED", "error": repr(exc)[:200]}
            )
            print(f"    FAILED: {exc!r}")

    print("\n" + "=" * 70)
    print(f"Summary ({parent}):")
    for row in rows:
        if row["status"] == "ok":
            print(
                f"  {row['task']:<26} {row['type']:<14} "
                f"search={row['search']}  {row['report']:<28} "
                f"ens={row['ensemble']}  inj=[{row['injected']}]"
            )
        else:
            print(f"  {row['task']:<26} FAILED  {row.get('error', '')}")
    print("\nGemini network calls: 0 (Claude authored every plan + proposal offline).")
    print(f"Artifacts: {parent}/<task>/{{result.json,summary.md}}")


if __name__ == "__main__":
    main()
