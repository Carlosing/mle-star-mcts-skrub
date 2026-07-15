"""Convert MLE-STAR final_state.json artifacts into uniform result.json files.

This script is a one-off migration helper. It:

1. Moves each ``results/openai/<task>/`` directory (which contains a raw
   ``final_state.json`` from ``machine_learning_engineering.agent.run_pipeline``)
   to ``results/mle-star-<task>/``.
2. Deletes leftover loose files under ``results/openai/`` (batch registry,
   debug logs).
3. Writes a ``result.json`` compatible with ``scripts/make_figures.py`` using
   MLE-STAR's own internal score (``submission_code_exec_result.score``), *not*
   a re-score against the shared holdout. This avoids importing the project
   environment and makes the figure pipeline runnable immediately.

Run:
    python scripts/convert_mlestar_final_state.py

Safe to re-run: it will overwrite existing ``result.json`` files and replace
already-moved directories.
"""

import argparse
import glob
import json
import os
import re
import shutil


# Metrics where sklearn's convention is ``neg_*`` (lower raw value = better).
_LOWER_IS_BETTER_METRICS = {
    "root_mean_squared_error",
    "rmse",
    "mean_squared_error",
    "mse",
    "mean_absolute_error",
    "mae",
    "log_loss",
}

# MLE-STAR uses 1e9 as a sentinel penalty score for failed executions.
_FAILED_SCORE = 1e9


def _extract_metric(task_description: str) -> str | None:
    """Extract the metric name from the task description text."""
    m = re.search(r"#\s*Metric\s*\n+\s*([A-Za-z0-9_]+)", task_description)
    if m:
        return m.group(1).strip().lower()
    return None


def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _best_internal_score(state: dict) -> tuple[float | None, str]:
    """Pick the best internal MLE-STAR score and report where it came from.

    MLE-STAR reports several scores. Prefer the final submission score, then
    fall back to ensemble/refined/initial scores.
    """
    def _valid(val):
        return val is not None and val != _FAILED_SCORE

    # Final submission execution result.
    sub_result = state.get("submission_code_exec_result") or {}
    if sub_result.get("returncode") == 0 and _valid(sub_result.get("score")):
        return float(sub_result["score"]), "submission"

    for key in ("best_ensemble_score", "best_refined_score", "best_initial_score"):
        val = state.get(key)
        if _valid(val):
            return float(val), key

    return None, "none"


def _normalize_score(score: float, metric: str | None) -> float:
    """Flip sign so higher is always better, matching the extension's convention.

    The extension uses sklearn scorers like ``neg_root_mean_squared_error``,
    where higher-is-better. MLE-STAR's internal scores are raw. We flip only
    for metrics where lower raw values are better.
    """
    return -score if metric in _LOWER_IS_BETTER_METRICS else score


def _emit_result_json(
    state: dict,
    active_dir: str,
    dry_run: bool,
) -> str | None:
    """Write a uniform result.json for one MLE-STAR final_state.json."""
    task = state.get("task_name", os.path.basename(active_dir).removeprefix("mle-star-"))
    internal_score, source = _best_internal_score(state)
    metric = _extract_metric(state.get("task_description", ""))

    if internal_score is None:
        holdout = {"error": "no internal score found in final_state.json"}
    else:
        normalized_score = _normalize_score(internal_score, metric)
        scorer = f"neg_{metric}" if metric in _LOWER_IS_BETTER_METRICS else metric
        holdout = {"scorer": scorer, "score": normalized_score}

    result = {
        "method": "mlestar",
        "task": task,
        "model": state.get("agent_model"),
        "time_budget_s": state.get("exec_timeout"),
        "web_search": state.get("use_web_search"),
        "relational": True,
        "status": "failed" if "error" in holdout else "ok",
        "abort_reason": holdout.get("error") if "error" in holdout else None,
        "wall_clock_s": 0.0,
        "llm_calls": 0,
        "tokens": {
            "prompt": 0,
            "completion": 0,
            "total": state.get("total_tokens_spent"),
        },
        "tokens_by_agent": {},
        "holdout": holdout,
        "reused_spec": False,
        "budget": None,
        "leaderboard": None,
    }

    result_path = os.path.join(active_dir, "result.json")
    if not dry_run:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    status_flag = "error" if "error" in holdout else "ok"
    score_str = (
        f"score={holdout['score']:.4f} (source={source}, metric={metric})"
        if "score" in holdout
        else holdout.get("error", "?")
    )
    print(f"{task}: {status_flag} — {score_str}")
    return result_path


def convert_mlestar_results(
    in_dir: str = "results/openai",
    out_root: str = "results",
    dry_run: bool = False,
) -> list[str]:
    """Move/rename MLE-STAR run folders and emit uniform result.json files."""
    generated: list[str] = []

    if not os.path.isdir(in_dir):
        print(f"input directory not found: {in_dir}")
        return generated

    # 1. Process each task subdirectory that contains a final_state.json.
    for entry in sorted(os.listdir(in_dir)):
        src_dir = os.path.join(in_dir, entry)
        if not os.path.isdir(src_dir):
            continue

        final_state_path = os.path.join(src_dir, "final_state.json")
        if not os.path.exists(final_state_path):
            continue

        state = _load_json(final_state_path)
        if not state:
            print(f"warning: could not parse {final_state_path}, skipping")
            continue

        task = state.get("task_name", entry)
        dest_dir = os.path.join(out_root, f"mle-star-{task}")

        if not dry_run:
            os.makedirs(out_root, exist_ok=True)
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.move(src_dir, dest_dir)

        active_dir = dest_dir if not dry_run else src_dir
        result_path = _emit_result_json(state, active_dir, dry_run)
        if result_path:
            generated.append(result_path)

    # 2. Delete loose files left under in_dir (including hidden ones).
    loose_files = [
        os.path.join(in_dir, entry)
        for entry in os.listdir(in_dir)
        if os.path.isfile(os.path.join(in_dir, entry))
    ]
    for p in loose_files:
        if not dry_run:
            os.remove(p)
        print(f"removed loose file: {p}")

    # 3. Optionally remove the empty in_dir if it has no subdirectories left.
    if not dry_run:
        remaining = [
            p for p in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, p))
        ]
        if not remaining:
            os.rmdir(in_dir)
            print(f"removed empty directory: {in_dir}")

    return generated


def fixup_existing_mlestar_results(
    out_root: str = "results",
    dry_run: bool = False,
) -> list[str]:
    """Regenerate result.json for directories already renamed to mle-star-*."""
    generated: list[str] = []
    if not os.path.isdir(out_root):
        print(f"output directory not found: {out_root}")
        return generated

    for entry in sorted(os.listdir(out_root)):
        if not entry.startswith("mle-star-"):
            continue
        active_dir = os.path.join(out_root, entry)
        if not os.path.isdir(active_dir):
            continue

        final_state_path = os.path.join(active_dir, "final_state.json")
        if not os.path.exists(final_state_path):
            continue

        state = _load_json(final_state_path)
        if not state:
            print(f"warning: could not parse {final_state_path}, skipping")
            continue

        result_path = _emit_result_json(state, active_dir, dry_run)
        if result_path:
            generated.append(result_path)

    return generated


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MLE-STAR final_state.json artifacts to result.json."
    )
    parser.add_argument(
        "--in-dir",
        default="results/openai",
        help="directory containing raw MLE-STAR task folders",
    )
    parser.add_argument(
        "--out-root",
        default="results",
        help="destination root for renamed run folders",
    )
    parser.add_argument(
        "--fixup",
        action="store_true",
        help="only regenerate result.json in already-renamed mle-star-* folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be done without changing files",
    )
    args = parser.parse_args()

    if args.fixup:
        generated = fixup_existing_mlestar_results(
            out_root=args.out_root,
            dry_run=args.dry_run,
        )
    else:
        generated = convert_mlestar_results(
            in_dir=args.in_dir,
            out_root=args.out_root,
            dry_run=args.dry_run,
        )
    print(
        f"\n{'would generate' if args.dry_run else 'generated'} "
        f"{len(generated)} result.json file(s)"
    )


if __name__ == "__main__":
    _main()
