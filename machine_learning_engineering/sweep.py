"""Sweep harness — pure orchestration over ``pipeline.run_pipeline``.

Runs a hand-picked list of configurations from a JSON sweep spec, sequentially
and in-process, and aggregates one row per run into ``sweep.csv`` +
``sweep.md``. **Zero new LLM call sites**: the analyst/plan-author agents run
once per *task* (the raw spec is captured and reused across every point via
``run_pipeline(spec_raw=...)``); only refine runs add proposer calls
(``outer_steps - 1`` each). That keeps a realistic sweep — e.g. 27 runs per
task (a c-grid, a retarget A/B, a propose-dosage block, 3 seeds each) — at
~14 LLM calls per task, ~42/day for three tasks: comfortably inside the
gemini-2.5-flash free tier (~10 req/min, ~250 req/day). The binding
constraint is wall-clock CV time (~2-5 min per run at budget 15), not quota.

Spec format — ``defaults`` merged under every entry of ``runs``; list-valued
fields cartesian-expand *within* their entry; ``n_proposes: N`` is the CLI
sugar (``outer_steps = N + 1``, ``refine = N > 0``):

Example:
    {"defaults": {"task": "credit-fraud", "budget": 15, "seed": 42},
     "runs": [{"c": [0.3, 0.5, 0.7], "seed": [42, 43]},
              {"n_proposes": 3, "top_k": 3}]}
    # -> 7 runs, 2 agent calls (one spec fetch) + 3 proposer calls

    uv run python -m machine_learning_engineering.sweep sweeps/example.json
"""

import argparse
import csv
import itertools
import json
import os
import time
from datetime import datetime

from machine_learning_engineering import pipeline
from machine_learning_engineering.search_loop import make_llm_proposer

_DEFAULTS = {
    "task": None,
    "c": 0.5,
    "budget": 15,
    "outer_steps": 1,
    "refine": False,
    "retarget": True,
    "top_k": 1,
    "seed": 42,
}
_KEYS = set(_DEFAULTS) | {"n_proposes"}

_COLUMNS = [
    "task", "seed", "c", "budget", "outer_steps", "refine", "retarget",
    "top_k", "best_search_score", "report_scorer", "report_score",
    "ensemble_score", "used_fallback_spec", "reused_spec", "llm_calls",
    "wall_clock_s", "status", "error", "run_dir",
]


def load_sweep_spec(path: str) -> list[dict]:
    """Parse a JSON sweep spec into a flat list of complete run configs.

    Merges ``defaults`` under each ``runs`` entry, cartesian-expands
    list-valued fields within an entry, applies the ``n_proposes`` sugar, and
    rejects unrecognized keys (typos fail loudly, not silently).

    Example:
        load_sweep_spec("sweeps/example.json")
        # -> [{"task": "credit-fraud", "c": 0.3, "budget": 15, ...}, ...]
    """
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    defaults = spec.get("defaults", {})
    entries = spec.get("runs", [])
    for entry in [defaults, *entries]:
        unknown = set(entry) - _KEYS
        if unknown:
            raise ValueError(f"unknown sweep key(s): {sorted(unknown)}")
    configs = []
    for entry in entries:
        merged = {**_DEFAULTS, **defaults, **entry}
        list_keys = [k for k, v in merged.items() if isinstance(v, list)]
        for combo in itertools.product(*(merged[k] for k in list_keys)):
            cfg = {**merged, **dict(zip(list_keys, combo))}
            n_proposes = cfg.pop("n_proposes", None)
            if n_proposes is not None:  # sugar, same as the pipeline CLI
                cfg["outer_steps"] = n_proposes + 1
                cfg["refine"] = n_proposes > 0
            if not cfg["task"]:
                raise ValueError("every run needs a 'task' (entry or defaults)")
            configs.append(cfg)
    return configs


def slug(cfg: dict, with_task: bool = False) -> str:
    """A filesystem-safe, human-scannable name for one grid point.

    Example:
        slug({"c": 0.5, "budget": 15, "outer_steps": 3, "refine": True,
              "retarget": True, "top_k": 3, "seed": 42})
        # -> "c0.5_b15_os3_ref1_rt1_tk3_s42"
    """
    s = (
        f"c{cfg['c']}_b{cfg['budget']}_os{cfg['outer_steps']}"
        f"_ref{int(cfg['refine'])}_rt{int(cfg['retarget'])}"
        f"_tk{cfg['top_k']}_s{cfg['seed']}"
    )
    return f"{cfg['task']}_{s}" if with_task else s


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


def run_one(cfg: dict, spec_raw: str | None, out_dir: str, retry_wait: int = 65) -> dict:
    """Run one sweep point through ``run_pipeline`` and return a result row.

    Quota errors (429 / RESOURCE_EXHAUSTED) sleep ``retry_wait`` seconds and
    retry once; any still-failing run becomes a ``status="failed"`` row so the
    sweep continues. The full ``result.json``/``summary.md`` artifacts land in
    ``out_dir`` via the pipeline's own persistence.

    Example:
        run_one({"task": "credit-fraud", "c": 0.5, ...}, spec_raw, "runs/sweep/x")
        # -> {"task": "credit-fraud", "best_search_score": 0.71, "status": "ok", ...}
    """
    row = {k: cfg[k] for k in ("task", "seed", "c", "budget", "outer_steps",
                               "refine", "retarget", "top_k")}
    row.update({c: "" for c in _COLUMNS if c not in row})
    row["run_dir"] = out_dir
    started = time.time()
    for attempt in (1, 2):
        try:
            result = pipeline.run_pipeline(
                task_name=cfg["task"],
                budget=cfg["budget"],
                out_dir=out_dir,
                seed=cfg["seed"],
                outer_steps=cfg["outer_steps"],
                propose=make_llm_proposer() if cfg["refine"] else None,
                top_k=cfg["top_k"],
                c=cfg["c"],
                retarget=cfg["retarget"],
                spec_raw=spec_raw,
            )
            report = result.get("report") or {}
            ens = result.get("ensemble") or {}
            row.update(
                best_search_score=result["best_search_score"],
                report_scorer=report.get("scorer", ""),
                report_score=report.get("score", ""),
                ensemble_score=ens.get("ensemble_score", ""),
                used_fallback_spec=result["used_fallback_spec"],
                reused_spec=result["reused_spec"],
                llm_calls=result["llm_calls"],
                status="ok",
                spec_raw=result["spec_raw"],  # popped by the caller, not a column
            )
            break
        except Exception as exc:
            if attempt == 1 and _is_quota_error(exc):
                time.sleep(retry_wait)
                continue
            row.update(status="failed", error=repr(exc)[:200])
            break
    row["wall_clock_s"] = round(time.time() - started, 1)
    return row


def append_row(csv_path: str, row: dict) -> None:
    """Append one row to sweep.csv (header on first write) — crash-safe.

    Example:
        append_row("runs/sweep_x/sweep.csv", {"task": "credit-fraud", ...})
    """
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def write_markdown(rows: list[dict], spec_path: str, path: str) -> None:
    """Write sweep.md — spec echo, totals, and a score-sorted results table.

    Example:
        write_markdown(rows, "sweeps/example.json", "runs/sweep_x/sweep.md")
    """
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]
    ordered = sorted(
        ok, key=lambda r: float(r.get("best_search_score") or 0.0), reverse=True
    ) + failed
    total_llm = sum(int(r.get("llm_calls") or 0) for r in rows)
    total_wall = sum(float(r.get("wall_clock_s") or 0.0) for r in rows)
    lines = [
        f"# Sweep: {spec_path}",
        f"date: {datetime.now():%Y-%m-%d %H:%M}  |  runs: {len(rows)} "
        f"({len(failed)} failed)  |  LLM calls: {total_llm}  |  "
        f"wall-clock: {total_wall / 60:.1f} min",
        "",
        "| " + " | ".join(_COLUMNS[:-2]) + " |",
        "|" + "---|" * len(_COLUMNS[:-2]),
    ]
    for r in ordered:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in _COLUMNS[:-2]) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Sweep over pipeline configs.")
    parser.add_argument("spec", help="JSON sweep spec (defaults + runs)")
    parser.add_argument("--out", default=None,
                        help="sweep dir (default: runs/sweep_<timestamp>)")
    parser.add_argument("--retry-wait", type=int, default=65,
                        help="seconds to wait before the one quota retry")
    parser.add_argument("--no-reuse-spec", action="store_true",
                        help="run the agents for every point (A/B plan variance)")
    parser.add_argument("--spec-cache", default=None,
                        help="JSON {task: raw_spec}; read if present, updated "
                        "after each fetch — shares agent calls across sweeps")
    args = parser.parse_args()

    configs = load_sweep_spec(args.spec)
    sweep_dir = args.out or os.path.join(
        "runs", f"sweep_{datetime.now():%Y%m%d-%H%M}"
    )
    os.makedirs(sweep_dir, exist_ok=True)
    csv_path = os.path.join(sweep_dir, "sweep.csv")

    spec_cache: dict = {}
    if args.spec_cache and os.path.exists(args.spec_cache):
        with open(args.spec_cache, encoding="utf-8") as f:
            spec_cache = json.load(f)

    tasks = list(dict.fromkeys(c["task"] for c in configs))
    many_tasks = len(tasks) > 1
    rows, seen = [], set()
    for task in tasks:  # task-outermost so each spec fetch serves its block
        for cfg in (c for c in configs if c["task"] == task):
            name = slug(cfg, with_task=many_tasks)
            while name in seen:  # repeated grid point -> disambiguate
                name += "_x"
            seen.add(name)
            reuse = None if args.no_reuse_spec else spec_cache.get(task)
            row = run_one(
                cfg, reuse, os.path.join(sweep_dir, name), args.retry_wait
            )
            fetched = row.pop("spec_raw", None)
            if fetched and task not in spec_cache:
                spec_cache[task] = fetched
                if args.spec_cache:
                    with open(args.spec_cache, "w", encoding="utf-8") as f:
                        json.dump(spec_cache, f, indent=2, ensure_ascii=False)
            append_row(csv_path, row)
            rows.append(row)
            print(
                f"[{len(rows)}/{len(configs)}] {name}: "
                f"{row.get('status')} score={row.get('best_search_score')}"
            )
    write_markdown(rows, args.spec, os.path.join(sweep_dir, "sweep.md"))
    print(f"\nSweep artifacts: {sweep_dir}/ (sweep.csv, sweep.md, <slug>/)")


if __name__ == "__main__":
    _main()
