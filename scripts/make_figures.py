"""Comparison figures + tables from uniform run artifacts.

Reads the ``result.json`` files the three runners emit (the extension via
``pipeline.run_pipeline``, AutoGluon via ``scripts/run_autogluon.py``, MLE-STAR
via ``scripts/run_mlestar.py``) — all share the fields ``method``, ``task``,
``holdout: {scorer, score}``, ``tokens: {total, ...}``, ``llm_calls``,
``wall_clock_s``, ``time_budget_s`` — and produces:

1. **quality-at-cost** — per task, each method's shared-holdout score against the
   tokens it spent (the extension clusters at a small constant; AutoGluon at 0;
   MLE-STAR far right). Bar-of-quality + token annotation, one panel per task.
2. **time-scaling** — the extension's holdout score vs its wall-clock/budget at
   CONSTANT token cost (the headline claim), read from extension result.json
   artifacts whose budget varies (several `make run-live BUDGET=...` runs
   under one `--runs` root).
3. **AutoGluon exploration progress** — a companion efficiency curve reconstructed
   from a SINGLE `best_quality` AutoGluon run: its own leaderboard already carries
   `fit_order` + `fit_time_marginal` (per-model training cost) + `score_val`
   (AutoGluon's internal validation score), so a running-best-score-vs-cumulative-
   fit-time trace needs no budget sweep, unlike the extension's curve above. Axis
   units and score source both differ from the extension curve (seconds vs
   rollouts; internal validation vs shared external holdout) — read side by side
   as two methods' own efficiency stories, not one merged curve. MLE-STAR is not
   included yet (no comparable per-step trace has been captured for it).
4. **mechanism table** — a static qualitative comparison (mechanism / debug cost
   / token cost / leakage handling / adaptivity), written as markdown.

Run:
    uv run python scripts/make_figures.py --runs results --out results/figures
Needs matplotlib (the `bench` extra):  uv sync --extra bench

Reads from ``results/`` (a small, git-shareable mirror of ``runs/`` containing
only each run's ``result.json`` — see ``scripts/collect_results.py``), not
``runs/`` itself, which also holds multi-GB AutoGluon model artifacts.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

# The 10 tasks we're actually comparing methods on for the writeup. Runs for
# other staged tasks (california-housing-prices, employee-salaries,
# midwest-survey, ...) may exist under `runs/` but are deliberately excluded
# from every figure/table below — filtered by this allowlist rather than by
# scanning whatever happens to be in the directory.
_TASK_ALLOWLIST = {
    "country-happiness",
    "toxicity",
    "credit-fraud",
    "traffic-violations",
    "movielens",
    "open-payments",
    "videogame-sales",
    "medical-charge",
    "flight-delays",
    "bike-sharing",
}


def load_results(roots: list[str]) -> list[dict]:
    """Collect uniform records from every result.json under the given roots.

    Only files that carry a ``method`` + ``holdout`` block, AND whose ``task``
    is in ``_TASK_ALLOWLIST``, are kept (so partial/legacy result.json and any
    task outside the current comparison set are both skipped). Returns a flat
    list of ``{method, task, score, scorer, tokens, llm_calls, wall_clock_s,
    time_budget_s, relational, leaderboard}`` dicts.
    """
    records: list[dict] = []
    for root in roots:
        for path in sorted(
            glob.glob(os.path.join(root, "**", "result.json"), recursive=True)
        ):
            try:
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
            except (OSError, ValueError):
                continue
            hold = d.get("holdout")
            if not d.get("method") or not hold or hold.get("score") is None:
                continue
            task = d.get("task", "?")
            if task not in _TASK_ALLOWLIST:
                continue
            records.append(
                {
                    "method": d["method"],
                    "task": task,
                    "score": float(hold["score"]),
                    "scorer": hold.get("scorer", ""),
                    "tokens": int((d.get("tokens") or {}).get("total", 0)),
                    "llm_calls": int(d.get("llm_calls", 0)),
                    "wall_clock_s": float(d.get("wall_clock_s") or 0.0),
                    "time_budget_s": d.get("time_budget_s"),
                    "budget": d.get("budget"),
                    "relational": bool(d.get("relational", False)),
                    "leaderboard": d.get("leaderboard"),
                    "path": path,
                }
            )
    return records


_METHOD_COLOR = {
    "extension": "#2b8cbe",
    "autogluon": "#31a354",
    "mlestar": "#e6550d",
}


def fig_quality_at_cost(records: list[dict], out_dir: str) -> str | None:
    """One panel per task: holdout score per method, annotated with token cost.

    Multiple runs of the SAME method on the same task (several
    `make run-live BUDGET=...` extension runs, provider/rt-on-off variants,
    repeated AutoGluon attempts, ...) are aggregated into ONE bar per method —
    mean score, error bars = min-max range across those runs — rather than
    plotting every historical run as its own bar. Bar charts key bars by their
    x-axis label, so repeated identical method labels used to draw multiple
    bars on top of each other with overlapping, unreadable annotation text.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    if not by_task:
        return None

    tasks = sorted(by_task)
    n = len(tasks)
    fig, axes = plt.subplots(
        1, n, figsize=(max(4, 3.2 * n), 3.6), squeeze=False
    )
    for ax, task in zip(axes[0], tasks):
        by_method = defaultdict(list)
        for r in by_task[task]:
            by_method[r["method"]].append(r)
        methods = sorted(by_method)

        means, err_lo, err_hi, tok_means, n_runs = [], [], [], [], []
        scorer_label = None
        for m in methods:
            rows = by_method[m]
            scores = [r["score"] for r in rows]
            mean = sum(scores) / len(scores)
            means.append(mean)
            err_lo.append(mean - min(scores))
            err_hi.append(max(scores) - mean)
            tok_means.append(sum(r["tokens"] for r in rows) / len(rows))
            n_runs.append(len(rows))
            scorer_label = scorer_label or rows[0]["scorer"]

        colors = [_METHOD_COLOR.get(m, "#888") for m in methods]
        bars = ax.bar(
            methods, means, color=colors, yerr=[err_lo, err_hi], capsize=3
        )
        for bar, mean, tok, n_run in zip(bars, means, tok_means, n_runs):
            tok_str = f"{tok:,.0f} tok" if tok else "0 tok"
            n_str = f" (n={n_run})" if n_run > 1 else ""
            ax.annotate(
                f"{mean:.3f}{n_str}\n{tok_str}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_title(task, fontsize=9)
        ax.set_ylabel(scorer_label or "holdout score", fontsize=8)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
    fig.suptitle(
        "Quality at cost — shared-holdout score vs token spend\n"
        "(bars = mean over repeated runs; error bars = min-max range)",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "quality_at_cost.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_time_scaling(records: list[dict], out_dir: str) -> str | None:
    """Extension holdout score vs budget/time at CONSTANT token cost.

    Plots extension-method records whose `time_budget_s` (or `budget`) varies
    — e.g. several `make run-live` result.json artifacts at different budgets
    under one `--runs` root — one quality curve per task, plus a token panel
    confirming LLM cost stays flat. This is the headline differentiator.
    Tasks with fewer than two distinct budget points are skipped.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _x(r):
        if r.get("time_budget_s") is not None:
            return float(r["time_budget_s"]), "wall-clock budget (s)"
        return float(r.get("budget") or 0), "search budget (rollouts)"

    by_task = defaultdict(list)
    xlabel = "search budget (rollouts)"
    for r in records:
        if r["method"] != "extension":
            continue
        x, xlabel = _x(r)
        by_task[r["task"]].append((x, r["score"], r["tokens"]))
    by_task = {
        task: pts
        for task, pts in by_task.items()
        if len({p[0] for p in pts}) >= 2
    }

    if not by_task:
        return None
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for task, pts in sorted(by_task.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        toks = [p[2] for p in pts]
        ax.plot(xs, ys, marker="o", label=task)
        ax2.plot(xs, toks, marker="s", label=task)
    ax.set_ylabel("shared-holdout score (higher better)")
    ax.set_title(
        "Time scaling — quality rises with budget at constant LLM cost"
    )
    ax.legend(fontsize=8)
    ax2.set_ylabel("tokens")
    ax2.set_xlabel(xlabel)
    ax2.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(out_dir, "time_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _autogluon_progress_curve(
    leaderboard: list[dict],
) -> list[tuple[float, float]]:
    """Reconstruct a running-best-score-vs-cumulative-fit-time trace.

    AutoGluon's own leaderboard already carries everything needed: `fit_order`
    (the sequence models were tried in), `fit_time_marginal` (each model's own
    training cost), and `score_val` (AutoGluon's internal validation score) —
    no extra instrumentation or budget sweep required, unlike the extension's
    time-scaling curve above. Two caveats this trace does NOT resolve: (1)
    `score_val` is AutoGluon's own internal validation split, not the shared
    external holdout; (2) this approximates "if this same run had stopped
    early," not "if a smaller time_limit had been given from the start" — a
    genuinely smaller budget can make categorically different model-selection
    choices rather than just truncate this timeline.
    """
    rows = sorted(leaderboard, key=lambda r: r.get("fit_order", 0))
    cum = 0.0
    best = float("-inf")
    points: list[tuple[float, float]] = []
    for row in rows:
        score_val = row.get("score_val")
        if score_val is None:
            continue
        cum += float(row.get("fit_time_marginal") or 0.0)
        best = max(best, float(score_val))
        points.append((cum, best))
    return points


def fig_autogluon_progress(records: list[dict], out_dir: str) -> str | None:
    """AutoGluon's own efficiency curve: running-best internal score vs cumulative fit time.

    Companion to `fig_time_scaling`, reconstructed from a SINGLE `best_quality`
    run per task (see `_autogluon_progress_curve`) rather than a budget sweep.
    Read the two figures side by side, not merged — the x-axis units (seconds
    vs rollouts) and y-axis score source (internal validation vs shared
    external holdout) both differ. MLE-STAR is intentionally excluded until a
    comparable per-step trace exists for it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_task = {}
    for r in records:
        if r["method"] != "autogluon" or not r.get("leaderboard"):
            continue
        pts = _autogluon_progress_curve(r["leaderboard"])
        if pts:
            by_task[r["task"]] = pts

    if not by_task:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    for task, pts in sorted(by_task.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=".", markersize=4, linestyle="--", label=task)
    ax.set_xlabel("cumulative model fit time (s) — reconstructed, single run")
    ax.set_ylabel(
        "AutoGluon internal validation score\n(NOT the shared external holdout)"
    )
    ax.set_title("AutoGluon exploration progress (best_quality, single run)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "autogluon_progress.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


_MECHANISM_ROWS = [
    (
        "Search mechanism",
        "MCTS over a skrub-DataOps space (structure + HPs)",
        "Ensemble/stacking of many pretrained model configs",
        "LLM writes + iteratively refines Python code",
    ),
    (
        "LLM call count",
        "O(1) per task (2 + N_PROPOSES), fixed & known up front",
        "0 (no LLM)",
        "Unbounded: ~26 best case, 1000+ with debug cascade",
    ),
    (
        "Cost scales with",
        "CV rollouts / wall-clock (pure code) — LLM cost constant",
        "Wall-clock (pure code)",
        "LLM calls (data-dependent debug retries)",
    ),
    (
        "Leakage handling",
        "skrub DataOps: transforms fit inside CV folds by construction",
        "AutoGluon internal CV / bagging",
        "Depends on the code the LLM generates (no guarantee)",
    ),
    (
        "Adaptivity",
        "Space authored once; search adapts within it (Option 1/3)",
        "Fixed AutoML pipeline",
        "Fully adaptive per step (at unbounded token cost)",
    ),
    (
        "Relational tables",
        "Yes — AggJoiner over aux_*.csv is a searchable stage",
        "No — flat table only",
        "Possible if the LLM writes the join",
    ),
]


def mechanism_table(out_dir: str) -> str:
    """Write the static qualitative mechanism-comparison table as markdown."""
    header = (
        "| Axis | MCTS-skrub (ours) | AutoGluon | MLE-STAR |\n"
        "|---|---|---|---|\n"
    )
    body = "".join(
        f"| {axis} | {ours} | {ag} | {ms} |\n"
        for axis, ours, ag, ms in _MECHANISM_ROWS
    )
    path = os.path.join(out_dir, "mechanism_table.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Mechanism comparison\n\n" + header + body)
    return path


def write_comparison_csv(records: list[dict], out_dir: str) -> str:
    """Flat CSV of every collected record (one row per method x task run)."""
    path = os.path.join(out_dir, "comparison.csv")
    cols = [
        "task",
        "method",
        "scorer",
        "score",
        "tokens",
        "llm_calls",
        "wall_clock_s",
        "time_budget_s",
        "relational",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda r: (r["task"], r["method"])):
            w.writerow(r)
    return path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark figures + tables.")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["results"],
        help="root dir(s) to scan for uniform result.json artifacts",
    )
    parser.add_argument("--out", default="results/figures", help="output dir")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records = load_results(args.runs)
    print(f"collected {len(records)} benchmark records from {args.runs}")

    made = [
        write_comparison_csv(records, args.out),
        mechanism_table(args.out),
    ]
    q = fig_quality_at_cost(records, args.out)
    if q:
        made.append(q)
    t = fig_time_scaling(records, args.out)
    if t:
        made.append(t)
    ag = fig_autogluon_progress(records, args.out)
    if ag:
        made.append(ag)
    print("wrote:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    _main()
