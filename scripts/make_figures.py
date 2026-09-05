"""Comparison figures + tables from uniform run artifacts.

Reads the ``result.json`` files the three runners emit (the extension via
``pipeline.run_pipeline``, AutoGluon via ``scripts/run_autogluon.py``, MLE-STAR
via ``scripts/run_mlestar.py``) — all share the fields ``method``, ``task``,
``holdout: {scorer, score}``, ``tokens: {total, ...}``, ``llm_calls``,
``wall_clock_s``, ``time_budget_s`` — and produces:

1. **quality-at-cost** — per task, each method's shared-holdout score against the
   tokens it spent (the extension clusters at a small constant; AutoGluon at 0;
   MLE-STAR far right). Bar-of-quality + token annotation, one panel per task,
   laid out as a `_QC_NCOLS`-wide grid (5x2 for the 10 comparison tasks) rather
   than a single horizontal strip. All three methods draw automatically from
   whatever `result.json` files are present — an MLE-STAR run appears the moment
   `make bench-mlestar` produces one (see `fig_quality_at_cost`).
2. **proposal scaling** — the extension's holdout score vs Optional Feature 3 call count
   (`n_proposes`), one subplot per task, minimal by design (just task/n_proposes/
   score — see `fig_proposal_scaling`). ONLY `scripts/replay_from_run.py`
   replays count (`reused_spec=True`) — a live run only ever produces one
   point, at whatever n_proposes it happened to run at, so mixing it with the
   deliberately isolated replay curve would compare a different measurement to
   a controlled one.
3. **token cost** — the real-LLM-token cost side of that same n_proposes axis
   (see `fig_token_cost`), the inverse filter: ONLY live runs (`reused_spec=
   False`), since a replay costs zero new tokens by construction. Every task is
   drawn on ONE shared axes (a labelled line each), so the whole comparison set's
   token cost reads at a glance. Endpoints (`n=0` and the run's own max
   n_proposes) are exact from logged per-agent totals; any point between is an
   even-split estimate (no per-call proposer token log exists, only an aggregate
   across all propose() calls in the run).
4. **mechanism table** — a static qualitative comparison (mechanism / debug cost
   / token cost / leakage handling / adaptivity), written as markdown.

Run:
    uv run python scripts/make_figures.py --runs results --out figures
Needs matplotlib (the `bench` extra):  uv sync --extra bench

Reads from ``results/`` (the git-shareable mirror of ``runs/`` — see
``scripts/collect_results.py``), not ``runs/`` itself, which also holds
multi-GB AutoGluon model artifacts.

Runs from before the on-disk split (commit ``517f954``) are scored on a bench
that no longer exists. ``_on_current_bench`` flags each record so the figures
can keep the two apart instead of averaging across them.
"""

import argparse
import csv
import glob
import json
import os
import re
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


# The on-disk train/holdout split landed 2026-07-14 17:13 (commit 517f954);
# out-of-fold ensemble selection and its `selection` stamp landed 43 minutes
# later (dbc0f77). Runs before that are scored on a bench that no longer
# exists, so they must never be averaged together with later ones.
_SPLIT_FIX_DAY = "20260715"


def _on_current_bench(result: dict, path: str) -> bool:
    """True if this run was scored on the on-disk split (post-`517f954`).

    Two tests, because neither alone is sufficient: the `oof_3fold` stamp is
    conclusive, but a run with a single ensemble member has no `ensemble` block
    to carry it — those are identified by the timestamp in their directory
    name instead. Pre-fix runs have no `selection` key at all (the stamping
    code shipped after them), so absence proves nothing on its own.

    Example:
        _on_current_bench({...}, "results/replay_x_np2_20260715-1325/result.json") -> True
    """
    ensemble = result.get("ensemble")
    if isinstance(ensemble, dict) and ensemble.get("selection") == "oof_3fold":
        return True
    stamps = re.findall(r"20\d{6}", os.path.basename(os.path.dirname(path)))
    return bool(stamps) and stamps[-1] >= _SPLIT_FIX_DAY


def load_results(roots: list[str]) -> list[dict]:
    """Collect uniform records from every result.json under the given roots.

    Only files that carry a ``method`` + ``holdout`` block, AND whose ``task``
    is in ``_TASK_ALLOWLIST``, are kept (so partial/legacy result.json and any
    task outside the current comparison set are both skipped). Returns a flat
    list of ``{method, task, score, scorer, tokens, llm_calls, wall_clock_s,
    time_budget_s, relational, leaderboard, reused_spec, tokens_by_agent}``
    dicts.
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
                    "relational": bool(d.get("relational", False)),
                    "leaderboard": d.get("leaderboard"),
                    "reused_spec": bool(d.get("reused_spec", False)),
                    "on_current_bench": _on_current_bench(d, path),
                    "tokens_by_agent": d.get("tokens_by_agent") or {},
                    "path": path,
                }
            )
    return records


_METHOD_COLOR = {
    "extension": "#2b8cbe",
    "autogluon": "#31a354",
    "mlestar": "#e6550d",
}

# quality-at-cost grid width. 10 comparison tasks -> 5 columns x 2 rows.
# Flip this to 2 for a portrait 2x5 layout.
_QC_NCOLS = 5


def _grid(n: int, ncols: int) -> tuple[int, int]:
    """(nrows, ncols) grid that fits n panels, ncols-wide (last row may be short).

    Example:
        _grid(10, 5)  # -> (2, 5)
        _grid(7, 5)   # -> (2, 5)  (three cells left empty)
    """
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)  # ceil
    return nrows, ncols


def fig_quality_at_cost(records: list[dict], out_dir: str) -> str | None:
    """One panel per task: holdout score per method, annotated with token cost.

    Bars are keyed by (method, bench), NOT by method alone: a pre-fix run and a
    current-bench run are scored on different holdouts, so averaging them
    produces a number measured on no bench at all. The extension therefore gets
    up to two bars per task ("extension" and "extension (clean)"); the baselines
    have only pre-fix runs, so they keep one each.

    Within a (method, bench) group, repeated runs — several
    `make run-live BUDGET=...` attempts, repeated AutoGluon runs — are still
    aggregated into ONE bar: mean score, error bars = min-max range, `n=` in the
    annotation. Bar charts key bars by their x label, so repeated identical
    labels would otherwise stack bars on top of each other with overlapping,
    unreadable text.
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
    nrows, ncols = _grid(n, _QC_NCOLS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.9 * nrows),
        squeeze=False,
    )
    flat_axes = [ax for row in axes for ax in row]
    for ax, task in zip(flat_axes, tasks):
        by_group = defaultdict(list)
        for r in by_task[task]:
            by_group[(r["method"], r["on_current_bench"])].append(r)
        # method order, pre-fix bar before its clean counterpart
        groups = sorted(by_group, key=lambda g: (g[0], g[1]))

        labels, means, err_lo, err_hi, tok_means, n_runs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        scorer_label = None
        for method, clean in groups:
            rows = by_group[(method, clean)]
            scores = [r["score"] for r in rows]
            mean = sum(scores) / len(scores)
            labels.append(f"{method}\n(clean)" if clean else method)
            means.append(mean)
            err_lo.append(mean - min(scores))
            err_hi.append(max(scores) - mean)
            tok_means.append(sum(r["tokens"] for r in rows) / len(rows))
            n_runs.append(len(rows))
            scorer_label = scorer_label or rows[0]["scorer"]

        # Method colour for every bar; the current-bench bar is hatched so it
        # reads as the same method on a different bench, not a fourth method.
        colors = [_METHOD_COLOR.get(m, "#888") for m, _ in groups]
        hatches = ["//" if clean else "" for _, clean in groups]
        bars = ax.bar(
            labels,
            means,
            color=colors,
            yerr=[err_lo, err_hi],
            capsize=3,
            hatch=hatches,
            edgecolor="white",
        )
        for bar, mean, tok, n_run, hi in zip(
            bars, means, tok_means, n_runs, err_hi
        ):
            tok_str = f"{tok:,.0f} tok" if tok else "0 tok"
            n_str = f" (n={n_run})" if n_run > 1 else ""
            # anchor above the error-bar cap on positive bars, below it on
            # negative ones, so the label never sits on top of the bar itself
            top = bar.get_height() >= 0
            ax.annotate(
                f"{mean:.3f}{n_str}\n{tok_str}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4 if top else -4),
                ha="center",
                va="bottom" if top else "top",
                fontsize=7,
            )
        # headroom so the two-line annotations clear the axes frame (the
        # collision the old tight ylim caused). ~32% above, ~12% below.
        lo, hi = ax.get_ylim()
        span = (hi - lo) or 1.0
        ax.set_ylim(lo - 0.12 * span, hi + 0.32 * span)
        ax.set_title(task, fontsize=9)
        ax.set_ylabel(scorer_label or "holdout score", fontsize=8)
        ax.tick_params(axis="x", labelsize=8, rotation=20)

    for ax in flat_axes[n:]:  # hide the empty cells in the last row
        ax.set_visible(False)

    fig.suptitle(
        "Quality at cost — holdout score vs token spend\n"
        "(bars = mean over repeated runs, error bars = min-max range; "
        "hatched = current bench, plain = pre-fix — not comparable)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=1.6, h_pad=2.4)
    path = os.path.join(out_dir, "quality_at_cost.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _n_proposes(r: dict) -> int:
    """Recover Optional Feature 3's call count from a record, live run or replay alike.

    Not stored directly (see docs/PROJECT_STATE.md's owed item: `llm_calls`
    miscounts offline replay proposers as real calls) — but that's exactly
    what makes it recoverable: `llm_calls` counts propose() calls regardless
    of whether they were real. A live run also pays 2 calls for
    data_analyst + plan_author; a `reused_spec` run (any `scripts/
    replay_from_run.py` replay, or the offline Claude driver) skips both
    agents, so its `llm_calls` IS `n_proposes` directly.
    """
    return r["llm_calls"] if r["reused_spec"] else max(0, r["llm_calls"] - 2)


def fig_proposal_scaling(records: list[dict], out_dir: str) -> str | None:
    """Extension holdout score vs Optional Feature 3 proposal count, one subplot per task.

    Deliberately minimal per-point: just task, n_proposes (`_n_proposes`), and
    score — no token/wall-clock panel, since the LLM-call axis IS n_proposes
    here, not a separate thing to also show (see `fig_token_cost` for the
    token side of this same axis).

    ONLY `scripts/replay_from_run.py` replays count (`reused_spec=True`) — a
    live run only ever produces ONE point, at whatever n_proposes it happened
    to be run at; it doesn't pause mid-search to also report what the score
    would have been at a lower n_proposes. Mixing that single live endpoint in
    with the replay-derived curve is comparing a different measurement to a
    deliberately isolated one. A task whose live run hasn't been replayed at
    every n_proposes yet will show fewer points here until it is (see
    `replay_from_run`'s own `n_proposes` ceiling — it can only replay up to
    however many real proposals the source run logged).

    Pre-fix and current-bench runs are drawn as SEPARATE series, never
    averaged together: they are scored on different holdouts, so a mean across
    them is a number measured on no bench at all. (Before this split, the
    country-happiness n=2 marker read -660 — the midpoint of a pre-fix -753.87
    and a clean -568.70.) Every clean replay so far sits at n=2, so the clean
    series is currently a lone marker per task rather than a curve; that is the
    honest picture until the missing n=0/n=1 replays are run.

    Points at the same (task, n_proposes, bench) — e.g. two replays sourced
    from different live runs that both used n_proposes=1 — are averaged. A
    series with fewer than two distinct n_proposes points is drawn as a marker
    with no line; a task with no series at all is skipped.

    One subplot per task, own y-axis each — scorers differ in scale AND sign
    across tasks (e.g. negative RMSE vs 0-1 accuracy/roc_auc), so a single
    shared axis would flatten every non-RMSE task into an invisible sliver.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_task_n = defaultdict(list)
    scorer_of = {}
    for r in records:
        if r["method"] != "extension" or not r["reused_spec"]:
            continue
        key = (r["task"], r["on_current_bench"], _n_proposes(r))
        by_task_n[key].append(r["score"])
        scorer_of[r["task"]] = r["scorer"]

    # task -> {on_current_bench: [(n, mean score), ...]}
    by_task = defaultdict(lambda: defaultdict(list))
    for (task, clean, n), scores in by_task_n.items():
        by_task[task][clean].append((n, sum(scores) / len(scores)))

    if not by_task:
        return None

    tasks = sorted(by_task)
    n = len(tasks)
    # One shared set of integer x-ticks for every panel. Left to the default
    # locator, a panel holding a single point (a task with only a clean n=2
    # replay) gets a degenerate x-range and renders unreadable fractional
    # ticks around it.
    all_ns = sorted({x for _, _, x in by_task_n})
    fig, axes = plt.subplots(
        1, n, figsize=(max(4, 3.2 * n), 3.8), squeeze=False
    )
    series = [
        (False, "pre-fix bench", "#9e9e9e", "o", "--"),
        (
            True,
            "current bench",
            _METHOD_COLOR.get("extension", "#2b8cbe"),
            "D",
            "-",
        ),
    ]
    for ax, task in zip(axes[0], tasks):
        for clean, label, color, marker, style in series:
            pts = sorted(by_task[task].get(clean, []))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs,
                ys,
                marker=marker,
                color=color,
                linestyle=style if len(pts) > 1 else "none",
                label=label,
                zorder=3 if clean else 2,
            )
        ax.set_title(task, fontsize=9)
        ax.set_ylabel(scorer_of.get(task, "holdout score"), fontsize=8)
        ax.set_xlabel("n_proposes", fontsize=8)
        ax.set_xticks(all_ns)
        ax.set_xlim(min(all_ns) - 0.35, max(all_ns) + 0.35)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0]:
        h, ls = ax.get_legend_handles_labels()
        for handle, lab in zip(h, ls):
            if lab not in labels:
                handles.append(handle)
                labels.append(lab)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Proposal scaling — quality vs Optional Feature 3 call count\n"
        "(benches shown separately — scores across them are not comparable)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(out_dir, "proposal_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _cumulative_tokens_by_n_proposes(r: dict) -> list[tuple[int, int]]:
    """Reconstruct a (n_proposes, cumulative tokens) curve from ONE live run's
    own `tokens_by_agent` breakdown.

    `n=0`'s cost (data_analyst + plan_author only, before any proposal) and
    the final point (that full total, matching `tokens.total` exactly) are
    EXACT — both come straight from logged per-agent totals. Any point in
    BETWEEN is an approximation: `tokens_by_agent["proposer"]` aggregates
    every propose() call in the run together (there's no per-call token log
    the way there's a per-call *proposal* log — `proposer_<k>_response.json`
    carries the proposal dict but not its own token count), so an
    intermediate n_proposes assumes the aggregate proposer cost split evenly
    across its calls. Only real runs have any of this — a
    `scripts/replay_from_run.py` replay's proposer cost is always 0 by
    construction, so this is meaningless for replays (see `fig_token_cost`'s
    `reused_spec` filter, the opposite of `fig_proposal_scaling`'s).

    Returns points sorted by n_proposes, e.g. ``[(0, 19917), (1, 31720),
    (2, 43523)]`` — the two ends exact, the middle an even-split estimate.
    """
    tba = r.get("tokens_by_agent") or {}
    n0 = tba.get("data_analyst", {}).get("total", 0) + tba.get(
        "plan_author", {}
    ).get("total", 0)
    proposer = tba.get("proposer") or {}
    calls = int(proposer.get("calls", 0))
    total = int(proposer.get("total", 0))
    if calls == 0:
        return [(0, n0)]
    per_call = total / calls
    return [(k, round(n0 + per_call * k)) for k in range(calls + 1)]


def fig_token_cost(records: list[dict], out_dir: str) -> str | None:
    """Real LLM token cost vs Optional Feature 3 proposal count — EVERY task on one axes,
    the cost-side companion to `fig_proposal_scaling`'s quality-side curve.

    ONLY live runs count (`reused_spec=False`) — the opposite filter from
    `fig_proposal_scaling`. A replay's whole point is that it costs zero new
    tokens (`make_fixed_sequence_proposer` never calls a real LLM), so it has
    nothing to show on a token axis; only a real run's logged
    `tokens_by_agent` breakdown has anything to plot (see
    `_cumulative_tokens_by_n_proposes` for exactly what's exact vs
    approximated within one run).

    One line per task on a single shared axes (labelled, in the legend) — the
    y-unit is tokens for every task, so they ARE directly comparable, and one
    plot shows the whole set's near-flat, low-constant cost at a glance (the
    headline: token cost barely moves with n_proposes). One live run per task
    (its own agent-authored plan generates this specific trace); points aren't
    averaged across runs, which would blur the "where does the token budget
    actually go" story.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_task = {}
    for r in records:
        if r["method"] != "extension" or r["reused_spec"]:
            continue
        pts = _cumulative_tokens_by_n_proposes(r)
        if len(pts) >= 2 and r["task"] not in by_task:
            by_task[r["task"]] = pts

    if not by_task:
        return None

    tasks = sorted(by_task)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, task in enumerate(tasks):
        pts = by_task[task]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=cmap(i % 10), label=task)
    ax.set_xlabel("n_proposes", fontsize=10)
    ax.set_ylabel(
        "cumulative tokens (exact at ends, interpolated between)", fontsize=10
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.margins(x=0.05, y=0.08)
    ax.legend(fontsize=8, title="task", ncol=2, loc="upper left")
    ax.set_title(
        "Token cost vs Optional Feature 3 call count (real live runs, all tasks)",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "token_cost.png")
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
        "Space authored once; search adapts within it (Optional Feature 1/3)",
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
    parser.add_argument("--out", default="figures", help="output dir")
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
    ps = fig_proposal_scaling(records, args.out)
    if ps:
        made.append(ps)
    tc = fig_token_cost(records, args.out)
    if tc:
        made.append(tc)
    print("wrote:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    _main()
