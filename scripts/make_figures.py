"""Comparison figures + tables from uniform run artifacts.

Reads the ``result.json`` files the three runners emit (the extension via
``pipeline.run_pipeline``, AutoGluon via ``scripts/run_autogluon.py``, MLE-STAR
via ``scripts/run_mlestar.py``) — all share the fields ``method``, ``task``,
``holdout: {scorer, score}``, ``tokens: {total, ...}``, ``llm_calls``,
``wall_clock_s``, ``time_budget_s`` — and produces:

1. **quality-at-cost** — per task, each method's shared-holdout score against the
   tokens it spent (the extension clusters at a small constant; AutoGluon at 0;
   MLE-STAR far right). Bar-of-quality + token annotation, one panel per task.
2. **AutoGluon exploration progress** — a companion efficiency curve reconstructed
   from a SINGLE `best_quality` AutoGluon run: its own leaderboard already carries
   `fit_order` + `fit_time_marginal` (per-model training cost) + `score_val`
   (AutoGluon's internal validation score), so a running-best-score-vs-cumulative-
   fit-time trace needs no budget sweep, unlike the extension's curve below. Axis
   units and score source both differ (seconds vs rollouts; internal validation
   vs shared external holdout) — read side by side as two methods' own
   efficiency stories, not one merged curve. MLE-STAR is not included yet (no
   comparable per-step trace has been captured for it).
3. **proposal scaling** — the extension's holdout score vs Option-3 call count
   (`n_proposes`), one subplot per task, minimal by design (just task/n_proposes/
   score — see `fig_proposal_scaling`). ONLY `scripts/replay_from_run.py`
   replays count (`reused_spec=True`) — a live run only ever produces one
   point, at whatever n_proposes it happened to run at, so mixing it with the
   deliberately isolated replay curve would compare a different measurement to
   a controlled one.
4. **token cost** — the real-LLM-token cost side of that same n_proposes axis
   (see `fig_token_cost`), the inverse filter: ONLY live runs (`reused_spec=
   False`), since a replay costs zero new tokens by construction. Endpoints
   (`n=0` and the run's own max n_proposes) are exact from logged per-agent
   totals; any point between is an even-split estimate (no per-call proposer
   token log exists, only an aggregate across all propose() calls in the run).
5. **budget scaling** — the extension's holdout score vs rollout budget at
   FIXED n_proposes (see `fig_budget_scaling`), replacing the old
   live-run-based time-scaling figure. Reads `scripts/replay_from_run.py
   --budget-sweep` artifacts (`results/sweep_<task>_np<n>_<ts>/budget_<B>/
   result.json`) — same plan + proposals held fixed across the whole sweep,
   only budget varies, so this isolates the budget axis as cleanly as
   `fig_proposal_scaling` isolates n_proposes.
6. **mechanism table** — a static qualitative comparison (mechanism / debug cost
   / token cost / leakage handling / adaptivity), written as markdown.

Run:
    uv run python scripts/make_figures.py --runs results --out figures
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
                    "budget": d.get("budget"),
                    "relational": bool(d.get("relational", False)),
                    "leaderboard": d.get("leaderboard"),
                    "reused_spec": bool(d.get("reused_spec", False)),
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

    Companion to `fig_budget_scaling`, reconstructed from a SINGLE `best_quality`
    run per task (see `_autogluon_progress_curve`) rather than a budget sweep.
    Read alongside `fig_budget_scaling`/`fig_proposal_scaling`, not merged into
    them — the y-axis score source (internal validation, not the shared
    external holdout) differs. MLE-STAR is intentionally excluded until a
    comparable per-step trace exists for it.

    One subplot per task, own x- AND y-axis each: tasks converge over very
    different cumulative-time ranges (seconds vs tens of minutes), and a
    shared axis would compress whichever task converges fastest into an
    invisible sliver at the left edge — same reasoning as
    `fig_proposal_scaling`'s per-task layout.
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

    tasks = sorted(by_task)
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 3.2 * n), 3.6), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        pts = by_task[task]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=".", markersize=4, linestyle="--")
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("cumulative fit time (s)", fontsize=8)
    axes[0][0].set_ylabel(
        "AutoGluon internal validation score\n(NOT the shared external holdout)",
        fontsize=8,
    )
    fig.suptitle(
        "AutoGluon exploration progress (best_quality, single run)", fontsize=11
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "autogluon_progress.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _n_proposes(r: dict) -> int:
    """Recover Option 3's call count from a record, live run or replay alike.

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
    """Extension holdout score vs Option-3 proposal count, one subplot per task.

    Deliberately minimal per-point: just task, n_proposes (`_n_proposes`), and
    score — no token/wall-clock panel, since the LLM-call axis IS n_proposes
    here, not a separate thing to also show (see `fig_token_cost` for the
    token side of this same axis, and `fig_budget_scaling` for the rollout-
    budget axis instead).

    ONLY `scripts/replay_from_run.py` replays count (`reused_spec=True`) — a
    live run only ever produces ONE point, at whatever n_proposes it happened
    to be run at; it doesn't pause mid-search to also report what the score
    would have been at a lower n_proposes. Mixing that single live endpoint in
    with the replay-derived curve is comparing a different measurement to a
    deliberately isolated one. A task whose live run hasn't been replayed at
    every n_proposes yet will show fewer points here until it is (see
    `replay_from_run`'s own `n_proposes` ceiling — it can only replay up to
    however many real proposals the source run logged).

    Points at the same (task, n_proposes) — e.g. two replays sourced from
    different live runs that both used n_proposes=1 — are averaged. Tasks
    with fewer than two distinct n_proposes points are skipped (nothing to
    draw a curve through).

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
        by_task_n[(r["task"], _n_proposes(r))].append(r["score"])
        scorer_of[r["task"]] = r["scorer"]

    by_task = defaultdict(list)
    for (task, n), scores in by_task_n.items():
        by_task[task].append((n, sum(scores) / len(scores)))
    by_task = {
        task: sorted(pts) for task, pts in by_task.items() if len(pts) >= 2
    }

    if not by_task:
        return None

    tasks = sorted(by_task)
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 3.2 * n), 3.6), squeeze=False)
    color = _METHOD_COLOR.get("extension", "#2b8cbe")
    for ax, task in zip(axes[0], tasks):
        pts = by_task[task]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=color)
        ax.set_title(task, fontsize=9)
        ax.set_ylabel(scorer_of.get(task, "holdout score"), fontsize=8)
        ax.set_xlabel("n_proposes", fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.suptitle("Proposal scaling — quality vs Option-3 call count", fontsize=11)
    fig.tight_layout()
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
    """Real LLM token cost vs Option-3 proposal count, one subplot per task —
    the cost-side companion to `fig_proposal_scaling`'s quality-side curve.

    ONLY live runs count (`reused_spec=False`) — the opposite filter from
    `fig_proposal_scaling`. A replay's whole point is that it costs zero new
    tokens (`make_fixed_sequence_proposer` never calls a real LLM), so it has
    nothing to show on a token axis; only a real run's logged
    `tokens_by_agent` breakdown has anything to plot (see
    `_cumulative_tokens_by_n_proposes` for exactly what's exact vs
    approximated within one run).

    One task, one live run's curve — unlike `fig_proposal_scaling`, points
    aren't averaged across multiple runs of the same task, since each live
    run's own agent-authored plan is the thing generating this specific cost
    trace; averaging costs across differently-authored plans would blur the
    "where does the token budget actually go" story this figure exists to
    tell.
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
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 3.2 * n), 3.6), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        pts = by_task[task]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color="#e6550d")
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("n_proposes", fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0][0].set_ylabel("cumulative tokens\n(exact at ends, interpolated between)",
                           fontsize=8)
    fig.suptitle("Token cost vs Option-3 call count (real live runs)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "token_cost.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_budget_scaling(records: list[dict], out_dir: str) -> str | None:
    """Extension holdout score vs rollout budget at FIXED n_proposes, one
    subplot per task — replaces the old live-run-based `fig_time_scaling`.

    ONLY `scripts/replay_from_run.py --budget-sweep` replays count
    (`reused_spec=True`), same reasoning as `fig_proposal_scaling`: a live run
    only ever ran at one budget, so it can't isolate the budget axis the way a
    sweep — same plan, same proposals, only budget varies — deliberately can.
    Points at the same (task, budget) are averaged. Tasks with fewer than two
    distinct budgets are skipped (nothing to draw a curve through).

    One subplot per task, own y-axis each — same scale/sign reasoning as
    `fig_proposal_scaling` (negative RMSE vs 0-1 accuracy/roc_auc).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_task_b = defaultdict(list)
    scorer_of = {}
    for r in records:
        if r["method"] != "extension" or not r["reused_spec"]:
            continue
        budget = r.get("budget")
        if budget is None:
            continue
        by_task_b[(r["task"], budget)].append(r["score"])
        scorer_of[r["task"]] = r["scorer"]

    by_task = defaultdict(list)
    for (task, budget), scores in by_task_b.items():
        by_task[task].append((budget, sum(scores) / len(scores)))
    by_task = {
        task: sorted(pts) for task, pts in by_task.items() if len(pts) >= 2
    }

    if not by_task:
        return None

    tasks = sorted(by_task)
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 3.2 * n), 3.6), squeeze=False)
    color = _METHOD_COLOR.get("extension", "#2b8cbe")
    for ax, task in zip(axes[0], tasks):
        pts = by_task[task]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=color)
        ax.set_title(task, fontsize=9)
        ax.set_ylabel(scorer_of.get(task, "holdout score"), fontsize=8)
        ax.set_xlabel("rollout budget", fontsize=8)
    fig.suptitle("Budget scaling — quality vs rollout budget (fixed n_proposes)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "budget_scaling.png")
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
    ag = fig_autogluon_progress(records, args.out)
    if ag:
        made.append(ag)
    ps = fig_proposal_scaling(records, args.out)
    if ps:
        made.append(ps)
    tc = fig_token_cost(records, args.out)
    if tc:
        made.append(tc)
    bs = fig_budget_scaling(records, args.out)
    if bs:
        made.append(bs)
    print("wrote:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    _main()
