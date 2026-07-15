"""Replay a completed run's exact plan + real proposals at reduced n_proposes.

Isolates the marginal value of each Option-3 `propose()` call without
spending new LLM budget: reads a past run's `result.json` (`spec_raw`, the
budget, task, ensemble size, ...) plus its `proposer_<k>_response.json`
artifacts — the REAL parsed proposal dicts the live LLM returned during that
run — and replays `pipeline.run_pipeline` using only the first `n_proposes`
of those real proposals as a deterministic stand-in proposer. Everything else
(initial plan, budget, seed) is held identical to the source run, so the
*only* thing that varies across replays is how many real proposals the search
got to react to.

Naming note: the INITIAL plan (`spec_raw`, authored once up front by
`data_analyst` -> `plan_author`) is NOT a "proposal" — only
`proposer_<k>_response.json` files (Option 3's mid-search extension calls)
count toward `n_proposes`. `plan_author_response.json` /
`data_analyst_response.json` are ignored here on purpose.

The source run is also the ceiling: it only ever called `propose()`
`n_proposes` times live, so replaying at a HIGHER `n_proposes` than that has
no real recorded content to draw on — this script raises rather than silently
padding with something that was never actually said by the LLM.

Writes straight into `results/` (the small, git-shareable mirror
`scripts/collect_results.py` otherwise populates from `runs/`) — same
`result.json`-only convention, no `runs/`-style side artifacts, since a
replay's compute never touches an API and never needs its own `data_analyst_
*`/`plan_author_*` files (those simply aren't produced when `spec_raw` skips
the agent graph, live or replayed).

Run:
    uv run python scripts/replay_from_run.py \\
        --source runs/credit-fraud_20260712-1806 --n-proposes 0
    uv run python scripts/replay_from_run.py \\
        --source runs/credit-fraud_20260712-1806 --n-proposes 1
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime

from machine_learning_engineering import pipeline


def _available_proposals(source_dir: str) -> list[dict]:
    """Real proposal dicts logged under `source_dir`, ordered by call number.

    Only `proposer_<k>_response.json` files count — see module docstring for
    why the initial plan doesn't. Each such file is a list of log entries
    (`search_loop.make_llm_proposer`'s `_log` appends one per phase); the
    last entry for a given call number carries that call's `"proposal"` dict.

    Example:
        _available_proposals("runs/credit-fraud_20260712-1806")
        # -> [proposal_1_dict, proposal_2_dict]
    """
    paths = glob.glob(os.path.join(source_dir, "proposer_*_response.json"))
    numbered = []
    for path in paths:
        m = re.search(r"proposer_(\d+)_response\.json$", os.path.basename(path))
        if m:
            numbered.append((int(m.group(1)), path))
    numbered.sort()

    proposals = []
    for _, path in numbered:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        proposal = records[-1].get("proposal") if records else None
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def make_fixed_sequence_proposer(proposals: list[dict]):
    """A pure, offline `propose(plan_json, context) -> dict | None` replaying
    REAL logged proposals in order, one per call.

    Returns `None` once past the recorded sequence — unlike
    `scripts/claude_agents.make_replay_proposer` (which repeats a single
    canned extension forever), here running out of real proposals should
    mean "nothing more to inject", not "repeat the last one".
    """
    calls = [0]

    def propose(plan_json, context=None):
        i = calls[0]
        calls[0] += 1
        return proposals[i] if i < len(proposals) else None

    return propose


def replay_from_run(
    source_dir: str,
    n_proposes: int,
    seed: int | None = None,
) -> dict:
    """Replay `source_dir`'s exact plan + first `n_proposes` real proposals.

    Pure compute — returns the result dict, touches no disk itself (mirrors
    `run_autogluon.run_autogluon`/`run_mlestar.run_mlestar`: the `_main`s own
    all persistence). Raises `ValueError` if `n_proposes` exceeds how many
    proposals the source run actually logged (see module docstring).

    Every field (budget, seed, ensemble size, time budget) is reused from the
    source run; the ONLY thing that varies across replays is `n_proposes`, so
    each one isolates the marginal value of an Option-3 call.
    """
    with open(os.path.join(source_dir, "result.json"), encoding="utf-8") as f:
        source = json.load(f)

    proposals = _available_proposals(source_dir)
    if n_proposes > len(proposals):
        logged = sorted(
            os.path.basename(p)
            for p in glob.glob(
                os.path.join(source_dir, "proposer_*_response.json")
            )
        )
        raise ValueError(
            f"requested n_proposes={n_proposes} but {source_dir!r} only logged "
            f"{len(proposals)} real proposal(s) ({logged}) — can't replay more "
            "than the source run actually called."
        )

    return pipeline.run_pipeline(
        task_name=source["task"],
        budget=source["budget"],
        seed=seed if seed is not None else source.get("seed", 42),
        outer_steps=n_proposes + 1,
        propose=make_fixed_sequence_proposer(proposals[:n_proposes])
        if n_proposes > 0
        else None,
        top_k=(source.get("ensemble") or {}).get("k", 1),
        spec_raw=source["spec_raw"],
        subsample_seeds=source.get("subsample_seeds"),
        time_budget_s=source.get("time_budget_s"),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a completed run's exact plan + real proposals "
        "at reduced n_proposes, offline and at zero new LLM cost."
    )
    parser.add_argument(
        "--source", required=True, help="path to the completed run directory"
    )
    parser.add_argument(
        "--n-proposes",
        type=int,
        required=True,
        help="how many of the source run's REAL logged proposals to replay "
        "(must be <= how many it actually called)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the replay seed (default: the source run's seed, or "
        "42 if not recorded)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="artifact dir (default: results/replay_<task>_np<n>_<ts> — the "
        "same small git-shareable mirror scripts/collect_results.py populates "
        "from runs/, result.json only)",
    )
    args = parser.parse_args()

    source_dir = args.source.rstrip("/")
    with open(os.path.join(source_dir, "result.json"), encoding="utf-8") as f:
        task = json.load(f)["task"]

    out_dir = args.out or os.path.join(
        "results",
        f"replay_{task}_np{args.n_proposes}_{datetime.now():%Y%m%d-%H%M}",
    )

    result = replay_from_run(source_dir, args.n_proposes, seed=args.seed)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    hold = result.get("report")
    print(
        f"\nReplay | task: {task}  source: {source_dir}  "
        f"n_proposes: {args.n_proposes}"
    )
    print(
        f"best_search_score: {result.get('best_search_score')}  |  "
        f"report: {hold}"
    )
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    _main()
