"""Revive the original MLE-STAR agent graph and run it under hard caps.

MLE-STAR (the upstream Google agent, ``machine_learning_engineering/sub_agents/
{initialization,refinement,ensemble,submission}``) is dead code in this fork: the
four sub-agents are built at import but never wired into a root and never driven.
This script re-composes the root ``SequentialAgent`` and drives it, so we can
measure its ACTUAL token + call cost — the "mystery" the MCTS extension is
contrasted against (the extension's LLM cost is a fixed ``2 + N_PROPOSES``; MLE-
STAR's is unbounded: ~26 calls best case, 1000+ when the debug cascade fires).

Because that cost is unpredictable and the school/Gemini budget is scarce, this
runner imposes HARD CAPS with zero edits to the sub-agent source:

- **clamped knobs** (``num_solutions=1``, ``max_debug_round=1`` …) set on
  ``config.CONFIG`` *before* the sub-agents import (they read config at import).
- **a global call counter** that raises ``_CallBudgetExceeded`` once ``max_calls``
  model calls have fired — attached as a ``before_model_callback`` on every
  ``LlmAgent`` via a tree-walk.
- **a wall-clock deadline** (same callback) and a **per-call output-token bound**.
- **token capture** (``run_logging.extract_usage``) so MLE-STAR's tokens land in
  the SAME uniform ``result.json`` the extension and AutoGluon emit.
- **provider routing**: on a non-Gemini model (``PROVIDER=school``) each agent's
  bare ``model=`` string is swapped for a ``LiteLlm`` (mirroring
  ``adk_agent._resolve_model``) and the Gemini-only ``google_search`` tool is
  stripped.

**Honest caveats (state these in the writeup):** MLE-STAR writes + executes
generated Python (subprocess, non-deterministic); a run can burn its whole token
budget in the debug cascade and still produce no valid submission; the score it
prints ("Final Validation Performance") is its OWN CV and is NOT comparable —
only a re-score of its ``final/submission.csv`` on our shared holdout is. Treat
the result as ONE clamped data point, not a swept curve.

Run (best-effort; needs a live provider + budget):
    PROVIDER=school uv run python scripts/run_mlestar.py --task california-housing-prices \
        --max-calls 60 --time-budget-s 3600
"""

import argparse
import os
import time
from datetime import datetime


class _CallBudgetExceeded(Exception):
    """Raised by the cap callback to abort a run that hit its LLM-call budget."""


# clamp knobs conservatively so the debug cascade can't explode the token bill.
_CLAMP = {
    "num_solutions": 1,
    "num_model_candidates": 1,
    "max_retry": 2,
    "max_debug_round": 1,
    "max_rollback_round": 1,
    "inner_loop_round": 1,
    "outer_loop_round": 1,
    "ensemble_loop_round": 1,
}


def _clamp_config(task: str, data_dir: str, overrides: dict | None = None):
    """Set clamped knobs + task on ``config.CONFIG`` BEFORE the sub-agents import.

    The sub-agents read ``config.CONFIG`` at import (e.g. ``LoopAgent``'s
    ``max_iterations=config.CONFIG.outer_loop_round``), so clamping must happen
    first or it has no effect on the built graph.
    """
    from machine_learning_engineering.shared_libraries import config

    for key, value in {**_CLAMP, **(overrides or {})}.items():
        setattr(config.CONFIG, key, value)
    config.CONFIG.task_name = task
    config.CONFIG.data_dir = data_dir
    return config.CONFIG


def _as_callback_list(existing):
    """Normalize an ADK callback field (None | callable | list) to a list."""
    if existing is None:
        return []
    if isinstance(existing, list):
        return list(existing)
    return [existing]


def _instrument_tree(agent, *, cap_before, token_after, resolved_model,
                     is_gemini):
    """Recursively attach caps/token capture (and route the model) on every agent.

    Prepends ``cap_before`` to each ``LlmAgent``'s ``before_model_callback`` (so
    the budget check runs first) and appends ``token_after`` to
    ``after_model_callback``. On a non-Gemini provider, swaps the bare model
    string for the resolved ``LiteLlm`` and strips ``google_search`` (Gemini-only).
    """
    if hasattr(agent, "model"):
        agent.before_model_callback = [cap_before] + _as_callback_list(
            getattr(agent, "before_model_callback", None)
        )
        agent.after_model_callback = _as_callback_list(
            getattr(agent, "after_model_callback", None)
        ) + [token_after]
        if not is_gemini and isinstance(getattr(agent, "model", None), str):
            agent.model = resolved_model
            tools = getattr(agent, "tools", None)
            if tools:
                agent.tools = [
                    t for t in tools
                    if "google_search" not in type(t).__name__.lower()
                    and getattr(t, "__name__", "") != "google_search"
                ]
    for child in getattr(agent, "sub_agents", None) or []:
        _instrument_tree(
            child,
            cap_before=cap_before,
            token_after=token_after,
            resolved_model=resolved_model,
            is_gemini=is_gemini,
        )


def build_capped_root(max_calls: int, deadline: float, max_output_tokens: int,
                      token_sink: dict, counter: dict):
    """Compose the MLE-STAR root SequentialAgent with caps + token capture.

    Import the four sub-agents AFTER the config clamp (call ``_clamp_config``
    first), wire them into the upstream order, and instrument the whole tree.
    """
    from google.adk import agents

    from machine_learning_engineering.adk_agent import _resolve_model
    from machine_learning_engineering.run_logging import add_usage, extract_usage
    from machine_learning_engineering.shared_libraries import config
    from machine_learning_engineering.sub_agents.initialization.agent import (
        initialization_agent,
    )
    from machine_learning_engineering.sub_agents.refinement.agent import (
        refinement_agent,
    )
    from machine_learning_engineering.sub_agents.ensemble.agent import (
        ensemble_agent,
    )
    from machine_learning_engineering.sub_agents.submission.agent import (
        submission_agent,
    )

    resolved_model, is_gemini = _resolve_model(config.CONFIG.agent_model)

    # limits live on the shared `counter` dict so they're introspectable and a
    # single composed root can be re-exercised (ADK forbids re-parenting the
    # sub-agent singletons, so the root can only be built once per process).
    counter.setdefault("calls", 0)
    counter["max_calls"] = max_calls
    counter["deadline"] = deadline
    counter["max_output_tokens"] = max_output_tokens

    def cap_before(callback_context, llm_request):
        if time.perf_counter() >= counter["deadline"]:
            raise _CallBudgetExceeded("wall-clock deadline reached")
        counter["calls"] += 1
        if counter["calls"] > counter["max_calls"]:
            raise _CallBudgetExceeded(
                f"exceeded max_calls={counter['max_calls']}"
            )
        try:  # best-effort per-call output cap
            llm_request.config.max_output_tokens = counter["max_output_tokens"]
        except Exception:
            pass
        return None

    def token_after(callback_context, llm_response):
        add_usage(
            token_sink,
            getattr(callback_context, "agent_name", "?"),
            extract_usage(llm_response),
        )
        return None

    root = agents.SequentialAgent(
        name="mlestar_root",
        description="Original MLE-STAR: init -> refine -> ensemble -> submission.",
        sub_agents=[
            initialization_agent,
            refinement_agent,
            ensemble_agent,
            submission_agent,
        ],
    )
    _instrument_tree(
        root,
        cap_before=cap_before,
        token_after=token_after,
        resolved_model=resolved_model,
        is_gemini=is_gemini,
    )
    return root


def _run(root, task: str):
    """Drive the graph once with InMemoryRunner; return the final session state."""
    import asyncio

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    async def _go():
        runner = InMemoryRunner(agent=root, app_name="mlestar")
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="driver"
        )
        content = types.Content(
            parts=[types.Part(text=f"Solve the task: {task}")], role="user"
        )
        async for _ in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=content,
        ):
            pass
        return await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=session.user_id,
            session_id=session.id,
        )

    return asyncio.run(_go())


def _score_submission(task, out_workspace, seed):
    """Best-effort: re-score MLE-STAR's final/submission.csv on the SHARED holdout.

    Stages a temp task dir where test.csv is our holdout WITH target retained is
    NOT how MLE-STAR is wired (it reads the standard task dir), so this is a
    best-effort alignment of ``final/submission.csv`` predictions to the shared
    holdout target by row order. Returns ``{scorer, score}`` or ``None``.
    """
    import glob

    import numpy as np

    from machine_learning_engineering import ensemble, metrics, pipeline

    df, target, task_type, metric, _desc, _aux = pipeline.load_task(task)
    scorer = metrics.report_scorer(metric)
    if scorer is None or scorer not in ensemble._METRIC_FNS:
        return None
    _train, holdout = ensemble.holdout_split(df, target, task_type, seed=seed)
    hits = glob.glob(
        os.path.join(out_workspace, "**", "final", "submission.csv"),
        recursive=True,
    )
    if not hits:
        return None
    import pandas as pd

    sub = pd.read_csv(sorted(hits)[0])
    if len(sub) != len(holdout):
        return None  # can't align a differently-sized submission
    pred_col = sub.columns[-1]  # convention: last column holds predictions
    pred = np.asarray(sub[pred_col])
    try:
        score = ensemble._score(scorer, holdout[target], pred, None)
    except Exception:
        return None
    return {"scorer": scorer, "score": score}


def run_mlestar(task, out_dir, max_calls=60, time_budget_s=3600.0,
                max_output_tokens=8192, seed=42, data_dir=None):
    """Run capped MLE-STAR and emit a uniform result.json-style dict."""
    data_dir = data_dir or "./machine_learning_engineering/tasks/"
    _clamp_config(task, data_dir)
    from machine_learning_engineering.shared_libraries import config

    token_sink: dict = {}
    counter = {"calls": 0}
    deadline = time.perf_counter() + time_budget_s
    workspace = config.CONFIG.workspace_dir

    base = {
        "method": "mlestar",
        "task": task,
        "time_budget_s": time_budget_s,
        "max_calls": max_calls,
        "clamp": _CLAMP,
        "model": config.CONFIG.agent_model,
        "relational": True,  # MLE-STAR can use aux tables if it writes the join
    }
    wall_start = time.perf_counter()
    aborted = None
    try:
        root = build_capped_root(
            max_calls, deadline, max_output_tokens, token_sink, counter
        )
        _run(root, task)
    except _CallBudgetExceeded as exc:
        aborted = str(exc)
    except Exception as exc:  # any generated-code / provider failure
        aborted = f"{type(exc).__name__}: {exc}"
    wall_clock_s = round(time.perf_counter() - wall_start, 1)

    tokens = {"prompt": 0, "completion": 0, "total": 0}
    for slot in token_sink.values():
        for k in tokens:
            tokens[k] += slot.get(k, 0)
    holdout = _score_submission(task, workspace, seed)

    return {
        **base,
        "status": "aborted" if aborted else "ok",
        "abort_reason": aborted,
        "wall_clock_s": wall_clock_s,
        "llm_calls": counter["calls"],
        "tokens": tokens,
        "tokens_by_agent": {k: dict(v) for k, v in token_sink.items()},
        "holdout": holdout,  # None unless a submission.csv could be re-scored
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run capped MLE-STAR.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-calls", type=int, default=60,
                        help="hard cap on total LLM calls (abort past it)")
    parser.add_argument("--time-budget-s", type=float, default=3600.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="per-call output-token bound")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = args.out or os.path.join(
        "runs", f"mlestar_{args.task}_{datetime.now():%Y%m%d-%H%M}"
    )
    os.makedirs(out_dir, exist_ok=True)
    result = run_mlestar(
        args.task,
        out_dir,
        max_calls=args.max_calls,
        time_budget_s=args.time_budget_s,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
    )
    import json

    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nMLE-STAR | task: {result['task']}  status: {result['status']}")
    if result.get("abort_reason"):
        print(f"abort: {result['abort_reason']}")
    print(f"LLM calls: {result['llm_calls']}  |  tokens: {result['tokens']}")
    print(f"wall clock: {result['wall_clock_s']}s")
    print(f"holdout (comparable): {result['holdout']}")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    _main()
