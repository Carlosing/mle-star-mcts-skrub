"""Run the MLE-STAR baseline under hard caps and emit a uniform result.json.

MLE-STAR is the comparison arm the MCTS extension is contrasted against. Its
LLM cost is the "mystery": the extension's is a fixed ``2 + N_PROPOSES`` calls,
while MLE-STAR's is unbounded — ~26 calls best case, far more once the debug
cascade fires. This script measures that cost.

The baseline itself is the OpenAI-compatible pipeline in
``machine_learning_engineering/{runner,agent}.py`` + ``sub_agents/`` (init ->
refine -> ensemble -> submission), driven by ``agent.run_pipeline``. It does NOT
use the ADK stack — that belongs to the extension's own agents.

Because the token bill is unpredictable and the school/Gemini budget is scarce,
this runner imposes HARD CAPS by wrapping ``runner.llm_call``, the single
chokepoint every sub-agent funnels through:

- **clamped knobs** (``num_solutions=1``, ``max_debug_round=1`` …) set on
  ``config.CONFIG``, which ``run_pipeline`` seeds onto the state the sub-agents
  actually read.
- **a call counter** raising ``_CallBudgetExceeded`` past ``max_calls``.
- **a wall-clock deadline** and a **per-call output-token bound**.
- **token capture** (``run_logging.extract_usage``, which already speaks the
  OpenAI ``usage`` shape) so MLE-STAR's tokens land in the SAME ``result.json``
  the extension and AutoGluon emit.

**Honest caveats (state these in the writeup):** MLE-STAR writes + executes
generated Python (subprocess, non-deterministic); a run can burn its whole token
budget in the debug cascade and still produce no valid submission; the score it
prints is its OWN CV and is NOT comparable — only a re-score of its
``final/submission.csv`` on our shared holdout is. Treat the result as ONE
clamped data point, not a swept curve.

Run (best-effort; needs a live provider + budget):
    PROVIDER=school uv run python scripts/run_mlestar.py --task california-housing-prices \
        --max-calls 60 --time-budget-s 3600
"""

import argparse
import os
import time
from datetime import datetime


class _CallBudgetExceeded(Exception):
    """Raised by the cap wrapper to abort a run that hit its LLM-call budget."""


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
    """Set clamped knobs + task on ``config.CONFIG`` before the pipeline runs.

    ``agent.run_pipeline`` seeds every config field onto the AgentState the
    sub-agents read, so setting them here is what actually bounds the run.
    """
    from machine_learning_engineering.shared_libraries import config

    for key, value in {**_CLAMP, **(overrides or {})}.items():
        setattr(config.CONFIG, key, value)
    config.CONFIG.task_name = task
    config.CONFIG.data_dir = data_dir
    return config.CONFIG


def _effective_model(model_name: str) -> str:
    """Strip the LiteLlm ``openai/`` prefix for the raw OpenAI-compatible client.

    ``SCHOOL_ROOT_AGENT_MODEL`` carries the prefix because the *extension's* ADK
    path routes through LiteLlm, which needs it. MLE-STAR talks to the endpoint
    directly, where ``openai/qwen3.5-397b-a17b`` is not a valid model id.

    Example:
        _effective_model("openai/qwen3.5-397b-a17b")  # -> "qwen3.5-397b-a17b"
        _effective_model("gemini-2.5-flash")          # -> "gemini-2.5-flash"
    """
    return model_name.removeprefix("openai/")


def install_caps(counter: dict, token_sink: dict, max_output_tokens: int):
    """Wrap ``runner.llm_call`` with the call/time caps and token capture.

    Patching the module attribute is enough even though the sub-agents bind
    ``run_agent`` by direct-name import: ``run_agent`` looks ``llm_call`` up in
    runner's globals at call time. Returns the original for restoration.
    """
    from machine_learning_engineering import runner
    from machine_learning_engineering.run_logging import (
        add_usage,
        extract_usage,
    )

    original = runner.llm_call
    default_model = _effective_model(counter["model"])

    # Mirror llm_call's signature exactly — a caller passing `model=` by keyword
    # must not break on the wrapper.
    def capped(messages, temperature=0.0, model=None, max_tokens=None):
        if time.perf_counter() >= counter["deadline"]:
            raise _CallBudgetExceeded("wall-clock deadline reached")
        counter["calls"] += 1
        if counter["calls"] > counter["max_calls"]:
            raise _CallBudgetExceeded(
                f"exceeded max_calls={counter['max_calls']}"
            )
        response = original(
            messages,
            temperature=temperature,
            model=model or default_model,
            max_tokens=max_tokens or max_output_tokens,
        )
        add_usage(token_sink, runner.CURRENT_AGENT, extract_usage(response))
        return response

    runner.llm_call = capped
    return original


def restore_caps(original) -> None:
    """Undo :func:`install_caps` so a second run in-process starts clean."""
    from machine_learning_engineering import runner

    runner.llm_call = original


def _score_submission(task, out_workspace, seed):
    """Score MLE-STAR's final/submission.csv against the SHARED holdout answers.

    MLE-STAR predicts the rows of the task's ``test.csv`` (it is told to, and it
    never sees ``test_answer.csv`` — create_workspace skips any file matching
    "answer"). Those are exactly the shared-bench rows, so its submission is
    directly comparable to the extension's and AutoGluon's holdout scores.

    Alignment is positional — the submission-format contract is one prediction
    per test row, in order — so a length mismatch means the submission does not
    correspond to the holdout and we REFUSE it. It must not fall through to a
    plausible-looking score: `test.csv` and a 25%-of-train holdout used to have
    the same length on 8 of 13 tasks, which is precisely how a meaningless number
    would get published.

    Returns ``{scorer, score, n}``, or ``{"error": ...}`` — never a silent None
    that reads as "no result" when it means "wrong result".
    """
    import glob

    import numpy as np

    from machine_learning_engineering import ensemble, metrics, pipeline

    _df, target, _task_type, metric, _desc, _aux = pipeline.load_task(task)
    scorer = metrics.report_scorer(metric)
    if scorer is None or scorer not in ensemble._METRIC_FNS:
        return {"error": f"no comparable report scorer for metric {metric!r}"}

    holdout = pipeline.load_holdout(task, target=target)
    if holdout is None:
        return {
            "error": "no staged holdout; run scripts/stage_tasks.py --force"
        }

    hits = glob.glob(
        os.path.join(out_workspace, "**", "final", "submission.csv"),
        recursive=True,
    )
    if not hits:
        return {"error": "MLE-STAR produced no final/submission.csv"}

    import pandas as pd

    sub = pd.read_csv(sorted(hits)[0])
    if len(sub) != len(holdout):
        return {
            "error": f"submission has {len(sub)} rows, holdout has "
            f"{len(holdout)} — not the same rows, refusing to score"
        }

    pred_col = sub.columns[-1]  # convention: last column holds predictions
    pred = np.asarray(sub[pred_col])
    try:
        score = ensemble._score(scorer, holdout[target], pred, None)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"scorer": scorer, "score": score, "n": len(holdout)}


def run_mlestar(
    task,
    out_dir,
    max_calls=60,
    time_budget_s=3600.0,
    max_output_tokens=8192,
    seed=42,
    data_dir=None,
    web_search=None,
):
    """Run capped MLE-STAR and emit a uniform result.json-style dict."""
    data_dir = data_dir or "./machine_learning_engineering/tasks/"
    overrides = {} if web_search is None else {"use_web_search": web_search}
    _clamp_config(task, data_dir, overrides)
    from machine_learning_engineering import agent
    from machine_learning_engineering.shared_libraries import config

    token_sink: dict = {}
    counter = {
        "calls": 0,
        "max_calls": max_calls,
        "deadline": time.perf_counter() + time_budget_s,
        "model": config.CONFIG.agent_model,
    }
    workspace = config.CONFIG.workspace_dir

    base = {
        "method": "mlestar",
        "task": task,
        "time_budget_s": time_budget_s,
        "max_calls": max_calls,
        "clamp": _CLAMP,
        "model": config.CONFIG.agent_model,
        "web_search": config.CONFIG.use_web_search,
        "relational": True,  # MLE-STAR can use aux tables if it writes the join
    }
    wall_start = time.perf_counter()
    aborted = None
    original = install_caps(counter, token_sink, max_output_tokens)
    try:
        agent.run_pipeline(
            task_name=task, data_dir=data_dir, workspace_dir=workspace
        )
    except _CallBudgetExceeded as exc:
        aborted = str(exc)
    except Exception as exc:  # any generated-code / provider failure
        aborted = f"{type(exc).__name__}: {exc}"
    finally:
        restore_caps(original)
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
    parser.add_argument(
        "--max-calls",
        type=int,
        default=60,
        help="hard cap on total LLM calls (abort past it)",
    )
    parser.add_argument("--time-budget-s", type=float, default=3600.0)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="per-call output-token bound",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--web-search",
        action="store_true",
        default=None,
        help="enable DuckDuckGo model retrieval (paper-faithful)",
    )
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
        web_search=args.web_search,
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
