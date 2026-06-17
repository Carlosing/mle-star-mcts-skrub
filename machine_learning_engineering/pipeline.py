"""End-to-end driver: task -> ADK plan -> MCTS search over skrub configs.

This is the glue that finally connects the agent layer to the search layer:

    load_task -> make_data_summary -> [ADK: analyst -> plan_author]
      -> resolve_spec -> build_staged_plan -> get_action_space + make_rollout_fn
      -> mcts_search -> (report on the incumbent)

The only live-LLM cost is the two agent turns; the MCTS rollouts are pure skrub
(no quota). Spec parsing/resolution is wrapped so a malformed or truncated LLM
response falls back to a minimal default rather than crashing the run.

Run:  uv run --no-sync python -m machine_learning_engineering.pipeline --budget 30
"""

import argparse
import asyncio
import os
import re

import pandas as pd
from google.adk.runners import InMemoryRunner
from google.genai import types

from machine_learning_engineering import metrics, skrub_ops
from machine_learning_engineering.adk_agent import build_root_agent
from machine_learning_engineering.data_summary import infer_task_type, make_data_summary
from machine_learning_engineering.mcts import mcts_search
from machine_learning_engineering.shared_libraries import config
from machine_learning_engineering.spec_resolver import resolve_spec

APP_NAME = "mle-mcts-skrub"


# --- task loading (minimal, target/metric parsed with safe fallbacks) --------


def load_task(
    task_name: str | None = None, data_dir: str | None = None, target: str | None = None
):
    """Return (df, target, task_type, metric) for a task directory.

    Reads ``train.csv`` and parses ``task_description.txt`` for the target
    ("Predict the X") and the metric ("# Metric" section). ``target`` can be
    passed explicitly to bypass the (brittle) description parse.
    """
    task_name = task_name or config.CONFIG.task_name
    data_dir = data_dir or config.CONFIG.data_dir
    task_dir = os.path.join(data_dir, task_name)

    df = pd.read_csv(os.path.join(task_dir, "train.csv"))
    desc = _read_description(task_dir)
    target = target or _parse_target(desc, df)
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in columns {list(df.columns)}")
    return df, target, infer_task_type(df, target), _parse_metric(desc)


def _read_description(task_dir: str) -> str:
    path = os.path.join(task_dir, "task_description.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def _parse_target(desc: str, df: pd.DataFrame) -> str:
    m = re.search(r"[Pp]redict\s+(?:the\s+)?[`'\"]?([A-Za-z0-9_]+)", desc)
    if m and m.group(1) in df.columns:
        return m.group(1)
    return df.columns[-1]  # fallback: last column is the label


def _parse_metric(desc: str) -> str:
    m = re.search(r"#\s*Metric\s*\n+\s*([A-Za-z0-9_]+)", desc)
    return m.group(1).strip().lower() if m else ""


# --- agent run ---------------------------------------------------------------


async def _run_agents(root_agent, user_text: str):
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="driver"
    )
    content = types.Content(parts=[types.Part(text=user_text)], role="user")
    async for _ in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
        pass
    return await runner.session_service.get_session(
        app_name=runner.app_name, user_id=session.user_id, session_id=session.id
    )


def _fallback_spec() -> dict:
    """Minimal, always-resolvable spec when the LLM output can't be used."""
    return {
        "encoder_options": ["GapEncoder"],
        "model": ["HistGradientBoosting", "RandomForest"],
    }


def _safe_resolve(raw, task_type: str) -> tuple[dict, bool]:
    """resolve_spec with a fallback; returns (spec, used_fallback)."""
    try:
        spec = resolve_spec(raw, task_type=task_type)
        return spec, False
    except Exception:
        return resolve_spec(_fallback_spec(), task_type=task_type), True


# --- the pipeline ------------------------------------------------------------


def run_pipeline(
    task_name: str | None = None,
    target: str | None = None,
    budget: int = 50,
    model=None,
    with_search: bool = True,
    log_dir: str | None = None,
    seed: int = 42,
) -> dict:
    """Run the full agent -> MCTS pipeline and return a results dict.

    ``model``/``with_search`` are forwarded to ``build_root_agent`` (tests pass a
    fake model + with_search=False to run fully offline).
    """
    df, target, task_type, metric = load_task(task_name, target=target)
    summary = make_data_summary(df, target)

    root = build_root_agent(model=model, with_search=with_search, log_dir=log_dir)
    session = asyncio.run(_run_agents(root, summary))

    raw = session.state.get("skrub_spec_raw", "")
    spec, used_fallback = _safe_resolve(raw, task_type)

    plan = skrub_ops.build_staged_plan(spec, df, target=target)
    action_space = skrub_ops.get_action_space(plan)
    start_state = skrub_ops.get_default_state(plan)
    rollout = skrub_ops.make_rollout_fn(
        plan, df, seed=seed, scoring=metrics.search_scorer(task_type)
    )

    best_state, best_score, _ = mcts_search(
        start_state, action_space, rollout, budget=budget
    )

    return {
        "task": task_name or config.CONFIG.task_name,
        "target": target,
        "task_type": task_type,
        "metric": metric,
        "search_scorer": metrics.search_scorer(task_type),
        "best_state": best_state,
        "best_search_score": best_score,
        "report": _report(plan, best_state, df, metric),
        "action_space": action_space,
        "used_fallback_spec": used_fallback,
        "analysis": session.state.get("dataset_analysis", ""),
        "spec_raw": raw,
    }


def _report(plan, state: dict, df, metric: str):
    """Score the incumbent on the task/competition metric, for reporting only."""
    scorer = metrics.report_scorer(metric)
    if scorer is None:
        return None
    try:
        return {
            "scorer": scorer,
            "score": skrub_ops.evaluate_full(plan, state, df, scoring=scorer),
        }
    except Exception:
        return None


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent->MCTS pipeline.")
    parser.add_argument("--task", default=None, help="task name under data_dir")
    parser.add_argument("--target", default=None, help="override target column")
    parser.add_argument("--budget", type=int, default=30, help="MCTS evaluations")
    args = parser.parse_args()

    result = run_pipeline(task_name=args.task, target=args.target, budget=args.budget)
    print(
        f"\nTask: {result['task']}  target={result['target']}  ({result['task_type']})"
    )
    print(
        f"Search scorer: {result['search_scorer']}  |  fallback spec: {result['used_fallback_spec']}"
    )
    print(f"Best config:   {result['best_state']}")
    print(
        f"Best search score ({result['search_scorer']}): {result['best_search_score']:.4f}"
    )
    if result["report"]:
        print(f"Report ({result['report']['scorer']}): {result['report']['score']:.4f}")


if __name__ == "__main__":
    _main()
