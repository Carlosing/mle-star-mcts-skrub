"""End-to-end driver: task -> ADK plan -> MCTS search over skrub configs.

This is the glue that finally connects the agent layer to the search layer:

    load_task -> make_data_summary -> [ADK: analyst -> plan_author]
      -> resolve_spec -> build_staged_plan -> get_action_space + make_rollout_fn
      -> mcts_search -> (report on the incumbent)

The only live-LLM cost is the two agent turns; the MCTS rollouts are pure skrub
(no quota). Spec parsing/resolution is wrapped so a malformed or truncated LLM
response falls back to a minimal default rather than crashing the run.

Run:  uv run python -m machine_learning_engineering.pipeline --budget 15
"""

import argparse
import asyncio
import glob
import json
import os
import re
from datetime import datetime

import pandas as pd
from google.adk.runners import InMemoryRunner
from google.genai import types

from machine_learning_engineering import ensemble, metrics, skrub_ops
from machine_learning_engineering.adk_agent import build_root_agent
from machine_learning_engineering.data_summary import infer_task_type, make_data_summary
from machine_learning_engineering.search_loop import make_llm_proposer, run_search_loop
from machine_learning_engineering.shared_libraries import config
from machine_learning_engineering.spec_resolver import resolve_spec

APP_NAME = "mle-mcts-skrub"


# --- task loading (minimal, target/metric parsed with safe fallbacks) --------


def load_task(
    task_name: str | None = None, data_dir: str | None = None, target: str | None = None
):
    """Return (df, target, task_type, metric, desc, aux_tables) for a task dir.

    Reads ``train.csv`` and parses ``task_description.txt`` for the target
    ("Predict the X") and the metric ("# Metric" section). ``target`` can be
    passed explicitly to bypass the (brittle) description parse. Any
    ``aux_<name>.csv`` next to ``train.csv`` is loaded as an auxiliary table
    (``aux_tables={name: df}``, empty dict for single-table tasks), enabling
    the relational ``assemble`` stage.
    """
    task_name = task_name or config.CONFIG.task_name
    data_dir = data_dir or config.CONFIG.data_dir
    task_dir = os.path.join(data_dir, task_name)

    df = pd.read_csv(os.path.join(task_dir, "train.csv"))
    aux_tables = {
        os.path.basename(path)[len("aux_") : -len(".csv")]: pd.read_csv(path)
        for path in sorted(glob.glob(os.path.join(task_dir, "aux_*.csv")))
    }
    desc = _read_description(task_dir)
    target = target or _parse_target(desc, df)
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in columns {list(df.columns)}")
    return (
        df,
        target,
        infer_task_type(df, target),
        _parse_metric(desc),
        desc,
        aux_tables,
    )


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


def _fallback_spec(task_type: str) -> dict:
    """Minimal, always-resolvable spec when the LLM output can't be used."""
    suffix = "Classifier" if task_type == "classification" else "Regressor"
    return {
        "encoder_options": ["skrub.GapEncoder"],
        "model": [
            f"sklearn.ensemble.HistGradientBoosting{suffix}",
            f"sklearn.ensemble.RandomForest{suffix}",
        ],
    }


def _safe_resolve(
    raw, task_type: str, aux_schemas=None, main_columns=None
) -> tuple[dict, bool]:
    """resolve_spec with a fallback; returns (spec, used_fallback)."""
    try:
        spec = resolve_spec(
            raw, task_type=task_type, aux_schemas=aux_schemas, main_columns=main_columns
        )
        return spec, False
    except Exception:
        return resolve_spec(_fallback_spec(task_type), task_type=task_type), True


# --- the pipeline ------------------------------------------------------------


def run_pipeline(
    task_name: str | None = None,
    target: str | None = None,
    budget: int = 15,
    model=None,
    with_search: bool = True,
    log_dir: str | None = None,
    out_dir: str | None = None,
    seed: int = 42,
    outer_steps: int = 1,
    propose=None,
    data_dir: str | None = None,
    top_k: int = 1,
    c: float = 0.5,
    retarget: bool = True,
    spec_raw: str | None = None,
) -> dict:
    """Run the full agent -> MCTS pipeline and return a results dict.

    ``model``/``with_search`` are forwarded to ``build_root_agent`` (tests pass a
    fake model + with_search=False to run fully offline). The search itself runs
    through ``search_loop.run_search_loop`` with model-gated HP search and a score
    cache; ``outer_steps > 1`` runs the budget as fixed-size slices on one
    persisted tree, re-targeting a stage between slices (Option 1) and, with a
    ``propose`` callable, injecting new options for it before each subsequent
    slice (Option 3 — the CLI's ``--n-proposes N`` is sugar for this). If
    ``out_dir`` is set, the run is persisted there.

    ``c``/``retarget`` are forwarded to ``run_search_loop`` (UCT exploration
    constant; whether Option 1 re-picks the target stage between slices).
    ``spec_raw`` skips the analyst/plan-author agents entirely and resolves the
    given raw plan instead — the sweep harness fetches the spec once per task
    and reuses it across points, so LLM calls stay O(tasks), not O(runs).
    """
    if out_dir is not None and log_dir is None:
        log_dir = out_dir

    df, target, task_type, metric, description, aux_tables = load_task(
        task_name, data_dir=data_dir, target=target
    )
    summary = make_data_summary(df, target, aux_tables=aux_tables)

    if spec_raw is None:
        root = build_root_agent(model=model, with_search=with_search, log_dir=log_dir)
        session = asyncio.run(_run_agents(root, summary))
        raw = session.state.get("skrub_spec_raw", "")
        analysis = session.state.get("dataset_analysis", "")
    else:
        raw, analysis = spec_raw, ""  # reused spec: no agent turns, no analysis
    spec, used_fallback = _safe_resolve(
        raw,
        task_type,
        aux_schemas={name: list(adf.columns) for name, adf in aux_tables.items()},
        main_columns=[c for c in df.columns if c != target],
    )

    search = run_search_loop(
        spec,
        df,
        target,
        scoring=metrics.search_scorer(task_type, metric),
        outer_steps=outer_steps,
        budget_per_step=budget,
        c=c,
        seed=seed,
        propose=propose,
        retarget=retarget,
        aux_tables=aux_tables or None,
        context_text=summary,
        stratify=(task_type == "classification"),
    )

    result = {
        "task": task_name or config.CONFIG.task_name,
        "task_description": description,
        "target": target,
        "task_type": task_type,
        "metric": metric,
        "model": config.CONFIG.agent_model,
        "budget": budget,
        "search_scorer": metrics.search_scorer(task_type, metric),
        "best_state": search["best_state"],
        "best_search_score": search["best_score"],
        "report": _report(
            search["plan"],
            search["best_state"],
            df,
            metric,
            aux=aux_tables or None,
            stratify=(task_type == "classification"),
        ),
        "action_space": search["action_space"],
        "target_key": search["target_key"],
        "injected_options": search["injected_options"],
        "ensemble": _top_k_report(
            search, df, target, task_type, metric, aux_tables, top_k, seed
        ),
        "used_fallback_spec": used_fallback,
        "reused_spec": spec_raw is not None,
        "llm_calls": (0 if spec_raw is not None else 2)
        + ((outer_steps - 1) if propose is not None else 0),
        "data_summary": summary,  # the data report fed to the analyst
        "analysis": analysis,  # report -> planner
        "spec_raw": raw,  # the plan the planner generated
    }
    if out_dir is not None:
        save_run_artifacts(result, out_dir)
    return result


# --- result persistence (for the team's presentation) ------------------------


def default_out_dir(task: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return os.path.join("runs", f"{task}_{stamp}")


def save_run_artifacts(result: dict, out_dir: str) -> str:
    """Write result.json (full) + summary.md (readable) into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(_result_markdown(result))
    return out_dir


def _result_markdown(r: dict) -> str:
    rep = r.get("report") or {}
    lines = [
        f"# Run: {r['task']}  ({r['task_type']}, metric={r['metric']})",
        f"model: {r.get('model')}  |  budget: {r.get('budget')}  |  "
        f"fallback spec: {r.get('used_fallback_spec')}",
        "",
        "## Task",
        (r.get("task_description") or "").strip() or "(no description)",
        "",
        "## 1. Data report (analyst output, sent to the planner agent)",
        (r.get("analysis") or "").strip() or "(empty)",
        "",
        "## 2. Generated plan (planner output)",
        "```json",
        (r.get("spec_raw") or "").strip() or "(empty)",
        "```",
        "",
        "## 3. Best configuration + score (MCTS incumbent)",
        "```json",
        json.dumps(r.get("best_state", {}), indent=2, default=str),
        "```",
        f"- search reward ({r.get('search_scorer')}): {r.get('best_search_score')}",
    ]
    if rep:
        lines.append(f"- report metric ({rep.get('scorer')}): {rep.get('score')}")
    ens = r.get("ensemble")
    if ens:
        lines.append(
            f"- top-{ens['k']} ensemble ({ens['scorer']}): {ens['ensemble_score']:.4f} "
            f"vs individuals {['%.4f' % s for s in ens['individual_scores']]}"
        )
    if r.get("target_key"):
        lines.append(f"- targeted stage (Option 1): {r['target_key']}")
    if r.get("injected_options"):
        lines.append(
            f"- injected options not in the original plan (Option 3): "
            f"{r['injected_options']}"
        )
    lines += [
        "",
        "## Appendix — MCTS search space",
        "```json",
        json.dumps(r.get("action_space", {}), indent=2, default=str),
        "```",
        "",
        "## Appendix — data digest (sent to the analyst)",
        "```",
        (r.get("data_summary") or "").strip(),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _top_k_report(search, df, target, task_type, metric, aux_tables, top_k, seed):
    """Top-k ensemble vs incumbent on a holdout (None when top_k <= 1 fails)."""
    if top_k <= 1:
        return None
    scorer = metrics.report_scorer(metric) or metrics.search_scorer(task_type, metric)
    try:
        states = ensemble.top_k_states(search["score_cache"], top_k)
        return ensemble.evaluate_top_k(
            search["plan"],
            states,
            df,
            target,
            task_type,
            scoring=scorer,
            aux=aux_tables or None,
            seed=seed,
        )
    except Exception:
        return None


def _report(
    plan, state: dict, df, metric: str, aux: dict | None = None, stratify: bool = False
):
    """Score the incumbent on the task/competition metric, for reporting only."""
    scorer = metrics.report_scorer(metric)
    if scorer is None:
        return None
    try:
        return {
            "scorer": scorer,
            "score": skrub_ops.evaluate_full(
                plan,
                state,
                df,
                aux=aux,
                main_var="data" if aux else None,
                scoring=scorer,
                stratify=stratify,
            ),
        }
    except Exception:
        return None


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent->MCTS pipeline.")
    parser.add_argument("--task", default=None, help="task name under data_dir")
    parser.add_argument("--target", default=None, help="override target column")
    parser.add_argument("--budget", type=int, default=15, help="MCTS evaluations")
    parser.add_argument(
        "--out", default=None, help="artifact dir (default: runs/<task>_<timestamp>)"
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=1,
        help="search phases; >1 enables ablation targeting (Option 1)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="enable LLM per-stage option injection (Option 3); needs --outer-steps>1",
    )
    parser.add_argument(
        "--n-proposes",
        type=int,
        default=None,
        help="interleave N proposer calls between fixed-budget slices "
        "(sugar for --refine --outer-steps N+1: search BUDGET -> propose -> "
        "search BUDGET -> ...)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="ensemble the top-k incumbents from the score cache (1 = off)",
    )
    parser.add_argument(
        "--c", type=float, default=0.5, help="UCT exploration constant"
    )
    parser.add_argument(
        "--no-retarget",
        action="store_true",
        help="keep the first Option-1 target stage for the whole run",
    )
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    args = parser.parse_args()

    task = args.task or config.CONFIG.task_name
    out_dir = args.out or default_out_dir(task)
    outer_steps, refine = args.outer_steps, args.refine
    if args.n_proposes is not None:  # sugar: N propose calls between slices
        outer_steps, refine = args.n_proposes + 1, args.n_proposes > 0
    propose = make_llm_proposer() if refine else None
    result = run_pipeline(
        task_name=args.task,
        target=args.target,
        budget=args.budget,
        out_dir=out_dir,
        seed=args.seed,
        outer_steps=outer_steps,
        propose=propose,
        top_k=args.top_k,
        c=args.c,
        retarget=not args.no_retarget,
    )
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
    if result.get("ensemble"):
        ens = result["ensemble"]
        print(
            f"Top-{ens['k']} ensemble ({ens['scorer']}): {ens['ensemble_score']:.4f} "
            f"(incumbent {ens['individual_scores'][0]:.4f})"
        )
    if result.get("target_key"):
        print(f"Targeted stage: {result['target_key']}")
    if result.get("injected_options"):
        print(f"Injected options (not in original plan): {result['injected_options']}")
    print(
        f"\nArtifacts written to: {out_dir}/ "
        "(summary.md, result.json, <agent>_<phase>.json)"
    )


if __name__ == "__main__":
    _main()
