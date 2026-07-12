"""End-to-end driver: task -> ADK plan -> MCTS search over skrub configs.

This is the glue that finally connects the agent layer to the search layer:

    load_task -> make_data_summary -> [ADK: analyst -> plan_author]
      -> resolve_spec -> build_staged_plan -> get_action_space + make_rollout_fn
      -> mcts_search -> (report on the incumbent)

The only live-LLM cost is the two agent turns; the MCTS rollouts are pure skrub
(no quota). Spec parsing/resolution is wrapped so a malformed or truncated LLM
response falls back to a minimal default rather than crashing the run.

Run:  uv run python -m machine_learning_engineering.pipeline --budget 20
"""

import argparse
import asyncio
import glob
import json
import os
import re
import time
from datetime import datetime

import pandas as pd
from google.adk.runners import InMemoryRunner
from google.genai import types

from machine_learning_engineering import ensemble, metrics, skrub_ops
from machine_learning_engineering.adk_agent import build_root_agent
from machine_learning_engineering.data_summary import (
    infer_task_type,
    make_data_summary,
)
from machine_learning_engineering.search_loop import (
    make_llm_proposer,
    run_search_loop,
)
from machine_learning_engineering.shared_libraries import config
from machine_learning_engineering import spec_resolver
from machine_learning_engineering.spec_resolver import resolve_spec

APP_NAME = "mle-mcts-skrub"


# --- task loading (minimal, target/metric parsed with safe fallbacks) --------


def load_task(
    task_name: str | None = None,
    data_dir: str | None = None,
    target: str | None = None,
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
    if df[target].isna().any():
        # unlabeled rows crash skrub's eager plan preview ("Input y contains
        # NaN") and would otherwise silently zero every regression rollout
        df = df.dropna(subset=[target])
    metric = _parse_metric(desc)
    return (
        df,
        target,
        infer_task_type(df, target, metric),
        metric,
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


_TRANSIENT_AGENT_ERRORS = ("503", "UNAVAILABLE", "500", "INTERNAL", "overloaded")


def _is_transient(exc: Exception) -> bool:
    """True for a provider-side hiccup worth retrying (not a quota or auth wall).

    Example:
        _is_transient(RuntimeError("503 UNAVAILABLE"))   # -> True
        _is_transient(RuntimeError("401 UNAUTHENTICATED"))  # -> False
    """
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in _TRANSIENT_AGENT_ERRORS)


async def _run_agents(
    root_agent, user_text: str, attempts: int = 4, backoff_s: float = 8.0
):
    """Drive the agent graph, retrying provider 5xx with exponential backoff.

    Gemini answers a small ping while 503-ing the real, google_search-grounded
    agent calls, so a run would die at data_analyst or plan_author and throw
    away whatever calls already succeeded. Quota/auth errors are NOT retried —
    they will not clear on their own.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
            session = await runner.session_service.create_session(
                app_name=runner.app_name, user_id="driver"
            )
            content = types.Content(
                parts=[types.Part(text=user_text)], role="user"
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
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last = exc
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            wait = backoff_s * (2**attempt)
            print(
                f"[agents] transient provider error ({exc.__class__.__name__}); "
                f"retry {attempt + 1}/{attempts - 1} in {wait:.0f}s"
            )
            await asyncio.sleep(wait)
    raise last  # unreachable; the loop either returns or raises


def _fallback_spec(task_type: str) -> dict:
    """Minimal, always-resolvable spec when the LLM output can't be used."""
    suffix = "Classifier" if task_type == "classification" else "Regressor"
    return {
        # the Cleaner + TableVectorizer backbones are always-on at skrub
        # defaults, so a bare model list is a complete, resolvable pipeline
        "model": [
            f"sklearn.ensemble.HistGradientBoosting{suffix}",
            f"sklearn.ensemble.RandomForest{suffix}",
        ],
    }


def _safe_resolve(
    raw, task_type: str, aux_schemas=None, main_columns=None
) -> tuple[dict, dict, bool]:
    """resolve_spec with a fallback; returns (spec, raw_plan, used_fallback).

    `raw_plan` is the parsed plan dict that was actually resolved (the LLM's,
    or the fallback) — the search loop hands it to the Option-3 proposer as
    the plan to extend.
    """
    try:
        raw_plan = spec_resolver.parse_spec_json(raw)
        spec = resolve_spec(
            raw_plan,
            task_type=task_type,
            aux_schemas=aux_schemas,
            main_columns=main_columns,
        )
        return spec, raw_plan, False
    except Exception:
        fallback = _fallback_spec(task_type)
        return resolve_spec(fallback, task_type=task_type), fallback, True


def _auto_subsample_seeds(df, target: str, task_type: str) -> int:
    """Auto-pick the rollout seed-averaging factor from the target profile.

    Imbalanced classification (minority share < 5%) gets 3 seeded subsamples
    per rollout — single-draw rewards there are too noisy for UCT (budget-80
    demonstrably chased a noisy proxy to a worse held-out config than
    budget-20 on 1.25%-positive credit-fraud). Everything else keeps 1.

    Example:
        _auto_subsample_seeds(fraud_df, "fraud", "classification")  # -> 3
        _auto_subsample_seeds(housing_df, "value", "regression")    # -> 1
    """
    if task_type != "classification":
        return 1
    share = float(df[target].value_counts(normalize=True).min())
    return 3 if share < 0.05 else 1


def _token_report(agent_usage: dict, propose) -> dict:
    """Assemble the ``tokens`` + ``tokens_by_agent`` result fields.

    ``agent_usage`` is the per-agent sink the ADK callbacks filled (analyst,
    plan_author); the proposer's tokens ride on ``propose.token_totals`` (an
    attribute set by ``make_llm_proposer``). All-zero when a run reused a spec
    and made no proposer calls, or when the provider returned no usage.

    Example:
        _token_report({"plan_author": {"prompt": 800, "completion": 1500,
                       "total": 2300, "calls": 1}}, None)
        # -> {"tokens": {"prompt": 800, "completion": 1500, "total": 2300},
        #     "tokens_by_agent": {"plan_author": {...}}}
    """
    by_agent = {k: dict(v) for k, v in agent_usage.items()}
    prop = getattr(propose, "token_totals", None)
    if prop is not None:
        by_agent["proposer"] = dict(prop)
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    for slot in by_agent.values():
        for k in tokens:
            tokens[k] += slot.get(k, 0)
    return {"tokens": tokens, "tokens_by_agent": by_agent}


# --- the pipeline ------------------------------------------------------------


def run_pipeline(
    task_name: str | None = None,
    target: str | None = None,
    budget: int = 20,
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
    subsample_seeds: int | None = None,
    n_jobs: int = 6,
    time_budget_s: float | None = None,
    ensemble_pool: int = 10,
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

    ``subsample_seeds`` averages each rollout's reward over that many seeded
    subsamples (``skrub_ops.make_rollout_fn(n_subsample_seeds=)``). ``None``
    (default) auto-selects: 3 for imbalanced classification (minority share
    < 5%, where single-draw rewards are too noisy for UCT), else 1.
    """
    if out_dir is not None and log_dir is None:
        log_dir = out_dir

    wall_start = time.perf_counter()
    df, target, task_type, metric, description, aux_tables = load_task(
        task_name, data_dir=data_dir, target=target
    )
    summary = make_data_summary(df, target, aux_tables=aux_tables, metric=metric)

    agent_usage: dict = {}  # per-agent token totals from the ADK callbacks
    if spec_raw is None:
        root = build_root_agent(
            model=model,
            with_search=with_search,
            log_dir=log_dir,
            usage_sink=agent_usage,
        )
        session = asyncio.run(_run_agents(root, summary))
        raw = session.state.get("skrub_spec_raw", "")
        analysis = session.state.get("dataset_analysis", "")
        # `google_search` grounding intermittently makes the plan_author emit an
        # EMPTY response instead of its JSON (grounding conflicts with strict
        # structured output). Rather than collapse to the minimal fallback spec,
        # retry the graph once with search OFF — that path reliably returns the
        # rich JSON plan. Only triggers on the search path, only when empty.
        if with_search and not (raw or "").strip():
            root = build_root_agent(
                model=model,
                with_search=False,
                log_dir=log_dir,
                usage_sink=agent_usage,  # same sink: both attempts' tokens count
            )
            session = asyncio.run(_run_agents(root, summary))
            raw = session.state.get("skrub_spec_raw", "") or raw
            analysis = session.state.get("dataset_analysis", "") or analysis
    else:
        raw, analysis = spec_raw, ""  # reused spec: no agent turns, no analysis
    aux_schemas = {
        name: list(adf.columns) for name, adf in aux_tables.items()
    }
    main_columns = [c for c in df.columns if c != target]
    spec, raw_plan, used_fallback = _safe_resolve(
        raw, task_type, aux_schemas=aux_schemas, main_columns=main_columns
    )

    def _resolve_plan(plan_json: dict) -> dict:
        # the Option-3 rebuild path: same validation envelope as the original
        return resolve_spec(
            plan_json,
            task_type=task_type,
            aux_schemas=aux_schemas,
            main_columns=main_columns,
        )

    if subsample_seeds is None:
        subsample_seeds = _auto_subsample_seeds(df, target, task_type)

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
        raw_spec=raw_plan,
        resolve=_resolve_plan,
        retarget=retarget,
        aux_tables=aux_tables or None,
        context_text=summary,
        stratify=(task_type == "classification"),
        n_subsample_seeds=subsample_seeds,
        n_jobs=n_jobs,
        time_budget_s=time_budget_s,
    )

    scoring_errors: dict = {}  # populated by _report/_holdout_report/_top_k_report
    result = {
        "method": "extension",
        "task": task_name or config.CONFIG.task_name,
        "task_description": description,
        "target": target,
        "task_type": task_type,
        "metric": metric,
        "model": config.CONFIG.agent_model,
        "budget": budget,
        "time_budget_s": time_budget_s,
        "search_scorer": metrics.search_scorer(task_type, metric),
        # best_search_score is a bounded UCT reward, not the raw scorer value
        "search_reward_scale": skrub_ops.reward_scale(
            metrics.search_scorer(task_type, metric)
        ),
        "best_state": search["best_state"],
        "best_search_score": search["best_score"],
        "report": _report(
            search["plan"],
            search["best_state"],
            df,
            metric,
            aux=aux_tables or None,
            stratify=(task_type == "classification"),
            errors=scoring_errors,
        ),
        # incumbent on the SHARED holdout — the cross-method comparable number
        "holdout": _holdout_report(
            search, df, target, task_type, metric, aux_tables, seed,
            errors=scoring_errors,
        ),
        "action_space": search["action_space"],
        "target_key": search["target_key"],
        "injected_options": search["injected_options"],
        "proposal_injection_error": search.get("proposal_injection_error"),
        "refined_dims": search.get("refined_dims", []),
        "subsample_seeds": subsample_seeds,
        "ensemble": _top_k_report(
            search, df, target, task_type, metric, aux_tables, top_k, seed,
            errors=scoring_errors, pool=ensemble_pool,
        ),
        "used_fallback_spec": used_fallback,
        "dropped_sections": spec.get("dropped_sections", []),
        "single_option_stages": spec.get("single_option_stages", []),
        # scoring-phase exceptions (report/holdout/ensemble), so a failed final
        # score is a visible error string, not a silent None
        "scoring_errors": scoring_errors or None,
        "reused_spec": spec_raw is not None,
        "llm_calls": (0 if spec_raw is not None else 2)
        + ((outer_steps - 1) if propose is not None else 0),
        "wall_clock_s": round(time.perf_counter() - wall_start, 1),
        **_token_report(agent_usage, propose),
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
    tok = r.get("tokens") or {}
    lines = [
        f"# Run: {r['task']}  ({r['task_type']}, metric={r['metric']})",
        f"model: {r.get('model')}  |  budget: {r.get('budget')}  |  "
        f"fallback spec: {r.get('used_fallback_spec')}",
        f"LLM calls: {r.get('llm_calls')}  |  tokens: "
        f"{tok.get('total', 0):,} (prompt {tok.get('prompt', 0):,} + "
        f"completion {tok.get('completion', 0):,})",
        "",
    ]
    if (
        r.get("dropped_sections")
        or r.get("single_option_stages")
        or r.get("proposal_injection_error")
    ):
        lines.append("## ⚠ Plan quality warnings")
        if r.get("dropped_sections"):
            lines.append(
                f"- **dropped sections** (present in the plan but malformed, so "
                f"resolved to nothing and not searched): {r['dropped_sections']}"
            )
        if r.get("single_option_stages"):
            lines.append(
                f"- **single-option stages** (only one choice, so nothing to "
                f"search): {r['single_option_stages']}"
            )
        if r.get("proposal_injection_error"):
            lines.append(
                f"- **proposal injection error** (Option 3 proposed an extension "
                f"that could not be resolved/built, so it was skipped): "
                f"{r['proposal_injection_error']}"
            )
        lines.append("")
    lines += [
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
        f"- search reward ({r.get('search_scorer')}, "
        f"scale={r.get('search_reward_scale', 'raw')}): "
        f"{r.get('best_search_score')}",
    ]
    if rep:
        lines.append(
            f"- report metric ({rep.get('scorer')}): {rep.get('score')}"
        )
    ens = r.get("ensemble")
    if ens:
        lines.append(
            f"- Caruana ensemble ({ens['scorer']}, {ens['k']} of "
            f"{ens.get('pool_size', '?')} pool): {ens['ensemble_score']:.4f} "
            f"(unweighted mean-combine {ens.get('ensemble_score_mean', float('nan')):.4f}) "
            f"vs individuals {['%.4f' % s for s in ens['individual_scores']]}"
        )
        if ens.get("weights"):
            lines.append(
                f"  - ensemble weights: {['%.2f' % w for w in ens['weights']]}"
            )
    if r.get("target_key"):
        lines.append(f"- targeted stage (Option 1): {r['target_key']}")
    if r.get("refined_dims"):
        lines.append(
            f"- focused-refinement bonus phase edited: {r['refined_dims']}"
        )
    if r.get("injected_options"):
        lines.append(
            f"- injected options not in the original plan (Option 3): "
            f"{r['injected_options']}"
        )
    if r.get("scoring_errors"):
        lines.append(
            f"- ⚠️ scoring-phase failures (report/holdout/ensemble): "
            f"{r['scoring_errors']}"
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


def _top_k_report(
    search, df, target, task_type, metric, aux_tables, top_k, seed,
    errors=None, pool=10,
):
    """Caruana ensemble over a candidate pool vs incumbent on a holdout.

    Fits the top-``max(2*top_k, pool)`` distinct cache configs (a pool wider than
    ``top_k`` so buried model-family diversity is reachable) and Caruana-selects
    a weighted ensemble of at most ``top_k`` members. None when top_k <= 1 or on
    any failure (recorded via ``errors``).
    """
    if top_k <= 1:
        return None
    scorer = metrics.report_scorer(metric) or metrics.search_scorer(
        task_type, metric
    )
    try:
        pool_states = ensemble.top_k_states(
            search["score_cache"],
            max(2 * top_k, pool),
            defaults=skrub_ops.get_default_state(search["plan"]),
        )
        return ensemble.evaluate_top_k(
            search["plan"],
            pool_states,
            df,
            target,
            task_type,
            scoring=scorer,
            aux=aux_tables or None,
            seed=seed,
            size=top_k,
        )
    except Exception as e:
        _note_error(errors, "ensemble", e)
        return None


def _note_error(errors, key, exc) -> None:
    """Record a scoring-phase exception so a failed report isn't a silent None."""
    if errors is not None:
        errors[key] = f"{type(exc).__name__}: {exc}"[:300]


def _holdout_report(
    search, df, target, task_type, metric, aux_tables, seed, errors=None
):
    """Score the incumbent on the SHARED seeded holdout (the cross-method bench).

    Uses the same split + scorer as the top-k ensemble (`ensemble.holdout_split`
    / `evaluate_top_k` with k=1), so this number is directly comparable to the
    AutoGluon and MLE-STAR baselines, which score on the identical rows. The
    extension's `report.score` is a full-data CV mean and is NOT comparable —
    this holdout number is the apples-to-apples one.

    Returns ``{"scorer": ..., "score": ...}`` or ``None`` on any failure.
    """
    scorer = metrics.report_scorer(metric)
    if scorer is None or scorer not in ensemble._METRIC_FNS:
        return None
    try:
        res = ensemble.evaluate_top_k(
            search["plan"],
            [search["best_state"]],
            df,
            target,
            task_type,
            scoring=scorer,
            aux=aux_tables or None,
            seed=seed,
        )
        return {"scorer": scorer, "score": res["individual_scores"][0]}
    except Exception as e:
        _note_error(errors, "holdout", e)
        return None


def _report(
    plan,
    state: dict,
    df,
    metric: str,
    aux: dict | None = None,
    stratify: bool = False,
    errors=None,
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
    except Exception as e:
        _note_error(errors, "report", e)
        return None


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the agent->MCTS pipeline."
    )
    parser.add_argument("--task", default=None, help="task name under data_dir")
    parser.add_argument("--target", default=None, help="override target column")
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="MCTS evaluations per phase (20 small / 80 large; a further "
        "ceil(budget/4) focused-refinement rollouts run after)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="artifact dir (default: runs/<task>_<timestamp>)",
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
        help="enable LLM whole-plan extending injection (Option 3); needs --outer-steps>1",
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
        help="max Caruana ensemble size selected from the score cache (1 = off)",
    )
    parser.add_argument(
        "--ensemble-pool",
        type=int,
        default=10,
        help="candidate-pool size for the Caruana ensemble (>= 2*top-k used)",
    )
    parser.add_argument(
        "--c", type=float, default=0.5, help="UCT exploration constant"
    )
    parser.add_argument(
        "--no-retarget",
        action="store_true",
        help="keep the first Option-1 target stage for the whole run",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="global random seed"
    )
    parser.add_argument(
        "--subsample-seeds",
        type=int,
        default=None,
        help="average each rollout over N seeded subsamples (default: auto — "
        "3 for imbalanced classification, else 1)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=6,
        help="CV fold-parallelism per rollout (default 6; safe with boosters)",
    )
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=None,
        help="wall-clock budget for the whole search (e.g. 3600 for 1h); "
        "--budget becomes an upper bound. LLM cost stays constant.",
    )
    args = parser.parse_args()

    task = args.task or config.CONFIG.task_name
    out_dir = args.out or default_out_dir(task)
    outer_steps, refine = args.outer_steps, args.refine
    if args.n_proposes is not None:  # sugar: N propose calls between slices
        outer_steps, refine = args.n_proposes + 1, args.n_proposes > 0
    propose = make_llm_proposer(log_dir=out_dir) if refine else None
    result = run_pipeline(
        task_name=args.task,
        target=args.target,
        budget=args.budget,
        out_dir=out_dir,
        seed=args.seed,
        outer_steps=outer_steps,
        propose=propose,
        top_k=args.top_k,
        ensemble_pool=args.ensemble_pool,
        c=args.c,
        retarget=not args.no_retarget,
        subsample_seeds=args.subsample_seeds,
        n_jobs=args.n_jobs,
        time_budget_s=args.time_budget_s,
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
        print(
            f"Report ({result['report']['scorer']}): {result['report']['score']:.4f}"
        )
    if result.get("ensemble"):
        ens = result["ensemble"]
        print(
            f"Top-{ens['k']} ensemble ({ens['scorer']}): {ens['ensemble_score']:.4f} "
            f"(incumbent {ens['individual_scores'][0]:.4f})"
        )
    if result.get("target_key"):
        print(f"Targeted stage: {result['target_key']}")
    if result.get("injected_options"):
        print(
            f"Injected options (not in original plan): {result['injected_options']}"
        )
    print(
        f"\nArtifacts written to: {out_dir}/ "
        "(summary.md, result.json, <agent>_<phase>.json)"
    )


if __name__ == "__main__":
    _main()
