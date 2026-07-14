"""Ensemble agent for MLE-STAR (OpenAI-compatible runner)."""

import json
import os
import re
import shutil

from machine_learning_engineering.runner import run_agent
from machine_learning_engineering.shared_libraries import code_util, common_util, debug_util
from machine_learning_engineering.sub_agents.ensemble import prompt


def _create_ensemble_workspace(state) -> None:
    """Create the ensemble workspace directory and copy input data."""
    data_dir = state.get("data_dir", "")
    workspace_dir = state.get("workspace_dir", "")
    task_name = state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name, "ensemble")
    if os.path.exists(run_cwd):
        shutil.rmtree(run_cwd)
    os.makedirs(run_cwd, exist_ok=True)
    os.makedirs(os.path.join(run_cwd, "input"), exist_ok=True)
    os.makedirs(os.path.join(run_cwd, "final"), exist_ok=True)
    files = os.listdir(os.path.join(data_dir, task_name))
    for file in files:
        if os.path.isdir(os.path.join(data_dir, task_name, file)):
            shutil.copytree(
                os.path.join(data_dir, task_name, file),
                os.path.join(run_cwd, "input", file),
            )
        elif "answer" not in file:
            common_util.copy_file(
                os.path.join(data_dir, task_name, file),
                os.path.join(run_cwd, "input"),
            )


def _extract_json_block(text: str):
    """Extract the first JSON object from a text string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _collect_solutions(state) -> str:
    """Collect refined solution code for each task_id."""
    num_solutions = state.get("num_solutions", 1)
    parts = []
    for task_id in range(1, num_solutions + 1):
        code = state.get(f"train_code_0_{task_id}", "")
        parts.append(f"# Solution {task_id}\n```python\n{code}\n```")
    return "\n\n".join(parts)


def _store_ensemble_plan(state, agent_name, response) -> None:
    """Parse and store an ensemble plan."""
    response_text = common_util.get_text_from_response(response)
    parsed = _extract_json_block(response_text)
    ensemble_iter = state.get("ensemble_iter", 0)
    plan = parsed.get("plan", "")
    state[f"ensemble_plan_{ensemble_iter}"] = plan
    previous_plans = state.get("previous_ensemble_plans", [])
    previous_plans.append(plan)
    state["previous_ensemble_plans"] = previous_plans


def get_init_ensemble_plan_agent_instruction(state, agent_name: str) -> str:
    """Build the initial ensemble plan prompt."""
    return prompt.INIT_ENSEMBLE_PLAN_INSTR.format(
        task_description=state.get("task_description", ""),
        solutions=_collect_solutions(state),
    )


def get_ensemble_plan_refine_agent_instruction(state, agent_name: str) -> str:
    """Build the refined ensemble plan prompt."""
    return prompt.ENSEMBLE_PLAN_REFINE_INSTR.format(
        task_description=state.get("task_description", ""),
        solutions=_collect_solutions(state),
        previous_plans="\n".join(state.get("previous_ensemble_plans", [])),
    )


def get_ensemble_plan_implement_agent_instruction(state, agent_name: str) -> str:
    """Build the ensemble implementation prompt."""
    ensemble_iter = state.get("ensemble_iter", 0)
    plan = state.get(f"ensemble_plan_{ensemble_iter}", "")
    return prompt.ENSEMBLE_PLAN_IMPLEMENT_INSTR.format(
        task_description=state.get("task_description", ""),
        plan=plan,
        solutions=_collect_solutions(state),
    )


def _select_best_ensemble(state) -> str:
    """Select the best ensemble implementation."""
    lower = state.get("lower", True)
    best_iter = 0
    best_score = 1e9 if lower else 0
    ensemble_loop_round = state.get("ensemble_loop_round", 0)

    for ensemble_iter in range(ensemble_loop_round + 1):
        result = state.get(f"ensemble_code_exec_result_{ensemble_iter}", {})
        score = result.get("score", 1e9 if lower else 0)
        if (lower and score < best_score) or (not lower and score > best_score):
            best_score = score
            best_iter = ensemble_iter

    best_code = state.get(f"ensemble_code_{best_iter}", "")
    state["best_ensemble_code"] = best_code
    state["best_ensemble_score"] = best_score
    state["best_ensemble_iter"] = best_iter
    return best_code


def run_init_ensemble_plan_agent(state) -> None:
    """Generate the initial ensemble plan."""
    run_agent(
        state,
        "init_ensemble_plan_agent",
        get_init_ensemble_plan_agent_instruction,
        after_model=_store_ensemble_plan,
        temperature=1.0,
    )


def run_init_ensemble_plan_implement_agent(state) -> None:
    """Implement the initial ensemble plan."""
    state["ensemble_iter"] = 0
    debug_util.run_and_debug(
        state,
        "ensemble_plan_implement",
        get_ensemble_plan_implement_agent_instruction,
        max_retry=state.get("max_retry", 1),
        max_debug=state.get("max_debug_round", 1),
    )


def run_ensemble_plan_refine_agent(state, ensemble_iter: int) -> None:
    """Generate a refined ensemble plan."""
    state["ensemble_iter"] = ensemble_iter
    run_agent(
        state,
        f"ensemble_plan_refine_{ensemble_iter}",
        get_ensemble_plan_refine_agent_instruction,
        after_model=_store_ensemble_plan,
        temperature=1.0,
    )


def run_ensemble_plan_implement_agent(state, ensemble_iter: int) -> None:
    """Implement a refined ensemble plan."""
    state["ensemble_iter"] = ensemble_iter
    debug_util.run_and_debug(
        state,
        "ensemble_plan_implement",
        get_ensemble_plan_implement_agent_instruction,
        max_retry=state.get("max_retry", 1),
        max_debug=state.get("max_debug_round", 1),
    )


def run_ensemble_pipeline(state) -> None:
    """Run the ensemble pipeline."""
    ensemble_loop_round = state.get("ensemble_loop_round", 0)
    if ensemble_loop_round < 1:
        return

    _create_ensemble_workspace(state)

    # Initial ensemble
    run_init_ensemble_plan_agent(state)
    run_init_ensemble_plan_implement_agent(state)

    # Refined ensembles
    for ensemble_iter in range(1, ensemble_loop_round + 1):
        run_ensemble_plan_refine_agent(state, ensemble_iter)
        run_ensemble_plan_implement_agent(state, ensemble_iter)

    _select_best_ensemble(state)
