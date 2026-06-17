"""Submission agent for Machine Learning Engineering (OpenAI-compatible MVP)."""

import os
import shutil

from machine_learning_engineering.shared_libraries import common_util, debug_util
from machine_learning_engineering.sub_agents.submission import prompt


def _create_submission_workspace(state) -> None:
    """Create the ensemble workspace directory used by the submission agent.

    Copies the task input files into the ensemble workspace so the final
    solution can load the test data from ./input. If the workspace already
    exists (e.g., created by the ensemble agent), it is preserved.
    """
    data_dir = state.get("data_dir", "")
    workspace_dir = state.get("workspace_dir", "")
    task_name = state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name, "ensemble")
    os.makedirs(run_cwd, exist_ok=True)
    os.makedirs(os.path.join(run_cwd, "input"), exist_ok=True)
    os.makedirs(os.path.join(run_cwd, "final"), exist_ok=True)
    if os.listdir(os.path.join(run_cwd, "input")):
        return
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


def _select_final_solution(state) -> str:
    """Choose the best refined solution or ensemble output for submission."""
    best_ensemble_code = state.get("best_ensemble_code", "")
    best_ensemble_score = state.get("best_ensemble_score")
    refined_score = state.get("best_refined_score")
    refined_code = state.get("best_refined_code", "")
    initial_code = state.get("best_initial_code", refined_code)

    # Pick the best available solution among successful refinements/ensembles.
    best_code = refined_code
    best_score = refined_score
    if best_ensemble_code and best_ensemble_score is not None:
        lower = state.get("lower", True)
        if best_score is None or (
            (lower and best_ensemble_score <= best_score)
            or (not lower and best_ensemble_score >= best_score)
        ):
            best_code = best_ensemble_code
            best_score = best_ensemble_score

    # If the selected solution is broken, fall back to the best initial candidate.
    if not best_code or best_score is None or best_score == 1e9:
        best_code = initial_code
    return best_code


def get_submission_and_debug_agent_instruction(state, agent_name: str) -> str:
    """Builds the submission prompt using the best available solution."""
    task_description = state.get("task_description", "")
    final_solution = _select_final_solution(state)
    return prompt.ADD_TEST_FINAL_INSTR.format(
        task_description=task_description,
        code=final_solution,
    )


def run_submission_pipeline(state) -> None:
    """Generates the final solution and submission.csv.

    Args:
        state: AgentState instance.

    Returns:
        None.
    """
    _create_submission_workspace(state)
    debug_util.run_and_debug(
        state,
        "submission",
        get_submission_and_debug_agent_instruction,
        max_retry=state.get("max_retry", 1),
        max_debug=state.get("max_debug_round", 1),
    )
