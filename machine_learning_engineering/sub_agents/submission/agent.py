"""Submission agent for Machine Learning Engineering (OpenAI-compatible MVP)."""

import os
import shutil

from machine_learning_engineering.shared_libraries import common_util, debug_util
from machine_learning_engineering.sub_agents.submission import prompt


def _create_submission_workspace(state) -> None:
    """Create the ensemble workspace directory used by the submission agent.

    Copies the task input files into the ensemble workspace so the final
    solution can load the test data from ./input.
    """
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


def get_submission_and_debug_agent_instruction(state, agent_name: str) -> str:
    """Builds the submission prompt using the best available solution."""
    task_description = state.get("task_description", "")
    # In the MVP the best solution is always train_code_0_1.
    final_solution = state.get("train_code_0_1", "")
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
