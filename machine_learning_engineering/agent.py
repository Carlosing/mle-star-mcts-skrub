"""Entry point for the OpenAI-compatible MLE-STAR MVP."""

import dataclasses
import json
import os

from machine_learning_engineering.runner import AgentState
from machine_learning_engineering.shared_libraries import config
from machine_learning_engineering.sub_agents.ensemble import (
    agent as ensemble_agent_module,
)
from machine_learning_engineering.sub_agents.initialization import (
    agent as initialization_agent_module,
)
from machine_learning_engineering.sub_agents.refinement import (
    agent as refinement_agent_module,
)
from machine_learning_engineering.sub_agents.submission import (
    agent as submission_agent_module,
)


def save_state(state: AgentState) -> None:
    """Saves the final state to a JSON file in the task workspace."""
    workspace_dir = state.get("workspace_dir", "")
    task_name = state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name)
    os.makedirs(run_cwd, exist_ok=True)
    with open(
        os.path.join(run_cwd, "final_state.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(state.to_dict(), f, indent=2, default=str)


def run_pipeline(
    task_name: str | None = None,
    data_dir: str | None = None,
    workspace_dir: str | None = None,
) -> AgentState:
    """Runs the MLE-STAR MVP pipeline.

    Every field of ``config.CONFIG`` is seeded onto the state first. The
    sub-agents read their knobs from ``state`` (``state.get("max_debug_round")``,
    ``state.get("use_web_search")``, …), so anything left unseeded silently falls
    back to a hardcoded default — which previously meant ``use_web_search`` could
    never fire here, and the benchmark harness's clamps were ignored. Seeding
    from config makes ``config.CONFIG`` the single source of truth it reads like.

    Args:
        task_name: Optional override for the task name.
        data_dir: Optional override for the data directory.
        workspace_dir: Optional override for the workspace directory.

    Returns:
        AgentState containing the full pipeline state.
    """
    state = AgentState()
    cfg = config.CONFIG

    for field in dataclasses.fields(cfg):
        state[field.name] = getattr(cfg, field.name)

    state["task_name"] = task_name or cfg.task_name
    state["data_dir"] = data_dir or cfg.data_dir
    state["workspace_dir"] = workspace_dir or cfg.workspace_dir
    state["total_tokens_spent"] = 0

    initialization_agent_module.run_initialization_pipeline(state)
    refinement_agent_module.run_refinement_pipeline(state)
    ensemble_agent_module.run_ensemble_pipeline(state)
    submission_agent_module.run_submission_pipeline(state)
    save_state(state)

    return state


if __name__ == "__main__":
    final_state = run_pipeline()
    print("\n=== Pipeline finished ===")
    print(f"Best score: {final_state.get('best_score_1')}")
    submission_result = final_state.get("submission_code_exec_result", {})
    print(f"Submission return code: {submission_result.get('returncode')}")
    print(f"Total tokens spent: {final_state.get('total_tokens_spent', 0)}")
