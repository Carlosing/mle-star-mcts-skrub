"""Simplified debug utilities for the OpenAI-compatible runner."""

from machine_learning_engineering.runner import run_agent
from machine_learning_engineering.shared_libraries import (
    check_leakage_util,
    code_util,
    common_util,
    debug_prompt,
)


def _filename_from_agent_name(state, agent_name: str) -> str:
    """Infer the script filename from the agent name for debugging messages."""
    suffix = code_util.get_updated_suffix(state, agent_name)
    if agent_name.startswith("model_eval"):
        model_id = agent_name.split("_")[-1]
        return f"init_code_{model_id}"
    elif agent_name.startswith("merger"):
        reference_idx = agent_name.split("_")[-1]
        return f"train0_{reference_idx}"
    elif agent_name.startswith("check_data_use"):
        return "train0"
    elif agent_name.startswith("ablation"):
        task_id = agent_name.split("_")[-1]
        step = state.get(f"refine_step_{task_id}", 0)
        return f"ablation_{step}"
    elif agent_name.startswith("plan_implement"):
        task_id = agent_name.split("_")[-1]
        step = state.get(f"refine_step_{task_id}", 0)
        inner_iter = state.get(f"inner_iter_{task_id}", 0)
        return f"train{step}_improve{inner_iter}"
    elif agent_name.startswith("ensemble_plan_implement"):
        return f"ensemble{suffix}"
    elif agent_name.startswith("submission"):
        return "final_solution"
    return "script"


def get_code_from_response(state, agent_name, response, do_eval: bool = True) -> None:
    """Extract code from LLM response, store it in state, and optionally evaluate it.

    Args:
        state: AgentState instance.
        agent_name: Name of the current agent.
        response: OpenAI chat completion response.
        do_eval: If True, execute the extracted code.

    Returns:
        None.
    """
    response_text = common_util.get_text_from_response(response)
    code = response_text.replace("```python", "").replace("```", "")
    suffix = code_util.get_updated_suffix(state, agent_name)
    code_state_key = code_util.get_code_state_key(agent_name, suffix)

    if agent_name.startswith("check_data_use"):
        if "All the provided information is used" in code:
            state[f"check_data_use_finish_{suffix}"] = True
        check_data_use_finish = state.get(f"check_data_use_finish_{suffix}", False)
        if check_data_use_finish:
            return
        new_code = code
    elif agent_name.startswith("plan_implement"):
        # The LLM returns the complete improved solution for the plan.
        new_code = code
        if "debug" not in agent_name:
            task_id = agent_name.split("_")[-1]
            step = state.get(f"refine_step_{task_id}", 0)
            inner_iter = state.get(f"inner_iter_{task_id}", 0)
            # Keep the current working solution in sync with the latest improvement.
            state[f"train_code_{step}_{task_id}"] = new_code
            state[f"train_code_improve_{inner_iter}_{step}_{task_id}"] = new_code
    else:
        new_code = code

    state[code_state_key] = new_code
    if do_eval:
        code_util.evaluate_code(state, agent_name)


def _get_result(state, agent_name: str) -> dict:
    """Get the execution result dict for an agent from state."""
    suffix = code_util.get_updated_suffix(state, agent_name)
    result_key = code_util.get_code_execution_result_state_key(agent_name, suffix)
    return state.get(result_key, {})


def _is_successful(result: dict) -> bool:
    """Return True if the result indicates successful execution with a score."""
    return result.get("returncode", 1) == 0 and result.get("score") is not None


def run_and_debug(
    state,
    agent_name: str,
    instruction,
    before_model=None,
    max_retry: int = 1,
    max_debug: int = 1,
) -> None:
    """Generate code, execute it, and debug on failure.

    Args:
        state: AgentState instance.
        agent_name: Name of the agent.
        instruction: Prompt string or callable(state, agent_name) -> str.
        before_model: Optional before-model callback.
        max_retry: Number of generation attempts before giving up.
        max_debug: Number of debug attempts per generation attempt.

    Returns:
        None. The final code and execution result are stored in state.
    """
    for attempt in range(max_retry):
        run_agent(
            state,
            agent_name,
            instruction,
            before_model=before_model,
            after_model=lambda s, n, r: get_code_from_response(s, n, r, do_eval=True),
            temperature=1.0,
        )

        result = _get_result(state, agent_name)
        if _is_successful(result):
            check_leakage_util.check_and_fix_leakage(
                state,
                agent_name,
                max_iterations=state.get("max_retry", 1),
            )
            return

        # Debug loop: summarize error and ask LLM to fix it.
        for debug_round in range(max_debug):
            result = _get_result(state, agent_name)
            bug = result.get("stderr", "")
            if not bug:
                break

            filename = _filename_from_agent_name(state, agent_name)
            suffix = code_util.get_updated_suffix(state, agent_name)
            code_state_key = code_util.get_code_state_key(agent_name, suffix)
            code = state.get(code_state_key, "")

            debug_instruction = debug_prompt.BUG_REFINE_INSTR.format(
                task_description=state.get("task_description", ""),
                code=code,
                bug=bug,
            )

            debug_agent_name = f"{agent_name}_debug_{debug_round}"
            run_agent(
                state,
                debug_agent_name,
                debug_instruction,
                after_model=lambda s, n, r: get_code_from_response(
                    s, agent_name, r, do_eval=True
                ),
                temperature=0.0,
            )

            result = _get_result(state, agent_name)
            if _is_successful(result):
                return
