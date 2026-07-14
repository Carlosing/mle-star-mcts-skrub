"""OpenAI-compatible data leakage checker utilities."""

import json
import re

from machine_learning_engineering.runner import run_agent
from machine_learning_engineering.shared_libraries import (
    code_util,
    common_util,
    data_leakage_prompt,
)


def _get_code_for_agent(state, agent_name: str) -> str:
    """Get the current code block associated with an agent."""
    suffix = code_util.get_updated_suffix(state, agent_name)
    code_state_key = code_util.get_code_state_key(agent_name, suffix)
    return state.get(code_state_key, "")


def _set_code_for_agent(state, agent_name: str, code: str) -> None:
    """Store the updated code block associated with an agent."""
    suffix = code_util.get_updated_suffix(state, agent_name)
    code_state_key = code_util.get_code_state_key(agent_name, suffix)
    state[code_state_key] = code


def _parse_leakage_response(text: str) -> tuple[str, str]:
    """Parse leakage status and code block from an LLM response.

    Returns:
        Tuple of (leakage_status, code_block). If parsing fails, defaults to
        "No Data Leakage" and an empty code block.
    """
    # Try to find a JSON list/object in the response.
    start_idx, end_idx = text.find("["), text.rfind("]") + 1
    if start_idx == -1 or end_idx == 0:
        start_idx, end_idx = text.find("{"), text.rfind("}") + 1
    if start_idx == -1 or end_idx == 0:
        return "No Data Leakage", ""

    snippet = text[start_idx:end_idx]
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, list):
            parsed = parsed[0]
        leakage_status = parsed.get("leakage_status", "No Data Leakage")
        code_block = parsed.get("code_block", "")
        code_block = code_block.replace("```python", "").replace("```", "")
        return leakage_status, code_block
    except Exception:
        return "No Data Leakage", ""


def _check_leakage_status(state, agent_name: str) -> tuple[str, str]:
    """Run the leakage checker agent and parse the result."""
    code = _get_code_for_agent(state, agent_name)
    if not code:
        return "No Data Leakage", ""

    instruction = data_leakage_prompt.CHECK_LEAKAGE_INSTR.format(code=code)
    response = run_agent(
        state,
        f"check_leakage_agent_{agent_name}",
        instruction,
        temperature=0.0,
    )
    response_text = common_util.get_text_from_response(response)
    return _parse_leakage_response(response_text)


def _refine_leakage_block(state, agent_name: str, leakage_block: str) -> str:
    """Ask the LLM to refine a leaky code block."""
    instruction = data_leakage_prompt.LEAKAGE_REFINE_INSTR.format(
        code=leakage_block
    )
    response = run_agent(
        state,
        f"refine_leakage_agent_{agent_name}",
        instruction,
        temperature=0.0,
    )
    response_text = common_util.get_text_from_response(response)
    refined = response_text.replace("```python", "").replace("```", "")
    return refined


def check_and_fix_leakage(
    state,
    agent_name: str,
    max_iterations: int = 2,
) -> bool:
    """Check an agent's code for data leakage and fix it when found.

    Args:
        state: AgentState instance.
        agent_name: Name of the agent whose code should be checked.
        max_iterations: Maximum leakage fix iterations.

    Returns:
        True if the code ended with no detected leakage, False otherwise.
    """
    if not state.get("use_data_leakage_checker", False):
        return True

    for _ in range(max_iterations):
        leakage_status, leakage_block = _check_leakage_status(state, agent_name)
        if leakage_status != "Yes Data Leakage" or not leakage_block:
            return True

        code = _get_code_for_agent(state, agent_name)
        if leakage_block not in code:
            # The reported block is not an exact substring; give up fixing.
            return False

        refined_block = _refine_leakage_block(state, agent_name, leakage_block)
        new_code = code.replace(leakage_block, refined_block, 1)
        _set_code_for_agent(state, agent_name, new_code)
        code_util.evaluate_code(state, agent_name)

        result = code_util.get_code_execution_result_state_key(
            agent_name, code_util.get_updated_suffix(state, agent_name)
        )
        exec_result = state.get(result, {})
        if exec_result.get("returncode", 1) != 0:
            return False

    leakage_status, _ = _check_leakage_status(state, agent_name)
    return leakage_status != "Yes Data Leakage"
