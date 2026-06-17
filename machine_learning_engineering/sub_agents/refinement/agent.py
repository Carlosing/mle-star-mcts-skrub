"""Refinement agent for MLE-STAR (OpenAI-compatible runner)."""

import json
import re

from machine_learning_engineering.runner import run_agent
from machine_learning_engineering.shared_libraries import code_util, common_util, debug_util
from machine_learning_engineering.sub_agents.refinement import prompt


def _get_current_solution(state, task_id, step):
    """Return the current refined solution for the given step."""
    return state.get(f"train_code_{step}_{task_id}", "")


def _set_current_solution(state, task_id, step, code):
    """Store the current refined solution for the given step."""
    state[f"train_code_{step}_{task_id}"] = code


_DEFAULT_PLAN = "Simplify the model training block and use a more robust validation setup."
_DEFAULT_CODE_BLOCK = ""


def _extract_json_block(text: str):
    """Extract the first JSON object from a text string."""
    # Try to find a JSON object delimited by braces.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def get_ablation_agent_instruction(state, agent_name: str) -> str:
    """Build the ablation study prompt."""
    task_id = agent_name.split("_")[-1]
    code = _get_current_solution(state, task_id, 0)
    return prompt.ABLATION_INSTR.format(
        task_description=state.get("task_description", ""),
        code=code,
    )


def get_ablation_summary_agent_instruction(state, agent_name: str) -> str:
    """Build the ablation summary prompt."""
    task_id = agent_name.split("_")[-1]
    step = state.get(f"refine_step_{task_id}", 0)
    ablation_result = state.get(f"ablation_code_exec_result_{step}_{task_id}", {})
    ablation_output = ablation_result.get("ablation_result", "")
    return prompt.ABLATION_SUMMARY_INSTR.format(
        task_description=state.get("task_description", ""),
        ablation_output=ablation_output,
    )


def get_init_plan_agent_instruction(state, agent_name: str) -> str:
    """Build the initial refinement plan prompt."""
    task_id = agent_name.split("_")[-1]
    step = state.get(f"refine_step_{task_id}", 0)
    code = _get_current_solution(state, task_id, step)
    ablation_summary = state.get(f"ablation_summary_{step}_{task_id}", "")
    return prompt.INIT_PLAN_INSTR.format(
        task_description=state.get("task_description", ""),
        code=code,
        ablation_summary=ablation_summary,
    )


def get_plan_refine_agent_instruction(state, agent_name: str) -> str:
    """Build the alternative refinement plan prompt."""
    task_id = agent_name.split("_")[-1]
    step = state.get(f"refine_step_{task_id}", 0)
    code = _get_current_solution(state, task_id, step)
    previous_plans = state.get(f"previous_plans_{step}_{task_id}", [])
    return prompt.PLAN_REFINE_INSTR.format(
        task_description=state.get("task_description", ""),
        code=code,
        previous_plans="\n".join(previous_plans),
    )


def get_plan_implement_agent_instruction(state, agent_name: str) -> str:
    """Build the plan implementation prompt."""
    task_id = agent_name.split("_")[-1]
    step = state.get(f"refine_step_{task_id}", 0)
    inner_iter = state.get(f"inner_iter_{task_id}", 0)
    code = _get_current_solution(state, task_id, step)
    plan = state.get(f"refine_plan_{inner_iter}_{step}_{task_id}", "")
    code_block = state.get(f"refine_code_block_{inner_iter}_{step}_{task_id}", "")
    return prompt.PLAN_IMPLEMENT_INSTR.format(
        task_description=state.get("task_description", ""),
        code=code,
        plan=plan,
        code_block=code_block,
    )


def _store_plan_from_response(state, agent_name, response) -> None:
    """Parse a plan response and store plan/code_block."""
    response_text = common_util.get_text_from_response(response)
    parsed = _extract_json_block(response_text)
    task_id = agent_name.split("_")[-1]
    step = state.get(f"refine_step_{task_id}", 0)
    inner_iter = state.get(f"inner_iter_{task_id}", 0)
    plan = parsed.get("plan", "") or _DEFAULT_PLAN
    code_block = parsed.get("code_block", "") or _DEFAULT_CODE_BLOCK
    state[f"refine_plan_{inner_iter}_{step}_{task_id}"] = plan
    state[f"refine_code_block_{inner_iter}_{step}_{task_id}"] = code_block
    previous_plans = state.get(f"previous_plans_{step}_{task_id}", [])
    previous_plans.append(plan)
    state[f"previous_plans_{step}_{task_id}"] = previous_plans


def _store_ablation_summary(state, agent_name, response) -> None:
    """Store the ablation summary in state."""
    response_text = common_util.get_text_from_response(response)
    task_id = agent_name.split("_")[-1]
    step = state.get(f"refine_step_{task_id}", 0)
    state[f"ablation_summary_{step}_{task_id}"] = response_text


def _select_best_improvement(state, task_id, step, inner_iters) -> str:
    """Select the best improvement from the inner loop."""
    lower = state.get("lower", True)
    best_code = _get_current_solution(state, task_id, step)
    best_score = state.get(
        f"train_code_exec_result_0_{task_id}", {}
    ).get("score", 1e9 if lower else 0)

    for inner_iter in inner_iters:
        result = state.get(
            f"train_code_improve_exec_result_{inner_iter}_{step}_{task_id}", {}
        )
        score = result.get("score", 1e9 if lower else 0)
        if (lower and score < best_score) or (not lower and score > best_score):
            best_score = score
            best_code = state.get(
                f"train_code_improve_{inner_iter}_{step}_{task_id}", best_code
            )

    _set_current_solution(state, task_id, step + 1, best_code)
    # Track the best refined solution seen so far across all steps.
    current_best_score = state.get("best_refined_score", 1e9 if lower else 0)
    if (lower and best_score < current_best_score) or (
        not lower and best_score > current_best_score
    ):
        state["best_refined_score"] = best_score
        state["best_refined_code"] = best_code
    return best_code


def run_ablation_and_debug_loop(state, task_id) -> None:
    """Generate and execute an ablation study script."""
    debug_util.run_and_debug(
        state,
        f"ablation_{task_id}",
        get_ablation_agent_instruction,
        max_retry=state.get("max_retry", 1),
        max_debug=state.get("max_debug_round", 1),
    )


def run_ablation_summary_agent(state, task_id) -> None:
    """Summarize ablation study results."""
    run_agent(
        state,
        f"ablation_summary_{task_id}",
        get_ablation_summary_agent_instruction,
        after_model=_store_ablation_summary,
        temperature=0.0,
    )


def run_init_plan_agent(state, task_id) -> None:
    """Generate the initial refinement plan."""
    run_agent(
        state,
        f"init_plan_agent_{task_id}",
        get_init_plan_agent_instruction,
        after_model=_store_plan_from_response,
        temperature=1.0,
    )


def run_init_plan_implement_agent(state, task_id) -> None:
    """Apply the initial refinement plan."""
    state[f"inner_iter_{task_id}"] = 0
    debug_util.run_and_debug(
        state,
        f"plan_implement_{task_id}",
        get_plan_implement_agent_instruction,
        max_retry=state.get("max_retry", 1),
        max_debug=state.get("max_debug_round", 1),
    )


def run_plan_refine_agent(state, task_id, inner_iter: int) -> None:
    """Generate an alternative refinement plan."""
    state[f"inner_iter_{task_id}"] = inner_iter
    run_agent(
        state,
        f"plan_refine_{inner_iter}_{task_id}",
        get_plan_refine_agent_instruction,
        after_model=_store_plan_from_response,
        temperature=1.0,
    )


def run_plan_implement_agent(state, task_id, inner_iter: int) -> None:
    """Apply an alternative refinement plan."""
    state[f"inner_iter_{task_id}"] = inner_iter
    debug_util.run_and_debug(
        state,
        f"plan_implement_{task_id}",
        get_plan_implement_agent_instruction,
        max_retry=state.get("max_retry", 1),
        max_debug=state.get("max_debug_round", 1),
    )


def run_refinement_pipeline(state) -> None:
    """Run the full refinement pipeline for each solution."""
    num_solutions = state.get("num_solutions", 1)
    outer_loop_round = state.get("outer_loop_round", 0)
    inner_loop_round = state.get("inner_loop_round", 0)

    for task_id in range(1, num_solutions + 1):
        # Seed the refinement step 0 with the initialization output.
        init_code = state.get(f"train_code_0_{task_id}", "")
        _set_current_solution(state, task_id, 0, init_code)
        init_score = state.get(f"train_code_exec_result_0_{task_id}", {}).get(
            "score"
        )
        lower = state.get("lower", True)
        state["best_refined_score"] = (
            init_score if init_score is not None else (1e9 if lower else 0)
        )
        state["best_refined_code"] = init_code

        for step in range(outer_loop_round):
            state[f"refine_step_{task_id}"] = step

            # Ablation study and summary
            run_ablation_and_debug_loop(state, task_id)
            run_ablation_summary_agent(state, task_id)

            # Initial plan and implementation
            run_init_plan_agent(state, task_id)
            run_init_plan_implement_agent(state, task_id)

            inner_iters = [0]
            # Alternative plans in the inner loop
            for inner_iter in range(1, inner_loop_round + 1):
                run_plan_refine_agent(state, task_id, inner_iter)
                run_plan_implement_agent(state, task_id, inner_iter)
                inner_iters.append(inner_iter)

            # Select the best improvement and promote it to the next step
            _select_best_improvement(state, task_id, step, inner_iters)

        # Store the final refined code for downstream agents
        state[f"train_code_0_{task_id}"] = state.get(
            "best_refined_code",
            _get_current_solution(state, task_id, outer_loop_round),
        )
