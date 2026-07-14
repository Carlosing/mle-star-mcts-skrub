"""Initialization agent for Machine Learning Engineering (OpenAI-compatible MVP)."""

import ast
import dataclasses
import os
import re
import shutil
import time

from machine_learning_engineering.runner import run_agent
from machine_learning_engineering.shared_libraries import (
    common_util,
    config,
    debug_util,
    web_search_util,
)
from machine_learning_engineering.sub_agents.initialization import prompt


def prepare_task(state) -> None:
    """Copies config into state and reads the task description."""
    config_dict = dataclasses.asdict(config.CONFIG)
    for key in config_dict:
        state[key] = config_dict[key]
    state["start_time"] = time.time()
    task_name = state.get("task_name", "")
    data_dir = state.get("data_dir", "")
    task_description_path = os.path.join(data_dir, task_name, "task_description.txt")
    with open(task_description_path, "r", encoding="utf-8") as f:
        task_description = f.read()
    state["task_description"] = task_description


def create_workspace(state, task_id) -> None:
    """Creates the workspace directory for a task and copies input data."""
    data_dir = state.get("data_dir", "")
    workspace_dir = state.get("workspace_dir", "")
    task_name = state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name, str(task_id))
    if os.path.exists(run_cwd):
        shutil.rmtree(run_cwd)
    os.makedirs(os.path.join(workspace_dir, task_name, str(task_id)), exist_ok=True)
    os.makedirs(
        os.path.join(workspace_dir, task_name, str(task_id), "input"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(workspace_dir, task_name, str(task_id), "model_candidates"),
        exist_ok=True,
    )
    files = os.listdir(os.path.join(data_dir, task_name))
    for file in files:
        if os.path.isdir(os.path.join(data_dir, task_name, file)):
            shutil.copytree(
                os.path.join(data_dir, task_name, file),
                os.path.join(workspace_dir, task_name, str(task_id), "input", file),
            )
        elif "answer" not in file:
            common_util.copy_file(
                os.path.join(data_dir, task_name, file),
                os.path.join(workspace_dir, task_name, str(task_id), "input"),
            )


def get_task_summary(state, agent_name, response) -> None:
    """Stores the task summary from the LLM response."""
    response_text = common_util.get_text_from_response(response)
    task_type = state.get("task_type", "Unknown Task")
    task_summary = f"Task: {task_type}\n{response_text}"
    state["task_summary"] = task_summary


def run_web_search_for_task(state, task_id) -> None:
    """Search the web for recent models/approaches for this task.

    Only runs when use_web_search is enabled in config. Search failures are
    swallowed so that the rest of the pipeline can continue without web data.
    """
    if not state.get("use_web_search", False):
        return

    task_summary = state.get("task_summary", "")
    task_type = state.get("task_type", "")
    query = f"effective machine learning models for {task_type} {task_summary}".strip()
    num_results = state.get("web_search_num_results", 5)
    results = web_search_util.search_web(query, num_results=num_results)
    state[f"web_search_results_{task_id}"] = results
    state[f"web_search_query_{task_id}"] = query


_DEFAULT_CANDIDATES = [
    {
        "model_name": "RandomForestRegressor",
        "example_code": (
            "from sklearn.ensemble import RandomForestRegressor\n"
            "model = RandomForestRegressor(n_estimators=100, random_state=42)"
        ),
    },
    {
        "model_name": "GradientBoostingRegressor",
        "example_code": (
            "from sklearn.ensemble import GradientBoostingRegressor\n"
            "model = GradientBoostingRegressor(random_state=42)"
        ),
    },
]


def _parse_model_candidates(response_text: str, num_model_candidates: int):
    """Parse model candidates from an LLM response.

    Tries several heuristics to handle imperfect JSON/markdown from small models.
    """
    # First attempt: extract a JSON/markdown list and parse it.
    start_idx, end_idx = response_text.find("["), response_text.rfind("]") + 1
    if start_idx != -1 and end_idx > start_idx:
        snippet = response_text[start_idx:end_idx]
        try:
            models = ast.literal_eval(snippet)
            if isinstance(models, list):
                return models[:num_model_candidates]
        except Exception:
            pass

    # Second attempt: look for code blocks and treat each as a candidate.
    code_blocks = re.findall(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    models = []
    for block in code_blocks[:num_model_candidates]:
        # Try to infer a model name from imports.
        name_match = re.search(r"from\s+sklearn\.\w+\s+import\s+(\w+)", block)
        name = name_match.group(1) if name_match else "Model"
        models.append({"model_name": name, "example_code": block})
    if models:
        return models

    return []


def get_model_candidates(state, agent_name, response, task_id) -> None:
    """Parses model candidates from the LLM response and stores them in state.

    The model retrieval prompt may include recent web search results when
    use_web_search is enabled. The response is expected to follow the JSON
    schema defined in MODEL_RETRIEVAL_INSTR.
    """
    workspace_dir = state.get("workspace_dir", "")
    task_name = state.get("task_name", "")
    num_model_candidates = state.get("num_model_candidates", 1)
    run_cwd = os.path.join(workspace_dir, task_name, str(task_id))
    response_text = common_util.get_text_from_response(response)
    state[f"model_retriever_raw_response_{task_id}"] = response_text

    models = _parse_model_candidates(response_text, num_model_candidates)

    # Fallback to default candidates if parsing produced nothing.
    if not models:
        models = _DEFAULT_CANDIDATES[:num_model_candidates]

    for j, model in enumerate(models):
        model_description = ""
        model_description += "## Model name\n"
        model_description += model["model_name"]
        model_description += "\n\n"
        model_description += "## Example Python code\n"
        model_description += model["example_code"]
        state[f"init_{task_id}_model_{j + 1}"] = {
            "model_name": model["model_name"],
            "example_code": model["example_code"],
            "model_description": model_description,
        }
        with open(
            os.path.join(run_cwd, "model_candidates", f"model_{j + 1}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(model_description)
    state[f"init_{task_id}_model_finish"] = bool(models)


def get_model_retriever_agent_instruction(state, agent_name: str) -> str:
    """Builds the model retrieval prompt."""
    task_summary = state.get("task_summary", "")
    num_model_candidates = state.get("num_model_candidates", 1)
    task_id = agent_name.split("_")[-1]
    results = state.get(f"web_search_results_{task_id}", [])
    web_search_results = web_search_util.format_results_for_prompt(results)
    return prompt.MODEL_RETRIEVAL_INSTR.format(
        task_summary=task_summary,
        num_model_candidates=num_model_candidates,
        web_search_results=web_search_results,
    )


def get_model_eval_agent_instruction(state, agent_name: str) -> str:
    """Builds the model evaluation prompt."""
    task_description = state.get("task_description", "")
    model_id = agent_name.split("_")[-1]
    task_id = agent_name.split("_")[-2]
    model_description = state.get(
        f"init_{task_id}_model_{model_id}",
        {},
    ).get("model_description", "")
    return prompt.MODEL_EVAL_INSTR.format(
        task_description=task_description,
        model_description=model_description,
    )


def rank_candidate_solutions(state, task_id) -> None:
    """Selects the best initial solution and stores it as the base solution."""
    workspace_dir = state.get("workspace_dir", "")
    task_name = state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name, str(task_id))
    num_model_candidates = state.get("num_model_candidates", 1)
    performance_results = []
    for k in range(num_model_candidates):
        model_id = k + 1
        init_code = state.get(f"init_code_{task_id}_{model_id}", "")
        init_code_exec_result = state.get(
            f"init_code_exec_result_{task_id}_{model_id}", {}
        )
        if init_code_exec_result:
            performance_results.append(
                (
                    init_code_exec_result.get("score", 0.0),
                    init_code,
                    init_code_exec_result,
                    model_id,
                )
            )
    if not performance_results:
        return

    if state.get("lower", True):
        performance_results.sort(key=lambda x: x[0])
    else:
        performance_results.sort(key=lambda x: x[0], reverse=True)

    best_score = performance_results[0][0]
    base_solution = (
        performance_results[0][1].replace("```python", "").replace("```", "")
    )
    best_model_id = performance_results[0][3]
    state[f"performance_results_{task_id}"] = performance_results
    state[f"best_score_{task_id}"] = best_score
    state[f"base_solution_{task_id}"] = base_solution
    state[f"best_model_id_{task_id}"] = best_model_id
    state[f"best_idx_{task_id}"] = 0

    # Write the best solution as the starting train0_0.py
    with open(f"{run_cwd}/train0_0.py", "w", encoding="utf-8") as f:
        f.write(base_solution)

    state[f"merger_code_{task_id}_0"] = performance_results[0][1]
    state[f"merger_code_exec_result_{task_id}_0"] = performance_results[0][2]


def select_best_solution(state, task_id) -> None:
    """Promotes the best base solution to train_code_0 for downstream agents."""
    workspace_dir = state.get("workspace_dir", "")
    task_name = state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name, str(task_id))
    best_idx = state.get(f"best_idx_{task_id}", 0)
    response = state.get(f"merger_code_{task_id}_{best_idx}", "")
    result_dict = state.get(f"merger_code_exec_result_{task_id}_{best_idx}", {})

    # If the selected merged solution did not run successfully, fall back to the
    # best individual model candidate that did run.
    if not result_dict or result_dict.get("returncode", 1) != 0:
        performance_results = state.get(f"performance_results_{task_id}", [])
        if performance_results:
            response = performance_results[0][1]
            result_dict = performance_results[0][2]

    code_text = response.replace("```python", "").replace("```", "")
    output_filepath = os.path.join(run_cwd, "train0.py")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(code_text)
    state[f"train_code_0_{task_id}"] = code_text
    state[f"train_code_exec_result_0_{task_id}"] = result_dict
    # Save the best initial candidate as a safe fallback for submission.
    state["best_initial_code"] = code_text
    state["best_initial_score"] = result_dict.get("score")


def run_model_eval_for_all_candidates(state, task_id) -> None:
    """Evaluate every model candidate proposed by the retriever."""
    num_model_candidates = state.get("num_model_candidates", 1)
    for model_id in range(1, num_model_candidates + 1):
        if not state.get(f"init_{task_id}_model_{model_id}"):
            continue
        debug_util.run_and_debug(
            state,
            f"model_eval_{task_id}_{model_id}",
            get_model_eval_agent_instruction,
            max_retry=state.get("max_retry", 1),
            max_debug=state.get("max_debug_round", 1),
        )


def get_merger_agent_instruction(state, agent_name: str) -> str:
    """Build the merger prompt that integrates a reference into the base."""
    reference_idx = int(agent_name.split("_")[-1])
    task_id = agent_name.split("_")[-2]
    base_code = state.get(f"base_solution_{task_id}", "")
    reference_code = state.get(
        f"init_code_{task_id}_{reference_idx}",
        "",
    )
    return prompt.CODE_INTEGRATION_INSTR.format(
        base_code=base_code,
        reference_code=reference_code,
    )


def run_merger_and_debug_loop(state, task_id) -> None:
    """Merge the best candidate with every other candidate and keep the best."""
    num_model_candidates = state.get("num_model_candidates", 1)
    if num_model_candidates < 2:
        return

    lower = state.get("lower", True)
    best_score = state.get(f"best_score_{task_id}", 1e9 if lower else 0)
    best_idx = 0
    best_model_id = state.get(f"best_model_id_{task_id}", 1)

    for reference_idx in range(1, num_model_candidates + 1):
        if reference_idx == best_model_id:
            # The base solution is already stored as merger_code_{task_id}_0.
            continue
        if not state.get(f"init_code_{task_id}_{reference_idx}", ""):
            continue
        debug_util.run_and_debug(
            state,
            f"merger_{task_id}_{reference_idx}",
            get_merger_agent_instruction,
            max_retry=state.get("max_retry", 1),
            max_debug=state.get("max_debug_round", 1),
        )
        result = state.get(f"merger_code_exec_result_{task_id}_{reference_idx}", {})
        score = result.get("score", 1e9 if lower else 0)
        if (lower and score < best_score) or (not lower and score > best_score):
            best_score = score
            best_idx = reference_idx

    state[f"best_score_{task_id}"] = best_score
    state[f"best_idx_{task_id}"] = best_idx


def get_check_data_use_agent_instruction(state, agent_name: str) -> str:
    """Build the data-use checker prompt."""
    task_id = agent_name.split("_")[-1]
    code = state.get(f"train_code_0_{task_id}", "")
    return prompt.CHECK_DATA_USE_INSTR.format(
        code=code,
        task_description=state.get("task_description", ""),
    )


def run_check_data_use_and_debug_loop(state, task_id) -> None:
    """Verify the selected solution uses all provided information."""
    if not state.get("use_data_usage_checker", False):
        return
    debug_util.run_and_debug(
        state,
        f"check_data_use_{task_id}",
        get_check_data_use_agent_instruction,
        max_retry=state.get("max_retry", 1),
        max_debug=state.get("max_debug_round", 1),
    )


def run_initialization_pipeline(state) -> None:
    """Runs the full initialization phase for the MVP.

    Args:
        state: AgentState instance.

    Returns:
        None.
    """
    task_id = 1
    prepare_task(state)
    create_workspace(state, task_id)

    # Summarize task
    summarization_instruction = prompt.SUMMARIZATION_AGENT_INSTR.format(
        task_description=state.get("task_description", ""),
        task_type=state.get("task_type", "Unknown Task"),
    )
    run_agent(
        state,
        "task_summarization_agent",
        summarization_instruction,
        after_model=get_task_summary,
        temperature=0.0,
    )

    # Optional web search before model retrieval
    run_web_search_for_task(state, task_id)

    # Retrieve/propose models
    run_agent(
        state,
        f"model_retriever_agent_{task_id}",
        get_model_retriever_agent_instruction,
        after_model=lambda s, n, r: get_model_candidates(s, n, r, task_id),
        temperature=1.0,
    )

    # Evaluate every proposed model candidate
    run_model_eval_for_all_candidates(state, task_id)

    # Rank candidates and select the best base solution
    rank_candidate_solutions(state, task_id)

    # Merge the best candidate with the remaining candidates
    run_merger_and_debug_loop(state, task_id)

    # Promote the best merged (or base) solution for downstream agents
    select_best_solution(state, task_id)

    # Ensure all provided information is used
    run_check_data_use_and_debug_loop(state, task_id)
