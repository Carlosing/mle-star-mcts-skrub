"""Integration tests for the initialization pipeline with mocked LLM calls."""

import os
from types import SimpleNamespace

import pytest

from machine_learning_engineering.runner import AgentState
from machine_learning_engineering.shared_libraries import code_util, debug_util, web_search_util
from machine_learning_engineering.sub_agents.initialization import agent as init_agent


def _make_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=0),
    )


@pytest.fixture
def mock_responses():
    """Return deterministic LLM responses for each agent in initialization."""
    return {
        "task_summarization_agent": _make_response(
            "California housing regression task predicting median house value."
        ),
        "model_retriever_agent_1": _make_response(
            '[\n'
            '  {"model_name": "RandomForestRegressor", "example_code": "from sklearn.ensemble import RandomForestRegressor\\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)"},\n'
            '  {"model_name": "GradientBoostingRegressor", "example_code": "from sklearn.ensemble import GradientBoostingRegressor\\nmodel = GradientBoostingRegressor(random_state=42)"}\n'
            ']'
        ),
        "model_eval_1_1": _make_response(
            "```python\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
            "model = RandomForestRegressor()\n"
            "print('Final Validation Performance: 0.5')\n"
            "```"
        ),
        "model_eval_1_2": _make_response(
            "```python\n"
            "from sklearn.ensemble import GradientBoostingRegressor\n"
            "model = GradientBoostingRegressor()\n"
            "print('Final Validation Performance: 0.8')\n"
            "```"
        ),
        "merger_1_2": _make_response(
            "```python\n"
            "# merged solution that will fail execution\n"
            "raise RuntimeError('simulated merge failure')\n"
            "print('Final Validation Performance: 0.7')\n"
            "```"
        ),
    }


@pytest.fixture
def setup_mocks(monkeypatch, mock_responses):
    """Patch run_agent and run_python_code so the pipeline runs offline."""

    def fake_run_agent(state, agent_name, instruction, before_model=None, after_model=None, temperature=0.0):
        response = mock_responses.get(agent_name)
        if response is None:
            raise ValueError(f"No mock response configured for {agent_name}")
        if after_model is not None:
            after_model(state, agent_name, response)
        return response

    def fake_run_python_code(code_text, run_cwd, py_filepath, exec_timeout):
        # Simulate that the merged solution fails (no stderr -> debug loop exits immediately).
        if "train0_2" in py_filepath:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": "",
                "execution_time": 0.1,
            }

        if "init_code_1" in py_filepath:
            score = 0.5
        elif "init_code_2" in py_filepath:
            score = 0.8
        else:
            score = 0.0

        return {
            "returncode": 0,
            "stdout": f"Final Validation Performance: {score}",
            "stderr": "",
            "execution_time": 0.1,
        }

    monkeypatch.setattr(init_agent, "prepare_task", lambda state: None)
    monkeypatch.setattr(init_agent, "create_workspace", lambda state, task_id: None)
    monkeypatch.setattr(init_agent, "run_agent", fake_run_agent)
    monkeypatch.setattr(debug_util, "run_agent", fake_run_agent)
    monkeypatch.setattr(code_util, "run_python_code", fake_run_python_code)

    return mock_responses


class TestInitializationPipeline:
    """Tests for run_initialization_pipeline."""

    def test_pipeline_creates_expected_state_keys(self, empty_state, setup_mocks):
        empty_state.update(
            task_name="california-housing-prices",
            data_dir="./machine_learning_engineering/tasks/",
            workspace_dir="./machine_learning_engineering/workspace/",
            lower=True,
            exec_timeout=300,
            num_model_candidates=2,
            max_retry=1,
            max_debug_round=1,
            use_data_usage_checker=False,
            use_data_leakage_checker=False,
        )

        init_agent.run_initialization_pipeline(empty_state)

        # Candidate codes and execution results were stored.
        assert "init_code_1_1" in empty_state
        assert "init_code_1_2" in empty_state
        assert "init_code_exec_result_1_1" in empty_state
        assert "init_code_exec_result_1_2" in empty_state

        # Best individual candidate was promoted.
        assert "train_code_0_1" in empty_state
        assert "train_code_exec_result_0_1" in empty_state
        assert "best_score_1" in empty_state

    def test_select_best_solution_fallback_to_individual(self, tmp_path):
        """If the selected merged solution failed, fall back to the best individual."""
        state = AgentState(
            task_name="test-task",
            workspace_dir=str(tmp_path),
            lower=True,
        )
        os.makedirs(tmp_path / "test-task" / "1", exist_ok=True)

        code_1 = "# model 1 code"
        code_2 = "# model 2 code"
        result_1 = {"returncode": 0, "score": 0.5}
        result_2 = {"returncode": 0, "score": 0.8}
        state["performance_results_1"] = [
            (0.5, code_1, result_1, 1),
            (0.8, code_2, result_2, 2),
        ]

        # Pretend the pipeline selected a merged solution that failed.
        state["best_idx_1"] = 2
        state["merger_code_1_2"] = "# failed merged code"
        state["merger_code_exec_result_1_2"] = {"returncode": 1, "stderr": "fail"}

        init_agent.select_best_solution(state, 1)

        # Fallback should pick the best individual (code_1 because lower=True).
        assert state["train_code_0_1"] == code_1
        assert state["train_code_exec_result_0_1"] == result_1

    def test_web_search_populates_state_when_enabled(
        self, empty_state, setup_mocks, monkeypatch
    ):
        """When use_web_search is enabled, web_search_results_1 is populated."""
        fake_results = [{"title": "Web Result", "body": "Web body"}]
        monkeypatch.setattr(
            web_search_util, "search_web", lambda query, num_results: fake_results
        )

        empty_state.update(
            task_name="california-housing-prices",
            data_dir="./machine_learning_engineering/tasks/",
            workspace_dir="./machine_learning_engineering/workspace/",
            lower=True,
            exec_timeout=300,
            num_model_candidates=2,
            max_retry=1,
            max_debug_round=1,
            use_data_usage_checker=False,
            use_data_leakage_checker=False,
            use_web_search=True,
            web_search_num_results=5,
        )

        init_agent.run_initialization_pipeline(empty_state)

        assert empty_state["web_search_results_1"] == fake_results
        assert "web_search_query_1" in empty_state

    def test_web_search_not_run_when_disabled(
        self, empty_state, setup_mocks, monkeypatch
    ):
        """When use_web_search is disabled, no web search state keys are created."""
        search_called = False

        def fake_search(query, num_results):
            nonlocal search_called
            search_called = True
            return []

        monkeypatch.setattr(web_search_util, "search_web", fake_search)

        empty_state.update(
            task_name="california-housing-prices",
            data_dir="./machine_learning_engineering/tasks/",
            workspace_dir="./machine_learning_engineering/workspace/",
            lower=True,
            exec_timeout=300,
            num_model_candidates=2,
            max_retry=1,
            max_debug_round=1,
            use_data_usage_checker=False,
            use_data_leakage_checker=False,
            use_web_search=False,
            web_search_num_results=5,
        )

        init_agent.run_initialization_pipeline(empty_state)

        assert not search_called
        assert "web_search_results_1" not in empty_state
