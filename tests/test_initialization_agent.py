"""Unit tests for sub_agents.initialization.agent helpers."""

import os
from unittest import mock

import pytest  # noqa: F401

from machine_learning_engineering.runner import AgentState
from machine_learning_engineering.sub_agents.initialization.agent import (
    _DEFAULT_CANDIDATES,
    _parse_model_candidates,
    get_model_candidates,
    get_model_retriever_agent_instruction,
    run_web_search_for_task,
)


class TestParseModelCandidates:
    """Tests for _parse_model_candidates."""

    def test_json_list_parsing(self):
        raw = """[
          {"model_name": "RandomForestRegressor", "example_code": "from sklearn.ensemble import RandomForestRegressor\\nmodel = RandomForestRegressor()"},
          {"model_name": "GradientBoostingRegressor", "example_code": "from sklearn.ensemble import GradientBoostingRegressor\\nmodel = GradientBoostingRegressor()"}
        ]"""
        models = _parse_model_candidates(raw, 2)
        assert len(models) == 2
        assert models[0]["model_name"] == "RandomForestRegressor"
        assert "RandomForestRegressor" in models[0]["example_code"]

    def test_respects_num_candidates(self):
        raw = """[
          {"model_name": "A", "example_code": "code_a"},
          {"model_name": "B", "example_code": "code_b"},
          {"model_name": "C", "example_code": "code_c"}
        ]"""
        models = _parse_model_candidates(raw, 2)
        assert len(models) == 2
        assert models[0]["model_name"] == "A"
        assert models[1]["model_name"] == "B"

    def test_markdown_code_block_fallback(self):
        raw = (
            "Here are two candidates:\n\n"
            "```python\nfrom sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()\n```\n\n"
            "```python\nfrom sklearn.ensemble import GradientBoostingRegressor\nmodel = GradientBoostingRegressor()\n```"
        )
        models = _parse_model_candidates(raw, 2)
        assert len(models) == 2
        assert models[0]["model_name"] == "RandomForestRegressor"
        assert models[1]["model_name"] == "GradientBoostingRegressor"

    def test_code_block_without_sklearn_import(self):
        raw = "```python\nimport numpy as np\nmodel = SomeModel()\n```"
        models = _parse_model_candidates(raw, 1)
        assert len(models) == 1
        assert models[0]["model_name"] == "Model"

    def test_returns_empty_list_when_unparseable(self):
        raw = "This is not a valid response at all."
        models = _parse_model_candidates(raw, 2)
        assert models == []

    def test_get_model_candidates_fallback_to_defaults(self, tmp_path):
        raw = "This is not a valid response at all."
        response = type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": raw})},
                    )
                ]
            },
        )()
        state = AgentState(
            task_name="test-task",
            workspace_dir=str(tmp_path),
            num_model_candidates=2,
        )
        os.makedirs(
            tmp_path / "test-task" / "1" / "model_candidates", exist_ok=True
        )

        get_model_candidates(state, "model_retriever_agent_1", response, 1)

        assert state["init_1_model_finish"] is True
        assert (
            state["init_1_model_1"]["model_name"]
            == _DEFAULT_CANDIDATES[0]["model_name"]
        )
        assert (
            state["init_1_model_2"]["model_name"]
            == _DEFAULT_CANDIDATES[1]["model_name"]
        )

    def test_partial_json_with_extra_text(self):
        raw = (
            "Some explanation before\n"
            "[\n"
            '  {"model_name": "RandomForestRegressor", "example_code": "code"}\n'
            "]\n"
            "Some explanation after"
        )
        models = _parse_model_candidates(raw, 1)
        assert len(models) == 1
        assert models[0]["model_name"] == "RandomForestRegressor"


class TestRunWebSearchForTask:
    """Tests for run_web_search_for_task."""

    def test_skips_search_when_disabled(self, empty_state):
        empty_state["use_web_search"] = False
        empty_state["task_summary"] = "summary"

        with mock.patch(
            "machine_learning_engineering.sub_agents.initialization.agent.web_search_util.search_web"
        ) as fake_search:
            run_web_search_for_task(empty_state, 1)

        fake_search.assert_not_called()
        assert "web_search_results_1" not in empty_state

    def test_runs_search_when_enabled(self, empty_state):
        empty_state["use_web_search"] = True
        empty_state["task_summary"] = "California housing regression"
        empty_state["task_type"] = "Tabular Regression"
        empty_state["web_search_num_results"] = 3
        fake_results = [{"title": "Result A", "body": "Body A"}]

        with mock.patch(
            "machine_learning_engineering.sub_agents.initialization.agent.web_search_util.search_web",
            return_value=fake_results,
        ) as fake_search:
            run_web_search_for_task(empty_state, 1)

        fake_search.assert_called_once()
        call_args = fake_search.call_args
        assert "California housing regression" in call_args.args[0]
        assert call_args.kwargs.get("num_results") == 3
        assert empty_state["web_search_results_1"] == fake_results
        assert "web_search_query_1" in empty_state

    def test_swallows_search_failure(self, empty_state):
        empty_state["use_web_search"] = True
        empty_state["task_summary"] = "summary"

        with mock.patch(
            "machine_learning_engineering.sub_agents.initialization.agent.web_search_util.search_web",
            return_value=[],
        ) as fake_search:
            run_web_search_for_task(empty_state, 1)

        fake_search.assert_called_once()
        assert empty_state["web_search_results_1"] == []


class TestGetModelRetrieverAgentInstruction:
    """Tests for get_model_retriever_agent_instruction."""

    def test_includes_search_results(self, empty_state):
        empty_state["task_summary"] = "summary"
        empty_state["num_model_candidates"] = 2
        empty_state["web_search_results_1"] = [
            {"title": "Result A", "body": "Body A"},
        ]

        instruction = get_model_retriever_agent_instruction(
            empty_state, "model_retriever_agent_1"
        )

        assert "summary" in instruction
        assert "Result A" in instruction
        assert "Body A" in instruction

    def test_handles_missing_search_results(self, empty_state):
        empty_state["task_summary"] = "summary"
        empty_state["num_model_candidates"] = 1

        instruction = get_model_retriever_agent_instruction(
            empty_state, "model_retriever_agent_1"
        )

        assert "No web search results available." in instruction
        assert "summary" in instruction
