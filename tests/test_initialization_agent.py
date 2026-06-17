"""Unit tests for sub_agents.initialization.agent helpers."""

import os

import pytest

from machine_learning_engineering.runner import AgentState
from machine_learning_engineering.sub_agents.initialization.agent import (
    _DEFAULT_CANDIDATES,
    _parse_model_candidates,
    get_model_candidates,
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
        raw = (
            "```python\nimport numpy as np\nmodel = SomeModel()\n```"
        )
        models = _parse_model_candidates(raw, 1)
        assert len(models) == 1
        assert models[0]["model_name"] == "Model"

    def test_returns_empty_list_when_unparseable(self):
        raw = "This is not a valid response at all."
        models = _parse_model_candidates(raw, 2)
        assert models == []

    def test_get_model_candidates_fallback_to_defaults(self, tmp_path):
        raw = "This is not a valid response at all."
        response = type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": raw})})]})()
        state = AgentState(
            task_name="test-task",
            workspace_dir=str(tmp_path),
            num_model_candidates=2,
        )
        os.makedirs(tmp_path / "test-task" / "1" / "model_candidates", exist_ok=True)

        get_model_candidates(state, "model_retriever_agent_1", response, 1)

        assert state["init_1_model_finish"] is True
        assert state["init_1_model_1"]["model_name"] == _DEFAULT_CANDIDATES[0]["model_name"]
        assert state["init_1_model_2"]["model_name"] == _DEFAULT_CANDIDATES[1]["model_name"]

    def test_partial_json_with_extra_text(self):
        raw = (
            "Some explanation before\n"
            "[\n"
            "  {\"model_name\": \"RandomForestRegressor\", \"example_code\": \"code\"}\n"
            "]\n"
            "Some explanation after"
        )
        models = _parse_model_candidates(raw, 1)
        assert len(models) == 1
        assert models[0]["model_name"] == "RandomForestRegressor"
