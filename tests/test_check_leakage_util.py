"""Unit tests for shared_libraries.check_leakage_util."""

from types import SimpleNamespace

import pytest

from machine_learning_engineering.runner import AgentState
from machine_learning_engineering.shared_libraries import check_leakage_util, code_util


def _make_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=0),
    )


class TestParseLeakageResponse:
    """Tests for the internal leakage response parser."""

    def test_no_leakage(self):
        text = '[{"leakage_status": "No Data Leakage", "code_block": ""}]'
        status, block = check_leakage_util._parse_leakage_response(text)
        assert status == "No Data Leakage"
        assert block == ""

    def test_yes_leakage(self):
        text = '[{"leakage_status": "Yes Data Leakage", "code_block": "leaky_block"}]'
        status, block = check_leakage_util._parse_leakage_response(text)
        assert status == "Yes Data Leakage"
        assert block == "leaky_block"

    def test_json_object_instead_of_list(self):
        text = '{"leakage_status": "Yes Data Leakage", "code_block": "block"}'
        status, block = check_leakage_util._parse_leakage_response(text)
        assert status == "Yes Data Leakage"
        assert block == "block"

    def test_strips_markdown_fences(self):
        text = '[{"leakage_status": "Yes Data Leakage", "code_block": "```python\\nblock\\n```"}]'
        status, block = check_leakage_util._parse_leakage_response(text)
        assert block == "\nblock\n"

    def test_malformed_json_defaults_to_no_leakage(self):
        text = "This is not valid JSON at all."
        status, block = check_leakage_util._parse_leakage_response(text)
        assert status == "No Data Leakage"
        assert block == ""


class TestCheckAndFixLeakage:
    """Tests for check_and_fix_leakage with mocked LLM responses."""

    @pytest.fixture
    def state(self):
        return AgentState(
            use_data_leakage_checker=True,
            lower=True,
        )

    def test_no_leakage_leaves_code_unchanged(self, state, monkeypatch):
        agent_name = "model_eval_1_1"
        suffix = code_util.get_updated_suffix(state, agent_name)
        code_key = code_util.get_code_state_key(agent_name, suffix)
        original_code = "print('hello')"
        state[code_key] = original_code

        def fake_run_agent(s, name, instruction, **kwargs):
            return _make_response(
                "[{'leakage_status': 'No Data Leakage', 'code_block': ''}]"
            )

        monkeypatch.setattr(check_leakage_util, "run_agent", fake_run_agent)

        assert check_leakage_util.check_and_fix_leakage(state, agent_name) is True
        assert state[code_key] == original_code

    def test_yes_leakage_replaces_block(self, state, monkeypatch):
        agent_name = "model_eval_1_1"
        suffix = code_util.get_updated_suffix(state, agent_name)
        code_key = code_util.get_code_state_key(agent_name, suffix)
        leaky_block = "X_train = scaler.fit_transform(X_full)"
        original_code = f"some_code\n{leaky_block}\nmore_code"
        refined_block = "X_train = scaler.fit_transform(X_train)"
        state[code_key] = original_code

        call_count = 0

        def fake_run_agent(s, name, instruction, **kwargs):
            nonlocal call_count
            call_count += 1
            if "check_leakage" in name:
                # First check says leakage; final verification says no leakage.
                if call_count == 1:
                    return _make_response(
                        f'[{{"leakage_status": "Yes Data Leakage", "code_block": "{leaky_block}"}}]'
                    )
                return _make_response(
                    '[{"leakage_status": "No Data Leakage", "code_block": ""}]'
                )
            # refine_leakage agent
            return _make_response(f"```python\n{refined_block}\n```")

        monkeypatch.setattr(check_leakage_util, "run_agent", fake_run_agent)
        monkeypatch.setattr(code_util, "evaluate_code", lambda s, n: None)

        result_key = code_util.get_code_execution_result_state_key(agent_name, suffix)
        state[result_key] = {"returncode": 0, "score": 1.0}

        assert check_leakage_util.check_and_fix_leakage(state, agent_name) is True
        assert refined_block in state[code_key]
        assert leaky_block not in state[code_key]

    def test_mismatched_block_returns_false(self, state, monkeypatch):
        agent_name = "model_eval_1_1"
        suffix = code_util.get_updated_suffix(state, agent_name)
        code_key = code_util.get_code_state_key(agent_name, suffix)
        state[code_key] = "print('hello')"

        def fake_run_agent(s, name, instruction, **kwargs):
            return _make_response(
                '[{"leakage_status": "Yes Data Leakage", "code_block": '
                '"this_block_is_not_in_the_code"}]'
            )

        monkeypatch.setattr(check_leakage_util, "run_agent", fake_run_agent)

        assert check_leakage_util.check_and_fix_leakage(state, agent_name) is False

    def test_malformed_response_treated_as_no_leakage(self, state, monkeypatch):
        agent_name = "model_eval_1_1"
        suffix = code_util.get_updated_suffix(state, agent_name)
        code_key = code_util.get_code_state_key(agent_name, suffix)
        original_code = "print('safe')"
        state[code_key] = original_code

        def fake_run_agent(s, name, instruction, **kwargs):
            return _make_response("Malformed response without JSON")

        monkeypatch.setattr(check_leakage_util, "run_agent", fake_run_agent)

        assert check_leakage_util.check_and_fix_leakage(state, agent_name) is True
        assert state[code_key] == original_code
