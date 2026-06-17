"""Unit tests for shared_libraries.code_util."""

import pytest

from machine_learning_engineering.shared_libraries import code_util


class TestGetUpdatedSuffix:
    """Tests for get_updated_suffix."""

    def test_model_eval(self, empty_state):
        suffix = code_util.get_updated_suffix(empty_state, "model_eval_1_2")
        assert suffix == "1_2"

    def test_merger(self, empty_state):
        suffix = code_util.get_updated_suffix(empty_state, "merger_1_3")
        assert suffix == "1_3"

    def test_check_data_use(self, empty_state):
        suffix = code_util.get_updated_suffix(empty_state, "check_data_use_1")
        assert suffix == "1"

    def test_ablation(self, empty_state):
        empty_state["refine_step_1"] = 4
        suffix = code_util.get_updated_suffix(empty_state, "ablation_1")
        assert suffix == "4_1"

    def test_plan_implement(self, empty_state):
        empty_state["refine_step_1"] = 2
        empty_state["inner_iter_1"] = 3
        suffix = code_util.get_updated_suffix(empty_state, "plan_implement_1")
        assert suffix == "3_2_1"

    def test_ensemble_plan_implement(self, empty_state):
        empty_state["ensemble_iter"] = 7
        suffix = code_util.get_updated_suffix(empty_state, "ensemble_plan_implement_1")
        assert suffix == "7"

    def test_submission(self, empty_state):
        suffix = code_util.get_updated_suffix(empty_state, "submission")
        assert suffix == ""

    def test_unknown_agent_raises(self, empty_state):
        with pytest.raises(ValueError, match="Unexpected agent name"):
            code_util.get_updated_suffix(empty_state, "unknown_agent_1")


class TestGetCodeStateKey:
    """Tests for get_code_state_key."""

    @pytest.mark.parametrize(
        ("agent_name", "suffix", "expected"),
        [
            ("model_eval_1_2", "1_2", "init_code_1_2"),
            ("merger_1_3", "1_3", "merger_code_1_3"),
            ("check_data_use_1", "1", "train_code_0_1"),
            ("ablation_1", "4_1", "ablation_code_4_1"),
            ("plan_implement_1", "3_2_1", "train_code_improve_3_2_1"),
            ("ensemble_plan_implement", "7", "ensemble_code_7"),
            ("submission", "", "submission_code"),
        ],
    )
    def test_known_agents(self, agent_name, suffix, expected):
        assert code_util.get_code_state_key(agent_name, suffix) == expected

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unexpected agent name"):
            code_util.get_code_state_key("unknown_agent_1", "x")


class TestGetCodeExecutionResultStateKey:
    """Tests for get_code_execution_result_state_key."""

    @pytest.mark.parametrize(
        ("agent_name", "suffix", "expected"),
        [
            ("model_eval_1_2", "1_2", "init_code_exec_result_1_2"),
            ("merger_1_3", "1_3", "merger_code_exec_result_1_3"),
            ("check_data_use_1", "1", "train_code_exec_result_0_1"),
            ("ablation_1", "4_1", "ablation_code_exec_result_4_1"),
            ("plan_implement_1", "3_2_1", "train_code_improve_exec_result_3_2_1"),
            ("ensemble_plan_implement", "7", "ensemble_code_exec_result_7"),
            ("submission", "", "submission_code_exec_result"),
        ],
    )
    def test_known_agents(self, agent_name, suffix, expected):
        assert (
            code_util.get_code_execution_result_state_key(agent_name, suffix) == expected
        )

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unexpected agent name"):
            code_util.get_code_execution_result_state_key("unknown_agent_1", "x")


class TestExtractPerformanceFromText:
    """Tests for extract_performance_from_text."""

    def test_exact_format(self):
        text = "Final Validation Performance: 123.45"
        assert code_util.extract_performance_from_text(text) == 123.45

    def test_extra_text_and_units(self):
        text = "Some header\nFinal Validation Performance: 0.8765 (RMSE)\nFooter"
        assert code_util.extract_performance_from_text(text) == 0.8765

    def test_scientific_notation(self):
        text = "Final Validation Performance: 1.23e-4"
        assert code_util.extract_performance_from_text(text) == 1.23e-4

    def test_missing_performance_line_returns_none(self):
        text = "Training complete. Best epoch: 42"
        assert code_util.extract_performance_from_text(text) is None
