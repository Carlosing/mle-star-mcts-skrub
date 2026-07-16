"""Shared test fixtures and utilities."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Ensure the package can be imported without a real .env file.
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-tests")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost/dummy")
os.environ.setdefault("ROOT_AGENT_MODEL", "dummy-model")

from machine_learning_engineering.runner import AgentState


@pytest.fixture
def empty_state() -> AgentState:
    """Return a fresh AgentState with minimal defaults."""
    return AgentState()


@pytest.fixture
def base_state() -> AgentState:
    """Return an AgentState populated with common config-like keys."""
    return AgentState(
        task_name="california-housing-prices",
        data_dir="./machine_learning_engineering/tasks/",
        workspace_dir="./machine_learning_engineering/workspace/",
        lower=True,
        exec_timeout=300,
        num_model_candidates=2,
        max_retry=1,
        max_debug_round=1,
        use_data_leakage_checker=False,
        use_data_usage_checker=False,
    )


def make_response(content: str) -> SimpleNamespace:
    """Build a minimal OpenAI-compatible response object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=MagicMock(total_tokens=0),
    )
