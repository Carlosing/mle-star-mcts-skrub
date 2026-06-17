"""Test cases for the Machine Learning Engineering agent and its sub-agents."""

import os
import sys
import textwrap
import unittest

import dotenv
import pytest
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# NOTE: superseded by tests/test_agent_orchestration.py (mocked + gated live).
# Repointed from machine_learning_engineering.agent (the OpenAI ManagerAgent,
# which has no root_agent) to the ADK graph, and gated behind RUN_LIVE_TESTS so
# it no longer breaks collection or makes a live call on every run. Safe to
# delete once you're happy with test_agent_orchestration.py.
from machine_learning_engineering.adk_agent import root_agent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

live = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1" or not os.environ.get("GOOGLE_API_KEY"),
    reason="set RUN_LIVE_TESTS=1 and GOOGLE_API_KEY to run this live Gemini test",
)

session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()


@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()


@live
@pytest.mark.asyncio
async def test_happy_path():
    """Runs the agent on a simple input and expects a normal response."""
    user_input = textwrap.dedent(
        """
        Question: who are you
        Answer: Hello! I am a Machine Learning Engineering Agent.
    """
    ).strip()

    app_name = "machine-learning-engineering"

    runner = InMemoryRunner(agent=root_agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="test_user"
    )
    content = types.Content(parts=[types.Part(text=user_input)], role="user")
    response = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content,
    ):
        print(event)
        if event.content.parts and event.content.parts[0].text:
            response = event.content.parts[0].text

    # Liveness check only: the slim analyst/plan-author graph need not
    # self-describe, so just assert a non-empty response came back.
    assert response.strip()


if __name__ == "__main__":
    unittest.main()
