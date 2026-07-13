"""Tests for the ADK agent orchestration loop (data_analyst -> plan_author).

Fully mocked and offline: a fake BaseLlm returns canned responses, so the loop
runs with zero quota. These assert the orchestration itself — sequential
execution, output_key state hand-off, and the sanity-check log. Live-boundary
validation is a real pipeline smoke run (`make run-live`), not a gated test.

Run:   uv run python -m pytest tests/test_agent_orchestration.py -q
"""

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import PrivateAttr

from machine_learning_engineering.adk_agent import build_root_agent
from machine_learning_engineering.run_logging import read_records

APP_NAME = "mle-mcts-skrub-test"

# Canned model turns for the mocked runs, in call order: analyst then author.
FAKE_ANALYSIS = (
    "ANALYSIS: regression on 'target'; high-cardinality city column."
)
FAKE_SPEC_JSON = (
    '{"vectorizer": {"slots": {"high_cardinality": ["GapEncoder", "MinHashEncoder"]}}, '
    '"model": ["HistGradientBoosting", "RandomForest"]}'
)


class FakeLlm(BaseLlm):
    """A BaseLlm that yields pre-scripted responses — no network, no quota.

    Each call pops the next canned string (clamping on the last one). Named as a
    Gemini model so any model-id checks pass even if a tool is attached.
    """

    model: str = "gemini-2.5-flash"
    _responses: list = PrivateAttr(default_factory=list)
    _idx: int = PrivateAttr(default=0)

    def set_responses(self, responses):
        self._responses = list(responses)
        self._idx = 0
        return self

    async def generate_content_async(self, llm_request, stream: bool = False):
        if self._responses:
            text = self._responses[min(self._idx, len(self._responses) - 1)]
        else:
            text = ""
        self._idx += 1
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)])
        )


async def _run(root_agent, user_text):
    """Drive a graph to completion and return (events, final_session)."""
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="u"
    )
    content = types.Content(parts=[types.Part(text=user_text)], role="user")
    events = []
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
        events.append(event)
    final = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=session.user_id, session_id=session.id
    )
    return events, final


# ---------------------------------------------------------------------------
# Mocked orchestration (offline, no API quota)
# ---------------------------------------------------------------------------


async def test_loop_runs_both_agents_in_order(tmp_path):
    """Both sub-agents run, and each writes its output_key into session state."""
    model = FakeLlm().set_responses([FAKE_ANALYSIS, FAKE_SPEC_JSON])
    root = build_root_agent(
        model=model, log_dir=str(tmp_path), with_search=False
    )

    _, session = await _run(root, "California housing; predict target.")

    assert session.state.get("dataset_analysis") == FAKE_ANALYSIS
    assert session.state.get("skrub_spec_raw") == FAKE_SPEC_JSON


async def test_state_handoff_analysis_injected_into_author_prompt(tmp_path):
    """plan_author's resolved prompt must contain the analyst's output.

    The instruction template references {dataset_analysis}; ADK fills it from
    state before the call. We read it back from the sanity log's request record.
    """
    model = FakeLlm().set_responses([FAKE_ANALYSIS, FAKE_SPEC_JSON])
    root = build_root_agent(
        model=model, log_dir=str(tmp_path), with_search=False
    )

    await _run(root, "California housing; predict target.")

    records = _read_log(tmp_path)
    author_request = next(
        r
        for r in records
        if r["agent"] == "plan_author" and r["phase"] == "request"
    )
    assert FAKE_ANALYSIS in author_request["system_instruction"]


async def test_sanity_log_is_legible(tmp_path):
    """The log captures a request+response per agent, parseable line by line."""
    model = FakeLlm().set_responses([FAKE_ANALYSIS, FAKE_SPEC_JSON])
    root = build_root_agent(
        model=model, log_dir=str(tmp_path), with_search=False
    )

    await _run(root, "California housing; predict target.")

    records = _read_log(tmp_path)
    agents_phases = {(r["agent"], r["phase"]) for r in records}
    assert ("data_analyst", "request") in agents_phases
    assert ("data_analyst", "response") in agents_phases
    assert ("plan_author", "request") in agents_phases
    assert ("plan_author", "response") in agents_phases

    # Responses carry the model output text; requests carry the instruction.
    analyst_resp = next(
        r
        for r in records
        if r["agent"] == "data_analyst" and r["phase"] == "response"
    )
    assert analyst_resp["output"] == FAKE_ANALYSIS


def _read_log(tmp_path) -> list[dict]:
    records = read_records(str(tmp_path))
    assert records, "sanity log files were not written"
    return records
