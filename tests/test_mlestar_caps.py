"""Offline checks for the MLE-STAR baseline's hard-cap machinery.

No network: the caps wrap ``runner.llm_call`` — the single chokepoint every
sub-agent funnels through — so we can stub that one function and prove the caps
abort cleanly, the output bound is applied, and tokens are attributed per agent.
Full end-to-end MLE-STAR needs a live provider + budget (it writes and executes
generated Python), which is out of scope for the offline suite.
"""

import dataclasses
import time
import types

import pytest

from machine_learning_engineering import runner
import scripts.run_mlestar as R


def _fake_response(total=30):
    """A minimal OpenAI-compatible completion with usage."""
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content="ok")
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=10, completion_tokens=20, total_tokens=total
        ),
    )


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace runner.llm_call with an offline stub; record the kwargs it saw."""
    seen = []

    def fake(messages, temperature=0.0, model=None, max_tokens=None):
        seen.append({"model": model, "max_tokens": max_tokens})
        return _fake_response()

    monkeypatch.setattr(runner, "llm_call", fake)
    return seen


def test_clamp_config_sets_conservative_knobs():
    cfg = R._clamp_config(
        "california-housing-prices", "./machine_learning_engineering/tasks/"
    )
    assert cfg.num_solutions == 1
    assert cfg.max_debug_round == 1
    assert cfg.outer_loop_round == 1
    assert cfg.task_name == "california-housing-prices"


def test_effective_model_strips_litellm_prefix():
    # The raw OpenAI-compatible client rejects the LiteLlm `openai/` prefix that
    # the extension's ADK path requires.
    assert R._effective_model("openai/qwen3.5-397b-a17b") == "qwen3.5-397b-a17b"
    assert R._effective_model("gemini-2.5-flash") == "gemini-2.5-flash"


def test_caps_abort_on_max_calls_and_apply_output_bound(stub_llm):
    counter = {
        "calls": 0,
        "max_calls": 2,
        "deadline": time.perf_counter() + 9999,
        "model": "openai/qwen3.5-397b-a17b",
    }
    sink = {}
    original = R.install_caps(counter, sink, max_output_tokens=4096)
    try:
        runner.llm_call([{"role": "user", "content": "hi"}])
        runner.llm_call([{"role": "user", "content": "hi"}])
        assert counter["calls"] == 2

        with pytest.raises(R._CallBudgetExceeded):
            runner.llm_call([{"role": "user", "content": "hi"}])
    finally:
        R.restore_caps(original)

    # per-call output bound applied, and the litellm prefix stripped on the way
    assert stub_llm[0]["max_tokens"] == 4096
    assert stub_llm[0]["model"] == "qwen3.5-397b-a17b"


def test_caps_abort_on_deadline_before_counting(stub_llm):
    counter = {
        "calls": 0,
        "max_calls": 99,
        "deadline": time.perf_counter() - 1.0,  # already past
        "model": "gemini-2.5-flash",
    }
    original = R.install_caps(counter, {}, max_output_tokens=4096)
    try:
        with pytest.raises(R._CallBudgetExceeded):
            runner.llm_call([{"role": "user", "content": "hi"}])
    finally:
        R.restore_caps(original)
    assert counter["calls"] == 0  # deadline check precedes the counter


def test_tokens_captured_and_attributed_per_agent(stub_llm):
    counter = {
        "calls": 0,
        "max_calls": 9,
        "deadline": time.perf_counter() + 9999,
        "model": "gemini-2.5-flash",
    }
    sink = {}
    original = R.install_caps(counter, sink, max_output_tokens=4096)
    try:
        # run_agent sets runner.CURRENT_AGENT, which the cap wrapper reads.
        runner.run_agent(runner.AgentState(), "init_agent", "do the thing")
        runner.run_agent(runner.AgentState(), "refine_agent", "do it again")
    finally:
        R.restore_caps(original)

    assert set(sink) == {"init_agent", "refine_agent"}
    assert sink["init_agent"]["total"] == 30
    assert sink["init_agent"]["prompt"] == 10
    assert sink["init_agent"]["completion"] == 20
    assert sink["init_agent"]["calls"] == 1


def test_restore_caps_unwraps():
    counter = {
        "calls": 0,
        "max_calls": 1,
        "deadline": time.perf_counter() + 9999,
        "model": "gemini-2.5-flash",
    }
    before = runner.llm_call
    original = R.install_caps(counter, {}, max_output_tokens=1)
    assert runner.llm_call is not before
    R.restore_caps(original)
    assert runner.llm_call is before


def test_run_pipeline_seeds_every_config_field_onto_state(monkeypatch):
    """The sub-agents read their knobs from `state`, not from config.

    Regression: run_pipeline seeded only 4 keys, so the harness's clamps and
    `use_web_search` silently never reached the sub-agents.
    """
    from machine_learning_engineering import agent
    from machine_learning_engineering.shared_libraries import config

    captured = {}

    def capture(state):
        captured.update(state)

    for mod in ("initialization", "refinement", "ensemble", "submission"):
        monkeypatch.setattr(
            getattr(agent, f"{mod}_agent_module"),
            f"run_{mod}_pipeline",
            capture,
        )
    monkeypatch.setattr(agent, "save_state", lambda state: None)

    config.CONFIG.use_web_search = True
    config.CONFIG.max_debug_round = 1
    agent.run_pipeline(task_name="california-housing-prices")

    for field in dataclasses.fields(config.CONFIG):
        assert field.name in captured, f"{field.name} never reached the state"
    assert captured["use_web_search"] is True
    assert captured["max_debug_round"] == 1
