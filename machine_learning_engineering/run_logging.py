"""Sanity-check logging of ADK agent prompts + LLM outputs.

A lightweight inspection aid — NOT the MCTS hand-off. For each LLM turn it
records *what each agent was asked* (resolved system instruction + contents) and
*what it produced* (output text), grouped into one **pretty JSON array per
agent + phase**: ``<out_dir>/<agent>_request.json`` and
``<out_dir>/<agent>_response.json`` (e.g. ``plan_author_request.json``). Pretty
arrays so they open cleanly in an editor's built-in JSON formatter. Attach the
returned pair to an ``LlmAgent``'s ``before_model_callback`` /
``after_model_callback``.

Separate from any richer persistence so it can be swapped or removed without
touching the agent graph.
"""

import glob
import json
import os
from datetime import datetime, timezone

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from machine_learning_engineering.shared_libraries import common_util


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _system_instruction_text(llm_request: LlmRequest) -> str:
    """Best-effort string view of the resolved system instruction."""
    config = getattr(llm_request, "config", None)
    si = getattr(config, "system_instruction", None) if config else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    if isinstance(si, list):
        return "\n\n".join(str(part) for part in si)
    return str(si)


def _contents_text(llm_request: LlmRequest) -> list[str]:
    """Flatten each Content's text parts into a list of strings."""
    out: list[str] = []
    for content in getattr(llm_request, "contents", []) or []:
        parts = getattr(content, "parts", None) or []
        text = "".join(getattr(p, "text", "") or "" for p in parts)
        role = getattr(content, "role", "")
        out.append(f"[{role}] {text}" if role else text)
    return out


def extract_usage(response) -> dict:
    """Best-effort ``{prompt, completion, total}`` token counts from a response.

    Handles the two shapes this project sees: ADK/native-genai
    (``response.usage_metadata`` with ``prompt_token_count`` /
    ``candidates_token_count`` / ``total_token_count``) and the OpenAI-compatible
    path (``response.usage`` with ``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens``). Returns all-zero when no usage is present (e.g. a mocked
    ``FakeLlm`` response), never raising.

    Example:
        extract_usage(gemini_resp)   # -> {"prompt": 812, "completion": 1500, "total": 2312}
        extract_usage(fake_resp)     # -> {"prompt": 0, "completion": 0, "total": 0}
    """
    zero = {"prompt": 0, "completion": 0, "total": 0}
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        prompt = getattr(meta, "prompt_token_count", None)
        completion = getattr(meta, "candidates_token_count", None)
        total = getattr(meta, "total_token_count", None)
        if any(v is not None for v in (prompt, completion, total)):
            prompt = prompt or 0
            completion = completion or 0
            return {
                "prompt": prompt,
                "completion": completion,
                "total": total if total is not None else prompt + completion,
            }
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None) or 0
        completion = getattr(usage, "completion_tokens", None) or 0
        total = getattr(usage, "total_tokens", None)
        if prompt or completion or total:
            return {
                "prompt": prompt,
                "completion": completion,
                "total": total if total is not None else prompt + completion,
            }
    return dict(zero)


def add_usage(sink: dict, agent: str, usage: dict) -> None:
    """Accumulate a per-call ``usage`` into ``sink[agent]`` (creating the slot).

    ``sink`` maps agent name -> running ``{prompt, completion, total, calls}``.
    A call with all-zero tokens still increments ``calls`` (it happened), so the
    count tracks real turns even when the provider omits usage.
    """
    slot = sink.setdefault(
        agent, {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
    )
    slot["prompt"] += usage.get("prompt", 0)
    slot["completion"] += usage.get("completion", 0)
    slot["total"] += usage.get("total", 0)
    slot["calls"] += 1


def make_prompt_logging_callbacks(
    out_dir: str | None, usage_sink: dict | None = None
):
    """Return ``(before_model, after_model)`` callbacks that log to ``out_dir``.

    Records are grouped into ``<agent>_<phase>.json`` files, each a pretty JSON
    array (a model turn may fire more than once, e.g. with tool use). The same
    pair can be attached to multiple agents; both callbacks return ``None`` so
    they never alter the request or response.

    ``out_dir=None`` disables the file writes (no directory is created) — use
    this when only ``usage_sink`` token capture is wanted, so an offline/mocked
    run leaves no stray log dir behind.

    If ``usage_sink`` (a dict) is given, ``after_model`` also accumulates each
    turn's token usage into it, keyed by agent name (see ``add_usage``) — this
    is how the pipeline reports a per-agent token cost without changing the agent
    graph. The same sink can be shared across a retry build so both attempts'
    tokens count.
    """
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    groups: dict[str, list] = {}

    def _write(agent: str, phase: str, record: dict) -> None:
        if out_dir is None:
            return
        key = f"{agent}_{phase}"
        groups.setdefault(key, []).append(record)
        with open(
            os.path.join(out_dir, f"{key}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(groups[key], f, ensure_ascii=False, indent=2)

    def before_model(
        callback_context: CallbackContext, llm_request: LlmRequest
    ):
        agent = callback_context.agent_name
        _write(
            agent,
            "request",
            {
                "ts": _now(),
                "agent": agent,
                "phase": "request",
                "system_instruction": _system_instruction_text(llm_request),
                "contents": _contents_text(llm_request),
            },
        )
        return None  # do not override the outgoing request

    def after_model(
        callback_context: CallbackContext, llm_response: LlmResponse
    ):
        agent = callback_context.agent_name
        usage = extract_usage(llm_response)
        if usage_sink is not None:
            add_usage(usage_sink, agent, usage)
        _write(
            agent,
            "response",
            {
                "ts": _now(),
                "agent": agent,
                "phase": "response",
                "output": common_util.get_text_from_response(llm_response),
                "tokens": usage,
            },
        )
        return None  # do not override the response

    return before_model, after_model


def read_records(out_dir: str) -> list[dict]:
    """Read back all logged records across every ``<agent>_<phase>.json``.

    Only the agent log files are read (``*_request.json`` / ``*_response.json``),
    so a sibling ``result.json`` in the same dir is ignored.
    """
    records: list[dict] = []
    for pattern in ("*_request.json", "*_response.json"):
        for path in sorted(glob.glob(os.path.join(out_dir, pattern))):
            with open(path, encoding="utf-8") as f:
                records.extend(json.load(f))
    return records
