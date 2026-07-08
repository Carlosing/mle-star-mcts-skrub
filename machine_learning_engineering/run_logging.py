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


def make_prompt_logging_callbacks(out_dir: str):
    """Return ``(before_model, after_model)`` callbacks that log to ``out_dir``.

    Records are grouped into ``<agent>_<phase>.json`` files, each a pretty JSON
    array (a model turn may fire more than once, e.g. with tool use). The same
    pair can be attached to multiple agents; both callbacks return ``None`` so
    they never alter the request or response.
    """
    os.makedirs(out_dir, exist_ok=True)
    groups: dict[str, list] = {}

    def _write(agent: str, phase: str, record: dict) -> None:
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
        _write(
            agent,
            "response",
            {
                "ts": _now(),
                "agent": agent,
                "phase": "response",
                "output": common_util.get_text_from_response(llm_response),
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
