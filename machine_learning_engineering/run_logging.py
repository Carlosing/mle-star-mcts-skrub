"""Sanity-check logging of ADK agent prompts + LLM outputs.

A lightweight inspection aid — NOT the MCTS hand-off. For each LLM turn it
appends one JSON record per line to ``<out_dir>/agent_io.jsonl`` so you can
eyeball *what each agent was actually asked* (resolved system instruction +
contents) and *what it produced* (output text). Attach the returned pair to an
``LlmAgent``'s ``before_model_callback`` / ``after_model_callback``.

This is deliberately separate from any richer persistence layer so it can be
swapped or removed without touching the agent graph.
"""

import json
import os
from datetime import datetime, timezone

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from machine_learning_engineering.shared_libraries import common_util

LOG_FILENAME = "agent_io.jsonl"


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


def _append(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_prompt_logging_callbacks(out_dir: str):
    """Return ``(before_model, after_model)`` callbacks that log to ``out_dir``.

    The same pair can be attached to multiple agents; records are distinguished
    by ``callback_context.agent_name``. Both callbacks return ``None`` so they
    never alter the request or response.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, LOG_FILENAME)

    def before_model(callback_context: CallbackContext, llm_request: LlmRequest):
        _append(
            path,
            {
                "ts": _now(),
                "agent": callback_context.agent_name,
                "phase": "request",
                "system_instruction": _system_instruction_text(llm_request),
                "contents": _contents_text(llm_request),
            },
        )
        return None  # do not override the outgoing request

    def after_model(callback_context: CallbackContext, llm_response: LlmResponse):
        _append(
            path,
            {
                "ts": _now(),
                "agent": callback_context.agent_name,
                "phase": "response",
                "output": common_util.get_text_from_response(llm_response),
            },
        )
        return None  # do not override the response

    return before_model, after_model
