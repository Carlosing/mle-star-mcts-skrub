"""Common utility functions.

Deliberately stdlib-only: this module sits on the ``pipeline.py`` import chain
(via ``run_logging``), so it must not pull in torch or any other OpenMP-loading
library. See docs/BUG_LEDGER.md #11.
"""

import os
import shutil


def get_text_from_response(response) -> str:
    """Extract the text of an LLM response, whichever stack produced it.

    Two response shapes coexist in this repo: the extension's ADK stack
    (``run_logging`` → ``LlmResponse``, text split across ``content.parts``)
    and the MLE-STAR baseline's OpenAI-compatible stack (``runner.llm_call`` →
    a chat completion). Dispatch on shape rather than making each caller pick.

    Example:
        >>> get_text_from_response(openai_completion)  # .choices[0].message
        'import pandas as pd'
        >>> get_text_from_response(adk_llm_response)   # .content.parts[*].text
        'import pandas as pd'
    """
    # OpenAI-compatible chat completion (MLE-STAR baseline).
    choices = getattr(response, "choices", None)
    if choices:
        return choices[0].message.content or ""

    # ADK LlmResponse (extension agents): text can arrive in several parts.
    content = getattr(response, "content", None)
    if content is not None and getattr(content, "parts", None):
        return "".join(
            part.text
            for part in content.parts
            if getattr(part, "text", None)
        )

    return ""


def copy_file(source_file_path: str, destination_dir: str) -> None:
    """Copies a file to the specified directory."""
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source_file_path, destination_dir)
