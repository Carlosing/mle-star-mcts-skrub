"""Package init — deliberately free of import-time LLM side effects.

The pure-logic modules (``mcts.py``, ``skrub_ops.py``) must be importable
without any LLM credentials. The OpenAI client used by the hand-rolled agents
(``agent.py``, ``prueba_api.py``) is therefore built **lazily** on first access
via PEP 562 module ``__getattr__`` — so ``from machine_learning_engineering
import client, MODEL_NAME`` keeps working unchanged, while
``import machine_learning_engineering.mcts`` constructs nothing and requires no
``.env``.
"""

import os

from dotenv import load_dotenv

# Loading the .env is harmless (no network, no raise) and keeps config.py and
# the agents seeing the same environment as before.
load_dotenv()


def _select_provider() -> None:
    """Map ``{PROVIDER}_*`` env vars onto the canonical names the code reads.

    ``PROVIDER=google`` (default) or ``PROVIDER=school`` picks which credential
    set in ``.env`` is active: ``{PREFIX}_ROOT_AGENT_MODEL``/``_API_KEY``/
    ``_API_BASE`` are copied to the plain ``ROOT_AGENT_MODEL``/``API_KEY``/
    ``API_BASE`` that ``config.py`` and ``adk_agent._resolve_model`` consume.
    Runs before any submodule (this is the package ``__init__``), so config
    sees the resolved values. Backward-compatible: a missing prefixed var
    leaves the canonical one untouched, so an old ``.env`` with a bare
    ``ROOT_AGENT_MODEL`` still works. The native-Gemini path reads
    ``GOOGLE_API_KEY`` directly, which is already the ``google`` prefix.
    """
    prefix = {"school": "SCHOOL"}.get(
        os.environ.get("PROVIDER", "google").strip().lower(), "GOOGLE"
    )
    for canonical in ("ROOT_AGENT_MODEL", "API_KEY", "API_BASE"):
        value = os.environ.get(f"{prefix}_{canonical}")
        if value:
            os.environ[canonical] = value


_select_provider()

# Plain env read, no side effects — safe as an eager module attribute.
MODEL_NAME = os.environ.get("ROOT_AGENT_MODEL", "meta-llama-3.1-8b-instruct")

# Cached singleton; only built when `client` is actually accessed.
_client = None


def _build_client():
    """Construct the OpenAI client for the hand-rolled (non-ADK) agents.

    Same semantics as before — reads OPENAI_API_KEY / OPENAI_API_BASE and
    raises if they are missing — only now this runs on first use of ``client``
    rather than at package import time.
    """
    from openai import OpenAI

    # Generic API_KEY/API_BASE are the current convention; fall back to the
    # older OPENAI_* names so existing setups keep working.
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("API_BASE") or os.environ.get("OPENAI_API_BASE")
    if not api_key or not api_base:
        raise ValueError(
            "CRITICAL: API_KEY or API_BASE not found in the environment. "
            "Make sure your .env file is configured."
        )

    client = OpenAI(api_key=api_key, base_url=api_base)
    print("\n🚀 [SYSTEM] Engine initialized successfully.")
    print(f"🧠 [SYSTEM] Model loaded: {MODEL_NAME}\n")
    return client


def __getattr__(name):
    """PEP 562: build ``client`` on first access, keep import side-effect-free."""
    if name == "client":
        global _client
        if _client is None:
            _client = _build_client()
        return _client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
