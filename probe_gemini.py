"""Standalone Gemini API probe — independent of the rest of the project.

Two things the API can actually tell you:
  1. models.list()      -> each model's input/output TOKEN limits (context size).
  2. a tiny generate    -> whether this key has LIVE QUOTA right now (200 vs 429).

Note: the API does NOT expose your rate limits (RPM/TPM/RPD) — those live in the
AI Studio / Cloud console. A 429 with "limit: 0" means the free tier grants this
model/project zero quota (not "you used it up").

Run:
    uv run python probe_gemini.py

Reads GOOGLE_API_KEY (and GOOGLE_GENAI_USE_VERTEXAI=FALSE) from .env.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

# Cheap 1-token pings; edit freely to probe whichever models you care about.
CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
]

# Seconds to wait between pings so the per-minute rate limit (RPM) doesn't make
# a model with real quota look rate-limited. Free tier is ~5-15 RPM.
PING_DELAY_S = 3.0


def main() -> None:
    load_dotenv()
    if not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        raise SystemExit("Set GOOGLE_API_KEY in .env first.")

    client = genai.Client()  # reads GOOGLE_API_KEY / GOOGLE_GENAI_USE_VERTEXAI

    print("== Available models (token limits from models.list) ==")
    for m in client.models.list():
        actions = (
            getattr(m, "supported_actions", None)
            or getattr(m, "supported_generation_methods", None)
            or []
        )
        if "generateContent" not in actions:
            continue
        name = m.name.removeprefix("models/")
        in_lim = getattr(m, "input_token_limit", "?")
        out_lim = getattr(m, "output_token_limit", "?")
        print(f"  {name:36} in={in_lim!s:>9}  out={out_lim!s:>7}")

    print("\n== Quota ping (does this key have live quota right now?) ==")
    for i, model in enumerate(CANDIDATES):
        if i:
            time.sleep(PING_DELAY_S)  # space out pings to respect RPM limits
        print(f"  {model:36} {ping(client, model)}")


def ping(client: "genai.Client", model: str) -> str:
    """Send a minimal request and classify the outcome."""
    try:
        client.models.generate_content(
            model=model,
            contents="ping",
            config={"max_output_tokens": 1},
        )
        return "OK — quota available"
    except errors.APIError as e:
        code = getattr(e, "code", "?")
        msg = getattr(e, "message", "") or str(e)
        if code == 429:
            if "limit: 0" in msg:
                return "429 — free-tier limit is 0 for this model/project"
            return "429 — rate-limited (quota exists; retry later)"
        return f"{code} — {msg[:90]}"
    except Exception as e:  # network, auth, unknown model, etc.
        return f"ERROR {type(e).__name__}: {str(e)[:90]}"


if __name__ == "__main__":
    main()
