"""Probe the school (GWDG chat-ai / OpenAI-compatible) endpoint.

Lists the models the key can actually see (the endpoint is authoritative — it
drifts from the docs), and with ``--smoke`` completes a one-word prompt on each
to classify it: does it return usable ``content``, is it a *reasoning* model
(emits a ``reasoning`` field and burns tokens thinking before the answer), and
does it follow instructions. Reasoning models need a generous ``max_tokens`` or
the answer never arrives (``content=None``, ``finish_reason=length``).

    uv run python probe_school.py            # list models
    uv run python probe_school.py --smoke     # + health/behaviour of each
    uv run python probe_school.py --smoke --model qwen3.5-397b-a17b
"""

import argparse
import os

import machine_learning_engineering  # loads .env (SCHOOL_API_KEY/_API_BASE)
from openai import OpenAI


def _client() -> OpenAI:
    key = os.environ.get("SCHOOL_API_KEY")
    base = os.environ.get("SCHOOL_API_BASE")
    if not key or not base:
        raise SystemExit("SCHOOL_API_KEY / SCHOOL_API_BASE missing from .env")
    return OpenAI(api_key=key, base_url=base)


def _smoke(client: OpenAI, model: str, max_tokens: int) -> str:
    try:
        r = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": "Reply with exactly one word: OK"}
            ],
        )
        ch = r.choices[0]
        d = ch.message.model_dump()
        reasoning = bool(d.get("reasoning") or d.get("reasoning_content"))
        content = (ch.message.content or "").strip()
        kind = "reasoning" if reasoning else "instruct"
        got = repr(content[:16]) if content else "None(!)"
        follows = "ok" if content.upper().startswith("OK") else "loose"
        return (
            f"{kind:9s} finish={ch.finish_reason:7s} content={got:12s} "
            f"follows={follows:5s} tok={r.usage.completion_tokens}"
        )
    except Exception as exc:  # noqa: BLE001 — report, don't crash the sweep
        return f"UNAVAILABLE: {type(exc).__name__} {str(exc)[:70]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--smoke", action="store_true", help="complete on each model"
    )
    ap.add_argument("--model", default=None, help="smoke only this model")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    client = _client()
    models = sorted(m.id for m in client.models.list().data)
    print(f"{len(models)} models at {os.environ['SCHOOL_API_BASE']}:")
    targets = [args.model] if args.model else models
    for mid in models:
        if not args.smoke or (args.model and mid != args.model):
            print(f"  {mid}")
            continue
        print(f"  {mid:34s} {_smoke(client, mid, args.max_tokens)}")


if __name__ == "__main__":
    main()
