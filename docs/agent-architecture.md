# Agent architecture — two provider-native stacks, one shared logic layer

The project deliberately has **two LLM clients**, not one. Different team members
hold different API keys (a free Google AI Studio / Gemini key, or an OpenAI key),
and the two web-search capabilities are provider-native and cannot be merged into
a single client. So instead of a unifying *client*, we unify the **logic layer**
and keep the clients thin and provider-native.

```
        ┌─────────────────────────┐        ┌──────────────────────────────┐
        │  ADK / Gemini stack      │       │  OpenAI stack                 │
        │  adk_agent.py            │       │  agent.py (ManagerAgent)      │
        │  LlmAgent + google_search│       │  raw openai client +          │
        │  (native Gemini)         │       │  Responses-API web_search     │
        └────────────┬────────────┘        └──────────────┬───────────────┘
                     │      thin, provider-native clients  │
                     └──────────────────┬──────────────────┘
                                        ▼
              SHARED, CLIENT-AGNOSTIC LOGIC LAYER (imports no LLM client)
              mcts.py · skrub_ops.py · (future) data-read + skrub-spec handoff
```

## The rule

**Clients may differ per provider; the logic layer never imports a client.**

`machine_learning_engineering/__init__.py` stays side-effect-free on import: the
OpenAI `client` is built lazily via PEP 562 `__getattr__`, so
`import machine_learning_engineering.mcts` / `skrub_ops` constructs nothing and
needs no credentials. Do **not** re-add an import-time client to `__init__.py` —
that would re-couple the pure MCTS/skrub layer to an LLM key.

## Which stack do I use?

| You hold… | Use | Web search |
|---|---|---|
| Google AI Studio (Gemini) key | **ADK stack** — `adk_agent.py` (`root_agent`) | Native `google_search` (free, Gemini-only) |
| Real OpenAI key (`api.openai.com`) | **OpenAI stack** — `agent.py` (`ManagerAgent`) | Native Responses-API `web_search` |

Set the model/keys in `.env` (see `.env.example`):
- ADK / Gemini: `GOOGLE_API_KEY`, `GOOGLE_GENAI_USE_VERTEXAI=FALSE`,
  `ROOT_AGENT_MODEL=gemini-2.5-flash`.
- OpenAI: `API_KEY` + `API_BASE` (pointing at OpenAI) and `ROOT_AGENT_MODEL` set
  to a real OpenAI model (e.g. `gpt-4o`).

## Web search is provider-native (two implementations, by design)

- **Gemini stack** — `data_analyst_agent` in `adk_agent.py` carries
  `tools=[google_search]`. The built-in tool is **Gemini-only**: ADK raises
  `ValueError` if `google_search` is attached to a non-Gemini model
  (`google/adk/tools/google_search_tool.py`). `gemini-2.5-flash` is a Gemini 2.x
  model, so it may use `google_search` alongside other tools (the Gemini 1.x
  single-tool restriction does not apply — keep the model on 2.x).
- **OpenAI stack** — `SubAgent(use_web_search=True)` in `agent.py` routes through
  the OpenAI **Responses API** with `tools=[{"type": "web_search"}]`. Default is
  `use_web_search=False`, preserving the original `chat.completions` flow.

  ⚠️ **Caveat:** OpenAI's native `web_search` only works against the **real**
  OpenAI endpoint with a real key and model. It is **not** available through AI
  Studio's OpenAI-compatible endpoint. A Gemini-only contributor should use the
  ADK stack, not the OpenAI stack with search enabled.

## Why not one unifying client?

Because the web-search capabilities are bound to their providers: Google Search
grounding is a Gemini-native feature and OpenAI web search is an OpenAI-native
feature — neither is reachable from the other's client. A "merged" client would
reimplement provider routing (which ADK's model layer already does) and still
could not offer Google Search on the OpenAI path. Unifying the **logic** (the
MCTS engine and the skrub layer) is what keeps the team working from a single
source of truth, regardless of which key each member holds.

See also: [pipeline-stages.md](pipeline-stages.md) (the skrub search space the
plan author targets) and [mcts-uct.md](mcts-uct.md) (the engine the plan is
handed to).
