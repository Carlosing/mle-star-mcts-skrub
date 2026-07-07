# Agent architecture — one ADK stack, provider switched by env

The project has **one** agent stack: the Google ADK graph in
[`adk_agent.py`](../machine_learning_engineering/adk_agent.py)
(`data_analyst → plan_author`). It drives **native Gemini** *or* an **OpenAI /
OpenAI-compatible** endpoint, chosen purely by environment variables — no code
change to switch providers. Underneath, the whole logic layer (MCTS + skrub)
imports no LLM client at all.

```
                    ┌──────────────────────────────────────────────┐
                    │  ADK agent graph — adk_agent.py               │
                    │  data_analyst → plan_author                   │
                    │  model resolved by ROOT_AGENT_MODEL:          │
                    │   • gemini-*      → native Gemini (+ search)  │
                    │   • anything else → LiteLlm(OpenAI/compat)    │
                    └───────────────────────┬──────────────────────┘
                                            │  rich JSON plan
                                            ▼
              SHARED, CLIENT-AGNOSTIC LOGIC LAYER (imports no LLM client)
              spec_resolver.py · skrub_ops.py · search_loop.py · mcts.py
```

## The rule

**The logic layer never imports an LLM client.** `mcts.py` / `skrub_ops.py` /
`spec_resolver.py` / `search_loop.py` stay import-clean and offline-testable;
the only lazy provider imports are inside `search_loop.make_llm_proposer`
(`google.genai`) and `adk_agent._resolve_model` (`LiteLlm`, only on the
non-Gemini path).

## Choosing a provider (env only)

Set the model + key in `.env` (see [`.env.example`](../.env.example)); the model
id is the switch (`adk_agent._resolve_model`):

| `ROOT_AGENT_MODEL` | Routes to | Key(s) | Web search |
|---|---|---|---|
| `gemini-2.5-flash` (any `gemini-*`) | **native Gemini** | `GOOGLE_API_KEY`, `GOOGLE_GENAI_USE_VERTEXAI=FALSE` | ✅ `google_search` (Gemini-only) |
| `openai/gpt-4o` (any non-`gemini`) | **LiteLlm** (real OpenAI, a university proxy, or an OpenAI-compat base) | `API_KEY` (+ `API_BASE`) | ❌ (see below) |

`_resolve_model` returns `(model, is_gemini)`; `build_root_agent` attaches
`google_search` only when `is_gemini` is true, so requesting `with_search=True`
on an OpenAI model silently drops the tool instead of raising.

## Web search is Gemini-native

`google_search` is a built-in Gemini tool — ADK raises `ValueError` if it is
attached to a non-Gemini model (`google/adk/tools/google_search_tool.py`).
`gemini-2.5-flash` is a Gemini 2.x model, so it may use `google_search`
alongside other tools (the Gemini 1.x single-tool restriction does not apply —
keep the model on 2.x). On the OpenAI/LiteLlm path there is no equivalent
built-in tool, so the analyst simply runs without web search: an OpenAI-only
contributor still gets the full analyst → plan_author → MCTS pipeline, just
without the SOTA-lookup step.

## Deprecated: the standalone OpenAI `ManagerAgent`

The earlier design carried a **second** stack — a hand-rolled OpenAI
`ManagerAgent` in `agent.py` with its own client built in
`machine_learning_engineering/__init__.py` (plus `sub_agents/`, `eval/`). Now
that OpenAI runs *through ADK* (LiteLlm), that stack is **deprecated and off the
MCTS path**, retained only for merge/reference history; new work targets the ADK
graph. Do **not** re-add an import-time client to `__init__.py` — that would
re-couple the pure MCTS/skrub layer to an LLM key.

See also: [pipeline-stages.md](pipeline-stages.md) (the skrub search space the
plan author targets) and [mcts-uct.md](mcts-uct.md) (the engine the plan is
handed to).
