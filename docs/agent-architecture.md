# Agent architecture — one ADK stack, provider switched by env

The repository serves a **three-way benchmark** (extension vs AutoGluon vs
MLE-STAR), and two of the three arms are agentic. This doc covers both:

| Arm | Agent stack | LLM cost model | Web search |
|---|---|---|---|
| **Extension** (the project) | ADK graph `data_analyst → plan_author` ([`adk_agent.py`](../machine_learning_engineering/adk_agent.py)) | **O(1)** calls/task (2, +1 per Optional Feature 3 proposal) | `google_search` (Gemini path only) |
| **MLE-STAR** (revived baseline) | ported `ManagerAgent` + `sub_agents/` over [`runner.py`](../machine_learning_engineering/runner.py) | **unbounded** (code-and-debug cascade), hard-capped by the harness | DuckDuckGo (`shared_libraries/web_search_util.py`) |
| AutoGluon | none (pure AutoML) | 0 | — |

The extension has **one** agent stack: the Google ADK graph
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
(`google.genai` on a `gemini-*` model, `openai` otherwise — the Optional Feature 3
proposer follows the same provider switch as the agents) and
`adk_agent._resolve_model` (`LiteLlm`, only on the non-Gemini path).

## Choosing a provider (env only)

`.env` holds both credential sets, prefixed `GOOGLE_*` and `SCHOOL_*`
(`_ROOT_AGENT_MODEL` / `_API_KEY` / `_API_BASE`); `PROVIDER=google|school`
(default `google`) picks which set is copied onto the canonical
`ROOT_AGENT_MODEL` / `API_KEY` / `API_BASE` names before config loads
(`machine_learning_engineering.__init__._select_provider`). A bare
`ROOT_AGENT_MODEL` in an old `.env` still works. Note: the school model id
MUST carry the `openai/` prefix (e.g. `openai/qwen3.5-397b-a17b`) or
`_resolve_model` misroutes it. See [`.env.example`](../.env.example) and
[USAGE.md](USAGE.md).

The model id is the switch (`adk_agent._resolve_model`):

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
without the SOTA-lookup step. When search is off, the search-instruction
fragments are also stripped from both agents' prompts, so the model is never
told to use a tool it doesn't have.

## Reasoning models (school endpoint)

The capable school (GWDG) models are *reasoning* models: they burn output
tokens thinking before the answer, and a low `max_tokens` ends them mid-thought
with empty content → fallback spec. The LiteLlm path therefore sets
`max_tokens=16384` by default (env override `LITELLM_MAX_TOKENS`). Probe what
the endpoint currently serves with `make probe-school` (`SMOKE=1` health-checks
each model).

## The second stack: the MLE-STAR port, revived as the benchmark baseline

The repo carries a **second** agent stack — the hand-rolled OpenAI `ManagerAgent`
in `agent.py` plus `sub_agents/` and `shared_libraries/`. This is the *upstream
MLE-STAR agent*, **ported off the ADK runtime onto plain OpenAI-compatible
calls** early in the project (commits `fac8795`/`241beeb`: a `ManagerAgent` with
a token budget and kill switch, a sub-agent execution protocol, the full
initialization → refinement → ensemble → submission pipeline). It writes and
debugs Python, rather than authoring a search space. It was deprecated when the
extension moved onto ADK (LiteLlm) — and then **revived as the thing we
benchmark against** (`make bench-mlestar`, `scripts/run_mlestar.py`; adopted
into the benchmark branch in merge `7867f06`).

It no longer runs on ADK. `runner.py` is a minimal OpenAI-compatible runner that
replaces the ADK runtime, and its **`llm_call` is the single chokepoint** every
sub-agent reaches the provider through — which is what lets the harness wrap one
function to impose the caps MLE-STAR needs (max LLM calls, per-call output-token
bound, wall-clock deadline). Its token cost is otherwise unbounded: it is a
code-and-debug loop, so a hard task can cascade indefinitely. That unboundedness
*is* the comparison — see [PROJECT_STATE.md](PROJECT_STATE.md). Under the cap it
can also fail to converge on a runnable script at all (videogame-sales did) —
a failure mode the extension's validated-plan design cannot produce.

### Web-search integration (the port's retrieval step)

Upstream MLE-STAR's first phase retrieves candidate models from the web. The
port restores this with **DuckDuckGo** search
(`shared_libraries/web_search_util.py`, commit `e7c80a8`) wired into the
initialization agent's model-retrieval step — the MLE-STAR analogue of the
extension analyst's `google_search`, and the reason `ddgs` is a core
dependency. So *both* agentic arms have a web-grounding step: Gemini-native
`google_search` on the extension's analyst, DuckDuckGo scraping on the
baseline's retriever.

Two rules still hold:

- **It is off the MCTS path.** No extension feature belongs in `agent.py` /
  `sub_agents/`. Touch it only to keep the baseline running.
- **Do not re-add an import-time client to `__init__.py`.** The package builds
  the OpenAI client lazily (PEP 562 `__getattr__`), so `import
  machine_learning_engineering.mcts` still constructs nothing and needs no
  `.env`. An eager client would re-couple the pure MCTS/skrub layer to an LLM key
  and break the offline suite.

See also: [pipeline-stages.md](pipeline-stages.md) (the skrub search space the
plan author targets) and [mcts-uct.md](mcts-uct.md) (the engine the plan is
handed to).
