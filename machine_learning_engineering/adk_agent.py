"""Slim ADK agent graph for the MCTS+skrub direction (native Gemini).

This is intentionally a *thin slice* of MLE-STAR: it keeps only the front half
— read/analyze the data, then author a **rich skrub plan** — and stops before
any Python code generation. The plan is handed to the MCTS+skrub layer
(``skrub_ops.build_staged_plan`` → ``mcts.mcts_search``), which does the search.

This ADK graph is now the **single** agent stack: it drives native Gemini *or*
an OpenAI / OpenAI-compatible endpoint, chosen purely by env vars — no code
change to switch providers. (The old hand-rolled OpenAI ``ManagerAgent`` in
``agent.py`` is deprecated and off the MCTS path.)

Model wiring (see ``_resolve_model``): ``config.CONFIG.agent_model`` comes from
``ROOT_AGENT_MODEL``. A bare Gemini id (e.g. ``gemini-2.5-flash``) routes to
**native Gemini** via ADK's LLMRegistry — with ``GOOGLE_API_KEY`` set and
``GOOGLE_GENAI_USE_VERTEXAI=FALSE`` that uses the free Google AI Studio key, no
Vertex/Cloud. Any other model id (e.g. ``openai/gpt-4o``) is wrapped in ADK's
``LiteLlm`` and reads ``API_KEY``/``API_BASE`` from the env, so the same graph
runs against real OpenAI or a proxy. ``google_search`` is a Gemini-native tool,
so web search is available **only** on the Gemini path (auto-disabled otherwise).

Construction goes through ``build_root_agent`` so tests can inject a fake model
and a prompt/output log directory without hitting the live API. The module-level
``root_agent`` is the default (real Gemini, web search on) for ``adk run``.

Hand-off (now implemented): each agent writes its text output to shared state
via ``output_key`` (``dataset_analysis``, ``skrub_spec_raw``). The driver in
``pipeline.py`` reads ``skrub_spec_raw``, resolves it to seeded estimator
instances (with hyperparameter choices) via ``spec_resolver.resolve_spec``
— allowed-list only, no ``eval`` — then runs ``build_staged_plan`` ->
``make_rollout_fn`` -> ``mcts_search``. The allowed operator/HP vocabulary is
injected into the plan_author instruction from the registry, so the model stays
in-bounds.
"""

import os

from google.adk import agents
from google.adk.tools.google_search_tool import google_search

from machine_learning_engineering.run_logging import (
    make_prompt_logging_callbacks,
)
from machine_learning_engineering.shared_libraries import config
from machine_learning_engineering.spec_resolver import format_allowed_for_prompt

# The model string (ROOT_AGENT_MODEL) -> native Gemini or a LiteLlm-wrapped
# OpenAI/compatible endpoint; see `_resolve_model`.
_MODEL = config.CONFIG.agent_model


def _is_gemini_model(model) -> bool:
    """True when `model` is a bare Gemini model id (the native-Gemini path)."""
    return isinstance(model, str) and "gemini" in model.lower()


def _resolve_model(model):
    """Resolve a model spec into an (adk_model, is_gemini) pair.

    A bare Gemini id is returned unchanged for ADK's native-Gemini routing. Any
    other string is wrapped in ``LiteLlm`` so the SAME agent graph can drive an
    OpenAI / OpenAI-compatible endpoint (real OpenAI, a university proxy, AI
    Studio's OpenAI-compat base, ...), selected purely by ``ROOT_AGENT_MODEL`` +
    ``API_KEY``/``API_BASE`` in the env — switching providers needs no code
    change. A non-string is an already-built model (tests' ``FakeLlm``) and is
    passed through. ``google_search`` is Gemini-native, so ``is_gemini`` tells
    the caller whether web search may be attached.

    Example:
        _resolve_model("gemini-2.5-flash")  # -> ("gemini-2.5-flash", True)
        _resolve_model("openai/gpt-4o")     # -> (LiteLlm(model="openai/gpt-4o"), False)
    """
    if not isinstance(model, str):
        return model, True  # pre-built BaseLlm; caller controls with_search
    if _is_gemini_model(model):
        return model, True
    from google.adk.models.lite_llm import LiteLlm

    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("API_BASE") or os.environ.get("OPENAI_API_BASE")
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    # Reasoning models (qwen3.*, gpt-oss, deepseek) burn output tokens THINKING
    # before the answer; a low cap ends the response mid-thought (content=None,
    # finish_reason=length) and the plan silently falls back. Give the whole
    # plan + its reasoning room. Tunable via LITELLM_MAX_TOKENS.
    kwargs["max_tokens"] = int(os.environ.get("LITELLM_MAX_TOKENS", "16384"))
    return LiteLlm(model=model, **kwargs), False


# --- Instructions (module constants so the factory stays readable) ----------

# The search hints are their own fragments so build_root_agent can strip them
# when search is off (non-Gemini providers): telling a reasoning model to use a
# tool it doesn't have wastes thinking tokens and invites hallucinated results.
_ANALYST_SEARCH_FRAGMENT = (
    "Use the google_search tool to retrieve current state-of-the-art encoders "
    "and model families for this kind of tabular task before you commit to "
    "recommendations.\n\n"
)

_ANALYST_INSTRUCTION = (
    "You are a tabular-data analyst. You are given a task description and a "
    "compact summary of the dataset (column names, dtypes, cardinality, "
    "missing-value rates, target column). Do NOT write Python code.\n\n"
    + _ANALYST_SEARCH_FRAGMENT
    + "Produce a short structured analysis covering:\n"
    "  - the task type (regression vs classification) and target column;\n"
    "  - which columns are high-cardinality / dirty categoricals, datetime, "
    "or numeric;\n"
    "  - candidate encoders (e.g. GapEncoder vs MinHashEncoder), cleaning "
    "steps, scaling/feature-engineering, and 2-3 candidate model families;\n"
    "  - any SPECIFIC column that deserves its own operator: a date column to "
    "expand into parts (keeping the raw date), a free-text/dirty column needing "
    "a dedicated encoder, or a numeric column to rescale after encoding;\n"
    "  - any relational/auxiliary-table opportunity (aggregate joins). If the "
    "summary lists auxiliary tables, name the join keys (the summary lists "
    "candidates) and which aux columns/aggregations look predictive.\n"
    "This analysis is the input to a planner that will turn it into a "
    "searchable skrub pipeline; be concrete about the *options* worth searching "
    "at each stage, not a single fixed choice."
)

_PLANNER_SEARCH_FRAGMENT = (
    "When the google_search tool is available, use it to check for current "
    "state-of-the-art operators and sensible hyperparameter ranges for this kind "
    "of tabular task before you finalize the menu (e.g. gradient-boosting "
    "settings, encoders) — but only propose operators on the allowed list "
    "below.\n\n"
)

_PLAN_AUTHOR_INSTRUCTION = (
    "Your entire reply must be a single JSON object and nothing else. Do not "
    "explain, justify or comment on your choices — not before the JSON, not "
    "after it. Your reply is parsed by a machine, never read by a human.\n\n"
    "You are the skrub plan author. Using {dataset_analysis}, output a JSON "
    "object describing, FOR EACH PIPELINE STAGE, the list of candidate "
    "operators to search over (not a single choice). The downstream engine "
    "searches this menu with MCTS, so offer 2-3 real options per stage wherever "
    "it is reasonable.\n\n"
    + _PLANNER_SEARCH_FRAGMENT
    + "Pipeline stages, in order: assemble (relational, optional) -> clean "
    "(optional) -> scoped operators pre-encode (optional) -> encode (required) "
    "-> scoped operators post-encode (optional) -> scale/feature-eng (optional) "
    "-> model (required). For optional stages, include a 'skip' option.\n\n"
    "An operator is a FULL DOTTED IMPORT PATH (bare, for defaults) OR an object "
    '{"name": <path>, "params": {...}} to ALSO search its hyperparameters — '
    "prefer tuning the model's key hyperparameters, and give GENEROUS, "
    "domain-informed ranges (wide enough to contain the optimum for THIS kind "
    "of data). You may tune ANY hyperparameter the operator's constructor "
    "accepts (not only a fixed list), and your ranges are used AS GIVEN (not "
    "clipped). Because the search cross-validates many configs under a "
    "wall-clock budget, AVOID ranges whose upper bound makes a single fit "
    "extremely slow (e.g. n_estimators or max_iter in the tens of thousands, "
    "very large n_components, a high polynomial degree) — keep upper bounds to "
    "what trains in a few seconds on a subsample. Any operator "
    'object may also carry "prior": 0.0-1.0 — your confidence that this option '
    "wins on THIS dataset; the search visits high-prior options first. Emit "
    "ONLY JSON of this shape (dotted paths + HP ranges; lazy import/resolution "
    "is downstream):\n"
    "{\n"
    '  "clean_options": ["skip", "skrub.Cleaner"],\n'
    '  "encoder_options": ["skrub.GapEncoder", "skrub.MinHashEncoder"],\n'
    '  "stages": [\n'
    '    {"name": "scale", "options": ["skip", "sklearn.preprocessing.StandardScaler"]},\n'
    '    {"name": "feature_eng", "options": ["skip",\n'
    '      {"name": "sklearn.preprocessing.PolynomialFeatures", "params": {"degree": {"int": [2, 3]}}}]}\n'
    "  ],\n"
    '  "model": [\n'
    '    {"name": "sklearn.ensemble.HistGradientBoostingRegressor", "prior": 0.8, "params": {\n'
    '       "learning_rate": {"float": [0.01, 0.3], "log": true},\n'
    '       "max_iter": {"int": [100, 600]}, "max_depth": {"int": [2, 16]}}},\n'
    '    {"name": "sklearn.ensemble.RandomForestRegressor", "params": {"n_estimators": {"int": [100, 500]}}}\n'
    "  ]\n"
    "}\n\n"
    "RELATIONAL (only when the data summary lists auxiliary tables — omit "
    '"assemble" otherwise): add an "assemble" key with 1-3 aggregate-join '
    "candidates over the auxiliary tables. Each entry is\n"
    '  {"name": <short label>, "table": <aux table name>, '
    '"key": <shared column> OR "main_key": <main col> + "aux_key": <aux col>, '
    '"operations": [<subset of mean,min,max,sum,median,std,count,mode>], '
    '"cols": [<aux value columns to aggregate>]}\n'
    "Use EXACT table and column names from the data summary (join-key "
    "candidates are listed there); sum/median/mean/std only on numeric "
    'columns. A "skip" option is added automatically.\n\n'
    "SCOPED OPERATORS (optional, at most 3 groups): when SPECIFIC columns "
    "deserve their own operator (a dirty high-cardinality categorical, free "
    "text, a date to expand, a numeric column to rescale, ...), add\n"
    '  "scoped_encodings": [{"name": <short label>, '
    '"cols": [<exact column names from the data summary>], '
    '"options": [<operator paths to search, e.g. "skrub.GapEncoder">], '
    '"position": "pre_encode"|"post_encode", "additive": true|false}]\n'
    "The engine searches each group independently (a 'skip' option is added "
    "automatically) and the default vectorizer still handles all unscoped "
    "columns. Use exact column names; invented names are dropped.\n"
    '"position" (default "pre_encode") says WHERE the operator runs: '
    '"pre_encode" = on the raw columns before vectorization (encoders, date '
    'expansion); "post_encode" = on the vectorized table (e.g. scaling — only '
    "NUMERIC columns keep their names through the vectorizer, so post_encode "
    'only works on numeric columns). Set "additive": true when the operation '
    "is NOT in-place — i.e. the original columns must be KEPT alongside the "
    "derived output (e.g. extracting month from a date while keeping the raw "
    "date). The engine concatenates the derived columns by row index "
    "automatically; leave false (default) to replace the columns.\n\n"
    + format_allowed_for_prompt()
    + "\n\nOUTPUT CONTRACT: reply with the JSON object alone. The first "
    "character is '{' and the last is '}'. No preamble, no markdown fence, no "
    "trailing remarks about the ranges you picked. Any prose is a failure."
)


def build_root_agent(
    model=None,
    log_dir: str | None = None,
    with_search: bool = True,
    usage_sink: dict | None = None,
):
    """Build the analyze -> author plan graph.

    Args:
      model: model string or ``BaseLlm`` instance. Defaults to native Gemini
        (``config.CONFIG.agent_model``). Tests pass a fake model here.
      log_dir: if given, prompts + outputs of every LLM turn are appended to
        ``<log_dir>/<agent>_<phase>.json`` for sanity inspection (see run_logging).
      with_search: *request* the Gemini-only ``google_search`` tool on both the
        analyst and the plan_author. It is attached only when the resolved model
        is native Gemini (`_resolve_model`); on an OpenAI/compatible model it is
        silently dropped, since ``google_search`` is Gemini-native. Set False for
        offline/mocked runs.
      usage_sink: if given, a dict the after-model callback accumulates per-agent
        token usage into (see ``run_logging.make_prompt_logging_callbacks``). The
        callbacks are attached when either ``log_dir`` or ``usage_sink`` is set.

    Returns:
      The root ``SequentialAgent``; its ``sub_agents`` are ``[analyst, author]``.
    """
    model = model if model is not None else _MODEL
    resolved, is_gemini = _resolve_model(model)
    use_search = with_search and is_gemini

    # Strip the search hints when the tool isn't attached, so the prompt never
    # instructs the model to use a tool it doesn't have.
    analyst_instruction = _ANALYST_INSTRUCTION
    planner_instruction = _PLAN_AUTHOR_INSTRUCTION
    if not use_search:
        analyst_instruction = analyst_instruction.replace(
            _ANALYST_SEARCH_FRAGMENT, ""
        )
        planner_instruction = planner_instruction.replace(
            _PLANNER_SEARCH_FRAGMENT, ""
        )

    before_cb, after_cb = (None, None)
    if log_dir is not None or usage_sink is not None:
        # log_dir=None still gets callbacks (for usage capture) but writes no
        # files — see make_prompt_logging_callbacks.
        before_cb, after_cb = make_prompt_logging_callbacks(
            log_dir, usage_sink=usage_sink
        )

    data_analyst = agents.LlmAgent(
        name="data_analyst",
        model=resolved,
        tools=[google_search] if use_search else [],
        before_model_callback=before_cb,
        after_model_callback=after_cb,
        description="Reads the task description and a dataset summary, retrieves "
        "current SOTA tabular approaches via web search, and reasons about "
        "column types, cardinality, missingness, and useful preprocessing/model "
        "families.",
        instruction=analyst_instruction,
        output_key="dataset_analysis",
    )

    plan_author = agents.LlmAgent(
        name="plan_author",
        model=resolved,
        tools=[google_search] if use_search else [],
        before_model_callback=before_cb,
        after_model_callback=after_cb,
        description="Turns the analyst's findings into a rich, per-stage menu of "
        "operator options — the skrub 'spec' that MCTS searches over.",
        instruction=planner_instruction,
        output_key="skrub_spec_raw",
    )

    return agents.SequentialAgent(
        name="mle_mcts_skrub_root",
        description="MCTS+skrub front-end: analyze the data, then author a rich "
        "skrub plan for the MCTS search layer.",
        sub_agents=[data_analyst, plan_author],
    )


# Default graph for `adk run` / `adk web`: real Gemini, web search on, no logging.
root_agent = build_root_agent()
data_analyst_agent, plan_author_agent = root_agent.sub_agents
