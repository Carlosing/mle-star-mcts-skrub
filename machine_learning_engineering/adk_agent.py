"""Slim ADK agent graph for the MCTS+skrub direction (native Gemini).

This is intentionally a *thin slice* of MLE-STAR: it keeps only the front half
— read/analyze the data, then author a **rich skrub plan** — and stops before
any Python code generation. The plan is handed to the MCTS+skrub layer
(``skrub_ops.build_staged_plan`` → ``mcts.mcts_search``), which does the search.

Why this lives in its own module (not ``agent.py``): the hand-rolled
OpenAI ``ManagerAgent`` in ``agent.py`` is the team's code and is kept intact.
This file is the parallel ADK path and is the one to point ``adk run`` /
``adk web`` at.

Model wiring: ``config.CONFIG.agent_model`` is a plain Gemini model string
(e.g. ``gemini-2.0-flash``), which ADK's LLMRegistry routes to **native
Gemini**. With ``GOOGLE_API_KEY`` set and ``GOOGLE_GENAI_USE_VERTEXAI=FALSE``
(see ``.env``), that uses the free Google AI Studio key — no Vertex/Cloud.

DEFERRED (step #4, not in this skeleton):
- an after-model callback on ``plan_author_agent`` that parses its JSON output
  into a validated skrub *spec dict* and resolves operator names to real
  estimator instances (no ``eval`` of model output);
- the driver that calls ``build_staged_plan`` / ``make_rollout_fn`` /
  ``mcts_search`` on that spec.
For now each agent just writes its text output to shared state via ``output_key``.
"""

from google.adk import agents

from machine_learning_engineering.shared_libraries import config

# The model string -> native Gemini (AI Studio key) via ADK's registry.
_MODEL = config.CONFIG.agent_model


# ---------------------------------------------------------------------------
# 1. Data analyst — read + analyze the dataset, no code generation.
# ---------------------------------------------------------------------------
data_analyst_agent = agents.LlmAgent(
    name="data_analyst",
    model=_MODEL,
    description="Reads the task description and a dataset summary, and reasons "
    "about column types, cardinality, missingness, and likely useful "
    "preprocessing/model families.",
    instruction=(
        "You are a tabular-data analyst. You are given a task description and a "
        "compact summary of the dataset (column names, dtypes, cardinality, "
        "missing-value rates, target column). Do NOT write Python code.\n\n"
        "Produce a short structured analysis covering:\n"
        "  - the task type (regression vs classification) and target column;\n"
        "  - which columns are high-cardinality / dirty categoricals, datetime, "
        "or numeric;\n"
        "  - candidate encoders (e.g. GapEncoder vs MinHashEncoder), cleaning "
        "steps, scaling/feature-engineering, and 2-3 candidate model families;\n"
        "  - any relational/auxiliary-table opportunity (aggregate joins).\n"
        "This analysis is the input to a planner that will turn it into a "
        "searchable skrub pipeline; be concrete about the *options* worth "
        "searching at each stage, not a single fixed choice."
    ),
    output_key="dataset_analysis",
)


# ---------------------------------------------------------------------------
# 2. Plan author — emit a rich skrub plan as a menu of options per stage.
# ---------------------------------------------------------------------------
plan_author_agent = agents.LlmAgent(
    name="plan_author",
    model=_MODEL,
    description="Turns the analyst's findings into a rich, per-stage menu of "
    "operator options — the skrub 'spec' that MCTS searches over.",
    instruction=(
        "You are the skrub plan author. Using {dataset_analysis}, output a JSON "
        "object describing, FOR EACH PIPELINE STAGE, the list of candidate "
        "operators to search over (not a single choice). The downstream engine "
        "searches this menu with MCTS, so offer 2-3 real options per stage "
        "wherever it is reasonable.\n\n"
        "Pipeline stages, in order: assemble (relational, optional) -> clean "
        "(optional) -> encode (required) -> scale/feature-eng (optional) -> "
        "model (required). For optional stages, include a 'skip' option.\n\n"
        "Emit ONLY JSON of this shape (use operator *names*, not Python code — "
        "name resolution to estimator instances happens downstream):\n"
        "{\n"
        '  "clean_options": ["skip", "Cleaner"],\n'
        '  "encoder_options": ["GapEncoder", "MinHashEncoder"],\n'
        '  "stages": [\n'
        '    {"name": "scale", "options": ["skip", "StandardScaler"]},\n'
        '    {"name": "feature_eng", "options": ["skip", "PolynomialFeatures"]}\n'
        "  ],\n"
        '  "model": ["HistGradientBoosting", "RandomForest"]\n'
        "}\n"
    ),
    output_key="skrub_spec_raw",
)


# ---------------------------------------------------------------------------
# Root: analyze -> author plan. (MCTS hand-off is the deferred step #4.)
# ---------------------------------------------------------------------------
root_agent = agents.SequentialAgent(
    name="mle_mcts_skrub_root",
    description="MCTS+skrub front-end: analyze the data, then author a rich "
    "skrub plan for the MCTS search layer.",
    sub_agents=[data_analyst_agent, plan_author_agent],
)
