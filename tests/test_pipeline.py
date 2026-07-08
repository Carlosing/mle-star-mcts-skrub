"""End-to-end pipeline tests — agents mocked, skrub+MCTS real, fully offline."""

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

from machine_learning_engineering import pipeline
from fixtures.california_agent_io import DATASET_ANALYSIS, SKRUB_SPEC_RAW

ANALYSIS = DATASET_ANALYSIS
SPEC_JSON = SKRUB_SPEC_RAW


class FakeLlm(BaseLlm):
    """Yields scripted responses — no network, no quota."""

    model: str = "gemini-2.5-flash"
    _responses: list = PrivateAttr(default_factory=list)
    _idx: int = PrivateAttr(default=0)

    def set_responses(self, responses):
        self._responses = list(responses)
        self._idx = 0
        return self

    async def generate_content_async(self, llm_request, stream: bool = False):
        text = (
            self._responses[min(self._idx, len(self._responses) - 1)]
            if self._responses
            else ""
        )
        self._idx += 1
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)])
        )


def test_load_task_california():
    df, target, task_type, metric, desc, aux_tables = pipeline.load_task(
        "california-housing-prices"
    )
    assert target == "median_house_value"
    assert task_type == "regression"
    assert metric == "root_mean_squared_error"
    assert target in df.columns
    assert "median_house_value" in desc  # full task description is returned
    assert aux_tables == {}  # single-table task -> no auxiliary tables


def test_run_pipeline_end_to_end_offline():
    model = FakeLlm().set_responses([ANALYSIS, SPEC_JSON])
    result = pipeline.run_pipeline(
        task_name="california-housing-prices",
        budget=15,
        model=model,
        with_search=False,  # FakeLlm -> skip the Gemini-only google_search tool
    )

    assert result["target"] == "median_house_value"
    assert result["task_type"] == "regression"
    assert result["search_scorer"] == "r2"
    assert not result["used_fallback_spec"]

    # MCTS searched a real action space and returned a config + score.
    assert "model" in result["action_space"]
    assert set(result["action_space"]["model"]) == {
        "HistGradientBoostingRegressor",
        "RandomForestRegressor",
        "Ridge",
    }
    # the richer spec also searches hyperparameters (nested HP dimensions)
    assert any("__" in k for k in result["action_space"])
    print(result["best_state"])
    print(result["best_search_score"])
    assert isinstance(result["best_state"], dict)
    assert isinstance(result["best_search_score"], float)

    # The report metric (RMSE) is computed on the incumbent for reporting only.
    assert result["report"] is not None
    assert result["report"]["scorer"] == "neg_root_mean_squared_error"


def test_run_pipeline_falls_back_on_bad_spec():
    # plan_author returns unparseable JSON -> resolver falls back, run still completes.
    model = FakeLlm().set_responses(
        [ANALYSIS, "sorry, I could not produce JSON"]
    )
    result = pipeline.run_pipeline(
        task_name="california-housing-prices",
        budget=4,
        model=model,
        with_search=False,
    )
    assert result["used_fallback_spec"] is True
    assert isinstance(result["best_search_score"], float)


def test_run_pipeline_classification_offline():
    # The classification path: string target -> accuracy scorer, Classifier models.
    from fixtures.open_payments_agent_io import RESPONSES as CLF_RESPONSES

    model = FakeLlm().set_responses(CLF_RESPONSES)
    result = pipeline.run_pipeline(
        task_name="open-payments", budget=4, model=model, with_search=False
    )

    assert result["task_type"] == "classification"
    assert result["search_scorer"] == "accuracy"
    assert not result["used_fallback_spec"]
    assert set(result["action_space"]["model"]) == {
        "HistGradientBoostingClassifier",
        "RandomForestClassifier",
        "LogisticRegression",
    }
    assert isinstance(result["best_search_score"], float)
    assert result["report"] is not None
    assert result["report"]["scorer"] == "accuracy"


def test_run_pipeline_reuses_provided_spec_offline(monkeypatch):
    # spec_raw= skips the agents entirely (the sweep harness fetches the spec
    # once per task); proving it by making any agent run blow up the test.
    def _boom(*a, **k):
        raise AssertionError("agents must not run when spec_raw is provided")

    monkeypatch.setattr(pipeline, "_run_agents", _boom)
    result = pipeline.run_pipeline(
        task_name="california-housing-prices", budget=4, spec_raw=SPEC_JSON
    )
    assert result["reused_spec"] is True
    assert result["llm_calls"] == 0
    assert not result["used_fallback_spec"]
    assert result["analysis"] == ""  # degrades gracefully without the analyst
    assert isinstance(result["best_search_score"], float)


def test_run_pipeline_forwards_c_and_retarget(monkeypatch):
    captured = {}

    def fake_search_loop(spec, df, target, **kwargs):
        captured.update(kwargs)
        return {
            "best_state": {},
            "best_score": 0.5,
            "plan": None,
            "action_space": {},
            "target_key": None,
            "injected_options": [],
            "score_cache": {},
        }

    monkeypatch.setattr(pipeline, "run_search_loop", fake_search_loop)
    pipeline.run_pipeline(
        task_name="california-housing-prices",
        budget=4,
        spec_raw=SPEC_JSON,
        c=0.9,
        retarget=False,
        seed=7,
    )
    assert captured["c"] == 0.9
    assert captured["retarget"] is False
    assert captured["seed"] == 7
