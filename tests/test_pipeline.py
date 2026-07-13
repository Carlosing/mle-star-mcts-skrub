"""End-to-end pipeline tests — agents mocked, skrub+MCTS real, fully offline."""

import asyncio

import pytest
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


class UsageFakeLlm(FakeLlm):
    """FakeLlm whose responses carry a usage_metadata block (per turn)."""

    async def generate_content_async(self, llm_request, stream: bool = False):
        text = (
            self._responses[min(self._idx, len(self._responses) - 1)]
            if self._responses
            else ""
        )
        self._idx += 1
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100,
                candidates_token_count=250,
                total_token_count=350,
            ),
        )


def test_token_capture_populates_result():
    # each agent turn reports 350 tokens; two turns (analyst + plan_author) -> 700.
    model = UsageFakeLlm().set_responses([ANALYSIS, SPEC_JSON])
    result = pipeline.run_pipeline(
        task_name="california-housing-prices",
        budget=4,
        model=model,
        with_search=False,
    )
    assert result["llm_calls"] == 2
    tokens = result["tokens"]
    assert tokens["total"] == tokens["prompt"] + tokens["completion"]
    assert tokens["total"] == 700  # 2 turns x 350
    by_agent = result["tokens_by_agent"]
    assert set(by_agent) == {"data_analyst", "plan_author"}
    assert by_agent["plan_author"]["total"] == 350


def test_token_capture_zero_without_usage():
    # a plain FakeLlm reports no usage -> zero tokens, but calls still counted.
    model = FakeLlm().set_responses([ANALYSIS, SPEC_JSON])
    result = pipeline.run_pipeline(
        task_name="california-housing-prices",
        budget=4,
        model=model,
        with_search=False,
    )
    assert result["tokens"]["total"] == 0
    assert result["tokens_by_agent"]["plan_author"]["calls"] == 1


def test_holdout_score_matches_top_ensemble_member():
    # the incumbent's shared-holdout score == the ensemble's top individual
    # score (same split, same scorer, k=1 slice of the same fit path).
    model = FakeLlm().set_responses([ANALYSIS, SPEC_JSON])
    result = pipeline.run_pipeline(
        task_name="california-housing-prices",
        budget=8,
        model=model,
        with_search=False,
        top_k=3,
    )
    assert result["holdout"] is not None
    assert result["holdout"]["scorer"] == "neg_root_mean_squared_error"
    assert result["holdout"]["score"] == pytest.approx(
        result["ensemble"]["individual_scores"][0]
    )


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
    # spec_raw= skips the agents entirely (the offline Claude driver and
    # stored-run replays rely on this); proving it by making any agent run
    # blow up the test.
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
    result = pipeline.run_pipeline(
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
    # regression -> the auto seed-averaging factor resolves to 1
    assert captured["n_subsample_seeds"] == 1
    assert result["subsample_seeds"] == 1


def test_auto_subsample_seeds_targets_imbalanced_classification():
    import pandas as pd

    balanced = pd.DataFrame({"y": ["a", "b"] * 50})
    imbalanced = pd.DataFrame({"y": ["neg"] * 990 + ["pos"] * 10})
    assert pipeline._auto_subsample_seeds(balanced, "y", "classification") == 1
    assert (
        pipeline._auto_subsample_seeds(imbalanced, "y", "classification") == 3
    )
    # regression never averages, whatever the target looks like
    assert pipeline._auto_subsample_seeds(imbalanced, "y", "regression") == 1


# --- transient provider errors ------------------------------------------------


class _Session:
    id = "s"
    user_id = "driver"


async def _no_sleep(*a, **k):
    return None


def test_is_transient_retries_5xx_but_not_quota_or_auth():
    """Gemini 503s the grounded agent calls while answering a small ping, so a
    run dies at data_analyst/plan_author. Quota and auth never self-clear."""
    assert pipeline._is_transient(RuntimeError("503 UNAVAILABLE"))
    assert pipeline._is_transient(RuntimeError("500 INTERNAL"))
    assert pipeline._is_transient(RuntimeError("model is overloaded"))
    assert not pipeline._is_transient(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not pipeline._is_transient(RuntimeError("401 UNAUTHENTICATED"))


def test_run_agents_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _FakeRunner:
        def __init__(self, *a, **k):
            self.app_name = "app"
            self.session_service = self

        async def create_session(self, **k):
            return _Session()

        async def get_session(self, **k):
            return "SESSION"

        async def _gen(self):
            if False:
                yield None

        def run_async(self, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("503 UNAVAILABLE")
            return self._gen()

    monkeypatch.setattr(pipeline, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(pipeline.asyncio, "sleep", _no_sleep)

    got = asyncio.run(pipeline._run_agents(object(), "hi", backoff_s=0.0))
    assert got == "SESSION"
    assert calls["n"] == 3


def test_run_agents_reraises_non_transient_immediately(monkeypatch):
    calls = {"n": 0}

    class _FakeRunner:
        def __init__(self, *a, **k):
            self.app_name = "app"
            self.session_service = self

        async def create_session(self, **k):
            return _Session()

        def run_async(self, **k):
            calls["n"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(pipeline, "InMemoryRunner", _FakeRunner)
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        asyncio.run(pipeline._run_agents(object(), "hi", backoff_s=0.0))
    assert calls["n"] == 1  # no retry burned on a quota wall
