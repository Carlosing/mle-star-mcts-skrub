"""prior_fn warm-start (Item: free-form LLM priors) + richer proposer protocol.

The prior is emitted by the plan author IN ITS EXISTING CALL (zero extra LLM
calls): options may carry `"prior": 0.0-1.0`, the resolver collects them into
`spec["priors"]`, and the search loop turns that into a pure-lookup `prior_fn`
that seeds every fresh child's Q/N. The proposer side: `run_search_loop` hands
the proposer the WHOLE current raw plan plus an evidence context (cross-stage
ledger, incumbent, digest, columns, target stage, vocabulary) and merges the
returned extension plan additively — including new scoped_encodings groups.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("skrub")

from machine_learning_engineering import mcts, spec_resolver
from machine_learning_engineering.search_loop import (
    _make_prior_fn,
    _parse_plan,
    make_llm_proposer,
    run_search_loop,
)

from fixtures.golden_plan import make_toy_df


# --- resolver collects priors ----------------------------------------------------


def test_resolver_collects_priors_across_stages():
    raw = {
        "vectorizer": {
            "slots": {
                "high_cardinality": [
                    {"name": "skrub.GapEncoder", "prior": 0.9},
                    "skrub.MinHashEncoder",
                ]
            }
        },
        "scoped_encodings": [
            {
                "name": "color_enc",
                "cols": ["color"],
                "options": [{"name": "skrub.StringEncoder", "prior": 0.7}],
            }
        ],
        "model": [
            {
                "name": "sklearn.ensemble.HistGradientBoostingClassifier",
                "prior": 0.8,
            },
            {"name": "sklearn.linear_model.LogisticRegression", "prior": 2.5},
        ],
    }
    spec = spec_resolver.resolve_spec(
        raw, task_type="classification", main_columns=["x1", "x2", "color"]
    )
    priors = spec["priors"]
    assert priors["model"]["HistGradientBoostingClassifier"] == 0.8
    assert priors["model"]["LogisticRegression"] == 1.0  # clipped to [0, 1]
    assert priors["scope_color_enc"]["StringEncoder"] == 0.7
    # backbone slot priors are keyed by the slot's choice name -> option repr
    hc = priors["vectorizer__high_cardinality"]
    assert any(k.startswith("GapEncoder") for k in hc)
    # options without a prior simply have no entry (prior_fn defaults to 0.5)
    assert not any(k.startswith("MinHashEncoder") for k in hc)


def test_spec_without_priors_has_no_priors_key():
    spec = spec_resolver.resolve_spec(
        {"model": ["sklearn.linear_model.LogisticRegression"]},
        task_type="classification",
    )
    assert "priors" not in spec


# --- prior_fn seeds every fresh child ---------------------------------------------


def test_prior_fn_seeds_all_children_and_orders_by_prior():
    parent = mcts.MCTSNode(state={"model": "A"})
    parent.N = 1  # already visited, as at expansion time
    rated = mcts.MCTSNode(state={"model": "B"}, parent=parent)
    unrated = mcts.MCTSNode(state={"model": "C"}, parent=parent)
    _make_prior_fn({"model": {"B": 0.9}})([rated, unrated])
    assert (rated.Q, rated.N) == (0.9, 1)
    assert (unrated.Q, unrated.N) == (
        0.5,
        1,
    )  # neutral default, NOT left at inf
    assert rated.uct(0.5) > unrated.uct(0.5)


def test_priors_steer_the_first_rollouts():
    """With a prior pointing at the best option, the incumbent is found in
    fewer evaluations than exploring in declaration order."""
    action_space = {"model": ["A", "B", "C", "D"]}
    rewards = {"A": 0.2, "B": 0.3, "C": 0.4, "D": 0.9}

    calls: list = []

    def rollout(state):
        calls.append(dict(state))
        return rewards[state["model"]]

    best, score, _ = mcts.mcts_search(
        {"model": "A"},
        action_space,
        rollout,
        budget=2,
        prior_fn=_make_prior_fn({"model": {"D": 0.95}}),
        score_cache={},
    )
    assert best == {"model": "D"}  # found within 2 rollouts thanks to the prior
    assert score == 0.9


def test_run_search_loop_uses_spec_priors_offline():
    df = make_toy_df()
    raw = {
        "model": [
            {
                "name": "sklearn.ensemble.HistGradientBoostingClassifier",
                "prior": 0.9,
            },
            {"name": "sklearn.linear_model.LogisticRegression", "prior": 0.2},
        ],
    }
    spec = spec_resolver.resolve_spec(raw, task_type="classification")
    result = run_search_loop(
        spec, df, "target", scoring="accuracy", budget_per_step=4
    )
    assert isinstance(result["best_score"], float)
    assert result["best_score"] > 0.0


# --- proposer protocol: plan parsing + evidence context ---------------------


def test_parse_plan_accepts_fenced_json_objects():
    assert _parse_plan('{"model": ["lightgbm.LGBMRegressor"]}') == {
        "model": ["lightgbm.LGBMRegressor"]
    }
    assert _parse_plan(
        '```json\n{"vectorizer": {"slots": {"high_cardinality": ["skrub.MinHashEncoder"]}}}\n```'
    ) == {"vectorizer": {"slots": {"high_cardinality": ["skrub.MinHashEncoder"]}}}
    # prose around the object is tolerated (parse_spec_json brace-scan)
    assert _parse_plan('Here you go: {"model": ["x.Y"]} hope it helps') == {
        "model": ["x.Y"]
    }


def test_parse_plan_rejects_non_objects():
    assert _parse_plan("") is None
    assert _parse_plan("no JSON here") is None
    # the old contract's array shape is NOT a plan
    assert _parse_plan('["skrub.MinHashEncoder"]') is None


def test_proposer_receives_whole_plan_and_scoped_extension_injects():
    df = make_toy_df()
    raw = {
        "vectorizer": {"slots": {"high_cardinality": ["skrub.GapEncoder"]}},
        "model": [
            "sklearn.ensemble.HistGradientBoostingClassifier",
            "sklearn.linear_model.LogisticRegression",
        ],
    }
    cols = [c for c in df.columns if c != "target"]
    spec = spec_resolver.resolve_spec(
        raw, task_type="classification", main_columns=cols
    )
    seen: dict = {"plans": []}

    def fake_propose(plan_json, context):
        seen["plans"].append(plan_json)
        seen["context"] = context
        return {
            "scoped_encodings": [
                {
                    "name": "color_mh",
                    "cols": ["color"],
                    "options": ["skrub.MinHashEncoder"],
                }
            ]
        }

    result = run_search_loop(
        spec,
        df,
        "target",
        scoring="accuracy",
        outer_steps=3,
        budget_per_step=6,
        propose=fake_propose,
        raw_spec=raw,
        resolve=lambda r: spec_resolver.resolve_spec(
            r, task_type="classification", main_columns=cols
        ),
        context_text="DIGEST SENTINEL",
    )
    # the proposer saw the WHOLE current raw plan, and on later calls the
    # already-extended plan (so it never re-proposes blind)
    assert seen["plans"][0] == raw
    assert "scoped_encodings" in seen["plans"][-1]
    ctx = seen["context"]
    assert ctx["digest"] == "DIGEST SENTINEL"
    assert ctx["columns"] == ["x1", "x2", "color"]
    assert "incumbent" in ctx and "stage_values" in ctx
    assert "target_stage" in ctx and "vocab" in ctx
    # the scoped extension became a new searchable dimension
    assert "scope_color_mh" in result["action_space"]
    assert "scope_color_mh" in result["injected_options"]


# --- proposer logging + symmetric search retry (make_llm_proposer) --------------


def _fake_genai_client(script):
    """A stand-in genai client whose generate_content follows `script(mode)`,
    where mode is 'search' when tools are attached else 'nosearch'."""

    class _Resp:
        def __init__(self, text):
            self.text = text

    class _Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            mode = "search" if config.tools else "nosearch"
            self.calls.append(mode)
            return _Resp(script(mode))

    class _Client:
        def __init__(self, *a, **k):
            self.models = _Models()

    return _Client


def test_proposer_retries_without_search_and_logs_by_call(tmp_path, monkeypatch):
    from google import genai

    # grounded (search) call returns nothing; the search-off retry returns a plan
    client_cls = _fake_genai_client(
        lambda mode: ""
        if mode == "search"
        else '{"model": ["lightgbm.LGBMRegressor"]}'
    )
    monkeypatch.setattr(genai, "Client", client_cls)

    propose = make_llm_proposer(model="gemini-2.5-flash", log_dir=str(tmp_path))
    out = propose(
        {"model": ["sklearn.linear_model.Ridge"]},
        {"digest": "d", "target_stage": "model"},
    )
    # the search-off retry recovered the plan the grounded call dropped
    assert out == {"model": ["lightgbm.LGBMRegressor"]}

    # first call logged under number 1, request + a two-record response
    assert (tmp_path / "proposer_1_request.json").exists()
    req = json.load(open(tmp_path / "proposer_1_request.json"))
    assert req[0]["call"] == 1 and req[0]["search"] is True
    resp = json.load(open(tmp_path / "proposer_1_response.json"))
    assert [r["search"] for r in resp] == [True, False]  # grounded, then retry
    assert resp[1]["retry"] is True

    # a second call is numbered 2
    propose({"model": []}, {})
    assert (tmp_path / "proposer_2_request.json").exists()


def test_proposer_no_retry_when_grounded_call_succeeds(tmp_path, monkeypatch):
    from google import genai

    client_cls = _fake_genai_client(
        lambda mode: '{"vectorizer": {"slots": {"high_cardinality": ["skrub.MinHashEncoder"]}}}'
    )
    monkeypatch.setattr(genai, "Client", client_cls)

    propose = make_llm_proposer(model="gemini-2.5-flash", log_dir=str(tmp_path))
    out = propose({"model": ["sklearn.linear_model.Ridge"]}, {"columns": ["a"]})

    assert out == {"vectorizer": {"slots": {"high_cardinality": ["skrub.MinHashEncoder"]}}}
    resp = json.load(open(tmp_path / "proposer_1_response.json"))
    assert len(resp) == 1  # grounded success -> no retry record
    assert not any("retry" in r for r in resp)
