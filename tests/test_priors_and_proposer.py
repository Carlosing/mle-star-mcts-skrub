"""prior_fn warm-start (Item: free-form LLM priors) + richer proposer protocol.

The prior is emitted by the plan author IN ITS EXISTING CALL (zero extra LLM
calls): options may carry `"prior": 0.0-1.0`, the resolver collects them into
`spec["priors"]`, and the search loop turns that into a pure-lookup `prior_fn`
that seeds every fresh child's Q/N. The proposer side: `run_search_loop` hands
the proposer an evidence context (cross-stage ledger, incumbent, digest,
columns) and accepts scoped proposals `{"name": path, "cols": [...]}`,
optionally flagged `"additive": true` / `"position": "post_encode"`.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("skrub")

from machine_learning_engineering import mcts, spec_resolver
from machine_learning_engineering.search_loop import (
    _make_prior_fn,
    _parse_proposals,
    run_search_loop,
)

from fixtures.golden_plan import make_toy_df


# --- resolver collects priors ----------------------------------------------------


def test_resolver_collects_priors_across_stages():
    raw = {
        "encoder_options": [
            {"name": "skrub.GapEncoder", "prior": 0.9},
            "skrub.MinHashEncoder",
        ],
        "scoped_encodings": [
            {"name": "color_enc", "cols": ["color"],
             "options": [{"name": "skrub.StringEncoder", "prior": 0.7}]}
        ],
        "model": [
            {"name": "sklearn.ensemble.HistGradientBoostingClassifier", "prior": 0.8},
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
    # encoder priors are keyed by the action-space label (the instance repr)
    assert any(k.startswith("GapEncoder") for k in priors["encoder"])
    # options without a prior simply have no entry (prior_fn defaults to 0.5)
    assert not any(k.startswith("MinHashEncoder") for k in priors["encoder"])


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
    assert (unrated.Q, unrated.N) == (0.5, 1)  # neutral default, NOT left at inf
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
            {"name": "sklearn.ensemble.HistGradientBoostingClassifier", "prior": 0.9},
            {"name": "sklearn.linear_model.LogisticRegression", "prior": 0.2},
        ],
    }
    spec = spec_resolver.resolve_spec(raw, task_type="classification")
    result = run_search_loop(spec, df, "target", scoring="accuracy", budget_per_step=4)
    assert isinstance(result["best_score"], float)
    assert result["best_score"] > 0.0


# --- proposer protocol: proposals parsing + evidence context ---------------------


def test_parse_proposals_accepts_paths_and_scoped_objects():
    text = ('["skrub.MinHashEncoder", '
            '{"name": "skrub.GapEncoder", "cols": ["color", 3]}, '
            '{"cols": ["x"]}, 42]')
    assert _parse_proposals(text) == [
        "skrub.MinHashEncoder",
        {"name": "skrub.GapEncoder", "cols": ["color"]},  # non-str col dropped
    ]


def test_parse_proposals_carries_additive_and_position_flags():
    text = ('[{"name": "skrub.DatetimeEncoder", "cols": ["date"], "additive": true}, '
            '{"name": "sklearn.preprocessing.StandardScaler", "cols": ["age"], '
            '"position": "post_encode"}, '
            '{"name": "skrub.GapEncoder", "cols": ["t"], "position": "bogus"}]')
    assert _parse_proposals(text) == [
        {"name": "skrub.DatetimeEncoder", "cols": ["date"], "additive": True},
        {"name": "sklearn.preprocessing.StandardScaler", "cols": ["age"],
         "position": "post_encode"},
        {"name": "skrub.GapEncoder", "cols": ["t"]},  # invalid position dropped
    ]


def test_proposer_receives_evidence_context_and_scoped_proposals_inject():
    df = make_toy_df()
    raw = {
        "encoder_options": ["skrub.GapEncoder"],
        "model": [
            "sklearn.ensemble.HistGradientBoostingClassifier",
            "sklearn.linear_model.LogisticRegression",
        ],
    }
    spec = spec_resolver.resolve_spec(raw, task_type="classification")
    seen: dict = {}

    def fake_propose(stage_key, ledger, vocab, context):
        seen["stage_key"], seen["context"] = stage_key, context
        if stage_key == "encoder":
            return [{"name": "skrub.MinHashEncoder", "cols": ["color"]}]
        return []

    result = run_search_loop(
        spec, df, "target", scoring="accuracy",
        outer_steps=3, budget_per_step=6, propose=fake_propose,
        context_text="DIGEST SENTINEL",
    )
    ctx = seen["context"]
    assert ctx["digest"] == "DIGEST SENTINEL"
    assert ctx["columns"] == ["x1", "x2", "color"]
    assert "incumbent" in ctx and "stage_values" in ctx
    if seen["stage_key"] == "encoder":  # targeting picked the encoder stage
        # the scoped proposal became a new scope group in the action space
        assert result["injected_options"] == ["skrub.MinHashEncoder"]
        assert any(k.startswith("scope_") for k in result["action_space"])


def test_old_three_arg_proposers_still_work():
    df = make_toy_df()
    raw = {
        "encoder_options": ["skrub.GapEncoder"],
        "model": [
            "sklearn.ensemble.HistGradientBoostingClassifier",
            "sklearn.linear_model.LogisticRegression",
        ],
    }
    spec = spec_resolver.resolve_spec(raw, task_type="classification")

    def legacy_propose(stage_key, ledger, vocab):
        return []

    result = run_search_loop(
        spec, df, "target", scoring="accuracy",
        outer_steps=2, budget_per_step=5, propose=legacy_propose,
    )
    assert isinstance(result["best_score"], float)
