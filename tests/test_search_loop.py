"""Integration tests for the outer search loop (Option 1 + Option 3), offline.

Real skrub + MCTS on california-housing; the LLM proposer is a stage-aware fake,
so no network. Covers tree persistence, ablation targeting, model-gated HP
canonicalization, and the headline: a run keeps an option not in the plan.
"""

import os

import pandas as pd

from machine_learning_engineering import spec_resolver
from machine_learning_engineering.skrub_ops import get_choice_gating
from machine_learning_engineering.search_loop import _parse_paths, run_search_loop


def test_parse_paths_handles_json_fences_and_prose():
    assert _parse_paths('["sklearn.preprocessing.RobustScaler"]') == [
        "sklearn.preprocessing.RobustScaler"
    ]
    assert _parse_paths('```json\n["skrub.MinHashEncoder"]\n```') == [
        "skrub.MinHashEncoder"
    ]
    # prose fallback extracts dotted sklearn/skrub paths
    assert _parse_paths("I suggest sklearn.linear_model.Ridge here.") == [
        "sklearn.linear_model.Ridge"
    ]
    assert _parse_paths("no operators here") == []


TARGET = "median_house_value"


def _california() -> pd.DataFrame:
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "machine_learning_engineering",
        "tasks",
        "california-housing-prices",
        "train.csv",
    )
    return pd.read_csv(path)


# A minimal spec, so proposed options are genuinely new to the plan.
RAW = {
    "encoder_options": ["skrub.GapEncoder"],
    "stages": [
        {"name": "scale", "options": ["skip", "sklearn.preprocessing.StandardScaler"]}
    ],
    "model": [
        "sklearn.ensemble.HistGradientBoostingRegressor",
        "sklearn.ensemble.RandomForestRegressor",
    ],
}

# Per-stage new options not present in RAW.
_NEW_PER_STAGE = {
    "scale": ["sklearn.preprocessing.RobustScaler"],
    "encoder": ["skrub.MinHashEncoder"],
    "clean": ["skrub.Cleaner"],
    "model": ["sklearn.linear_model.Ridge"],
}


def _spec():
    return spec_resolver.resolve_spec(RAW, task_type="regression")


def test_tree_persists_across_outer_steps():
    res = run_search_loop(
        _spec(), _california(), TARGET, scoring="r2", outer_steps=2, budget_per_step=6
    )
    # backprop increments root.N once per iteration across both phases (no reset)
    assert res["root"].N == 2 * 6
    # score cache holds one entry per distinct evaluated state
    assert len(res["score_cache"]) >= 1


def test_targeting_picks_an_operator_stage():
    res = run_search_loop(
        _spec(), _california(), TARGET, scoring="r2", outer_steps=2, budget_per_step=8
    )
    assert res["target_key"] is not None
    assert "__" not in res["target_key"]  # an operator/model stage, not an HP key


def test_run_keeps_an_option_not_in_the_plan():
    calls = {"n": 0}

    def fake_propose(stage_key, ledger, vocab):
        calls["n"] += 1
        return _NEW_PER_STAGE.get(stage_key, [])

    res = run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=3,
        budget_per_step=8,
        propose=fake_propose,
    )

    assert res["injected_options"]  # at least one new option was injected
    tk = res["target_key"]
    injected_classes = {p.rsplit(".", 1)[-1] for p in res["injected_options"]}
    # the injected option(s) now appear in the (rebuilt) action space for the stage
    assert injected_classes & set(res["action_space"].get(tk, []))
    # one LLM call per refinement step, never per rollout
    assert calls["n"] <= 3


def test_run_states_are_model_gated_canonical():
    res = run_search_loop(
        _spec(), _california(), TARGET, scoring="r2", outer_steps=2, budget_per_step=8
    )
    gating = get_choice_gating(res["plan"])
    # no cached state carries a hyperparameter for a non-selected model
    for key in res["score_cache"]:
        state = dict(key)
        for k in state:
            if k in gating:
                parent, activating = gating[k]
                assert state.get(parent) == activating
