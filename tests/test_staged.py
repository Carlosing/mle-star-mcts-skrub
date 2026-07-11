"""Tests for staged (constructive) plan building — run INSIDE Docker:

    docker run --rm -v "$PWD":/app mle-star python -m pytest tests/test_staged.py -v

These pin down the behavior the constructive MCTS topology relies on:
the per-stage menu becomes a searchable action space, partial states are
leak-safe when completed over defaults, and reward climbs as the pipeline is
enriched (a linear model is weak alone but strong once feature engineering is
added).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

skrub = pytest.importorskip("skrub")

from machine_learning_engineering import skrub_ops

from fixtures.staged_plan import (
    make_interaction_df,
    make_relational_data,
    relational_spec,
    staged_spec,
)


@pytest.fixture(scope="module")
def df():
    return make_interaction_df()


@pytest.fixture(scope="module")
def plan(df):
    return skrub_ops.build_staged_plan(staged_spec(), df)


def test_stage_menu_becomes_the_action_space(plan):
    space = skrub_ops.get_action_space(plan)
    assert {"clean", "encoder", "scale", "feature_eng", "model"} <= set(space)
    assert space["model"] == ["LogReg", "GBM", "RF"]
    # skip ('None') is the default (first) option for optional stages
    assert space["clean"][0] == "None"
    assert space["scale"][0] == "None"
    assert space["feature_eng"][0] == "None"


def test_default_state_is_the_skip_baseline(plan):
    default = skrub_ops.get_default_state(plan)
    assert default["scale"] == "None"
    assert default["feature_eng"] == "None"
    assert default["model"] == "LogReg"  # first model = default
    # every default label is a valid action-space option -> apply_state-safe
    space = skrub_ops.get_action_space(plan)
    for name, label in default.items():
        if name in space:
            assert label in space[name]


def test_partial_states_are_leak_safe(plan, df):
    rollout = skrub_ops.make_rollout_fn(plan, df)
    poly = skrub_ops.get_action_space(plan)["feature_eng"][1]
    enriched = rollout({"model": "LogReg", "feature_eng": poly})
    # immediately re-score LogReg alone: feature_eng must reset, not leak
    bare = rollout({"model": "LogReg"})
    assert bare < enriched
    assert bare == rollout({"model": "LogReg"})  # deterministic + still reset


def test_reward_climbs_with_enrichment(plan, df):
    """The headline: a linear model is weak alone, strong once enriched;
    a tree model is strong natively. This is constructive search's whole point."""
    rollout = skrub_ops.make_rollout_fn(plan, df)
    poly = skrub_ops.get_action_space(plan)["feature_eng"][1]

    logreg_bare = rollout({"model": "LogReg"})
    logreg_poly = rollout({"model": "LogReg", "feature_eng": poly})
    gbm_bare = rollout({"model": "GBM"})

    assert logreg_poly - logreg_bare > 0.15  # enrichment lifts the linear model
    assert gbm_bare - logreg_bare > 0.15  # tree captures interaction natively


def test_chosen_pipeline_fits_and_predicts(plan, df):
    # a partial state is fine: apply_state resets undecided stages to default
    score = skrub_ops.evaluate_full(plan, {"model": "GBM"}, df)
    assert 0.5 < score <= 1.0


# --- Relational: the assemble (join/aggregate) stage --------------------------


@pytest.fixture(scope="module")
def relational():
    main, aux = make_relational_data()
    plan = skrub_ops.build_staged_plan(relational_spec(), main, aux_tables=aux)
    return plan, main, aux


def test_assemble_stage_in_action_space_with_clean_labels(relational):
    plan, _, _ = relational
    space = skrub_ops.get_action_space(plan)
    assert "assemble" in space
    # dict-labeled options, not the giant AggJoiner repr; >=2 entries also get
    # the composed "all_aggregates" option (chains every join)
    assert space["assemble"] == [
        "skip",
        "aux_mean",
        "aux_mean_max_min",
        "all_aggregates",
    ]


def test_assemble_improves_relational_score(relational):
    """The relational headline: the target depends on a per-id aggregate of
    the auxiliary table, so the AggJoiner stage lifts a ~chance score."""
    plan, main, aux = relational
    rollout = skrub_ops.make_rollout_fn(plan, main, aux=aux, main_var="data")

    skip = rollout({"model": "GBM"})  # assemble defaults to skip
    joined = rollout({"model": "GBM", "assemble": "aux_mean"})

    assert skip < 0.7  # no aggregated feature -> near chance
    assert joined - skip > 0.15  # aggregating the aux table surfaces signal
