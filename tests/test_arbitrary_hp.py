"""Arbitrary (non-curated) hyperparameter tuning + per-param safety nets.

The safety envelope is at the IMPORT level (allow-list roots), not the search
level: the LLM may tune any hyperparameter the operator's constructor accepts,
with the range it gave (no clipping). Two params are dropped *individually*
without dropping the operator — one the class does not accept, and any
RNG-identity param (determinism). Driven by a stored plan_author answer
(`fixtures/arbitrary_hp_agent_io`), so no API call.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("skrub")

from machine_learning_engineering import spec_resolver
from machine_learning_engineering.skrub_ops import (
    build_staged_plan,
    get_action_space,
)

from fixtures.arbitrary_hp_agent_io import SKRUB_SPEC_RAW


def _df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )


# --- end-to-end: the stored answer resolves + surfaces in the action space ----


def test_arbitrary_hps_resolve_and_surface_in_action_space():
    spec = spec_resolver.resolve_spec(SKRUB_SPEC_RAW, task_type="regression")
    space = get_action_space(build_staged_plan(spec, _df(), target="target"))
    keys = set(space)

    # arbitrary, valid HPs became searchable dims (free-form, no registry rule)
    assert "model__HistGradientBoostingRegressor__max_leaf_nodes" in keys
    assert "model__RandomForestRegressor__min_impurity_decrease" in keys
    assert "model__Lasso__alpha" in keys  # non-registry model, fully free-form
    assert "feature_eng__PCA__n_components" in keys  # non-curated transformer

    # a curated HP still works alongside the arbitrary ones
    assert "model__HistGradientBoostingRegressor__learning_rate" in keys

    # every proposed model was built, including the non-registry Lasso
    assert set(space["model"]) == {
        "HistGradientBoostingRegressor",
        "RandomForestRegressor",
        "Lasso",
    }


def test_invalid_param_dropped_without_dropping_operator():
    spec = spec_resolver.resolve_spec(SKRUB_SPEC_RAW, task_type="regression")
    space = get_action_space(build_staged_plan(spec, _df(), target="target"))
    keys = set(space)

    # RF has no `learning_rate`; the param is omitted but RF still built + tuned
    assert "model__RandomForestRegressor__learning_rate" not in keys
    assert "RandomForestRegressor" in space["model"]
    assert "model__RandomForestRegressor__n_estimators" in keys

    # a tuned random_state is never a search dimension (determinism invariant)
    assert not any(k.endswith("__random_state") for k in keys)


def test_arbitrary_ranges_used_as_given_not_clipped():
    spec = spec_resolver.resolve_spec(SKRUB_SPEC_RAW, task_type="regression")
    space = get_action_space(build_staged_plan(spec, _df(), target="target"))

    pca = space["feature_eng__PCA__n_components"]  # LLM asked int[2, 6]
    assert min(pca) == 2 and max(pca) == 6

    alpha = space["model__Lasso__alpha"]  # LLM asked float[1e-4, 10] log
    assert min(alpha) == pytest.approx(1e-4) and max(alpha) == pytest.approx(
        10.0
    )


# --- unit tests for the safety nets ------------------------------------------


def test_make_keeps_operator_when_a_param_is_invalid():
    inst = spec_resolver._make(
        "sklearn.ensemble.RandomForestRegressor",
        {
            "n_estimators": {"int": [50, 200]},
            "learning_rate": {"float": [0.1, 0.5]},
        },
        seed=42,
        context="model",
    )
    # would be None if the invalid kwarg had been passed to the constructor
    assert inst is not None and type(inst).__name__ == "RandomForestRegressor"
    assert "choose_int" in repr(inst.get_params()["n_estimators"])


def test_make_refuses_to_tune_random_state():
    inst = spec_resolver._make(
        "sklearn.linear_model.Lasso",
        {"alpha": {"float": [1e-3, 1.0]}, "random_state": {"int": [0, 9]}},
        seed=42,
        context="model",
    )
    assert inst is not None
    # random_state stays the central seed (an int), not a choose_* node
    assert inst.get_params()["random_state"] == spec_resolver.SEED


def test_accepts_param_reads_the_constructor_signature():
    from sklearn.ensemble import RandomForestRegressor

    assert spec_resolver._accepts_param(RandomForestRegressor, "n_estimators")
    assert not spec_resolver._accepts_param(
        RandomForestRegressor, "learning_rate"
    )


def test_build_free_choice_structural_sanity():
    bc = spec_resolver._build_free_choice
    assert bc("n", {"int": [2, 6]}) is not None
    assert bc("n", {"float": [0.1, 1.0], "log": True}) is not None
    assert bc("n", {"choice": ["a", "b"]}) is not None
    assert bc("n", {"type": "int", "range": [1, 9]}) is not None  # aliased form
    assert bc("n", {"float": [1.0, 1.0]}) is None  # low == high
    assert (
        bc("n", {"float": [-1.0, 1.0], "log": True}) is None
    )  # log needs low > 0
    assert bc("n", {"int": ["x", "y"]}) is None  # non-numeric bounds
    assert bc("n", "not-a-dict") is None
