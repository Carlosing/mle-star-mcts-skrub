"""Tests for spec_resolver: LLM name-spec -> seeded instance-spec (allowed list)."""

import os

import pandas as pd
import pytest
import skrub
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler

from machine_learning_engineering import skrub_ops
from machine_learning_engineering.spec_resolver import (
    parse_spec_json,
    resolve_spec,
    unknown_operators,
)

RAW = {
    "clean_options": ["skip", "Cleaner"],
    "encoder_options": ["GapEncoder", "MinHashEncoder"],
    "stages": [{"name": "scale", "options": ["skip", "StandardScaler"]}],
    "model": ["HistGradientBoosting", "RandomForest"],
}


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


def test_parse_json_fenced_and_prose():
    fenced = '```json\n{"model": ["RandomForest"]}\n```'
    assert parse_spec_json(fenced) == {"model": ["RandomForest"]}
    prose = 'Here you go:\n{"model": ["RandomForest"]}\nHope that helps.'
    assert parse_spec_json(prose) == {"model": ["RandomForest"]}


def test_resolve_maps_names_to_instances():
    spec = resolve_spec(RAW, task_type="regression")

    assert None in spec["clean_options"]
    assert any(isinstance(o, skrub.Cleaner) for o in spec["clean_options"])

    assert all(o is not None for o in spec["encoder_options"])
    assert any(isinstance(o, skrub.GapEncoder) for o in spec["encoder_options"])

    assert spec["stages"][0]["name"] == "scale"
    assert None in spec["stages"][0]["options"]
    assert any(isinstance(o, StandardScaler) for o in spec["stages"][0]["options"])

    # model name-list -> {label: instance} dict, regression variant
    assert set(spec["model"]) == {"HistGradientBoosting", "RandomForest"}
    assert isinstance(spec["model"]["HistGradientBoosting"], HistGradientBoostingRegressor)


def test_model_variant_switches_with_task_type():
    spec = resolve_spec({"model": ["HistGradientBoosting"]}, task_type="classification")
    assert isinstance(spec["model"]["HistGradientBoosting"], HistGradientBoostingClassifier)


def test_unknown_operators_are_dropped_and_reported():
    raw = {
        "encoder_options": ["GapEncoder", "NotARealEncoder"],
        "model": ["RandomForest", "Bogus"],
    }
    reported = unknown_operators(raw)
    assert "NotARealEncoder" in reported and "Bogus" in reported

    spec = resolve_spec(raw)
    assert len(spec["encoder_options"]) == 1  # unknown encoder dropped
    assert set(spec["model"]) == {"RandomForest"}  # unknown model dropped


def test_strict_raises_on_unknown():
    with pytest.raises(ValueError):
        resolve_spec({"model": ["RandomForest"], "encoder_options": ["Bogus"]}, strict=True)


def test_no_known_model_raises():
    with pytest.raises(ValueError):
        resolve_spec({"model": ["Bogus"]})


def test_seeding_is_applied():
    spec = resolve_spec({"model": ["RandomForest"]})
    assert spec["model"]["RandomForest"].get_params()["random_state"] == 42


def test_resolved_spec_builds_a_plan_on_california():
    df = _california()
    spec = resolve_spec(RAW, task_type="regression")
    plan = skrub_ops.build_staged_plan(spec, df, target="median_house_value")
    space = skrub_ops.get_action_space(plan)
    assert "model" in space
    assert set(space["model"]) == {"HistGradientBoosting", "RandomForest"}


# --- hyperparameter search ---------------------------------------------------

HP_RAW = {
    "model": [
        {"name": "RandomForest", "params": {"n_estimators": {"int": [100, 500]}}},
        {"name": "HistGradientBoosting", "params": {
            "learning_rate": {"float": [0.01, 0.3], "log": True}}},
    ]
}


def test_tuned_hps_surface_in_action_space():
    df = _california()
    plan = skrub_ops.build_staged_plan(
        resolve_spec(HP_RAW, task_type="regression"), df, target="median_house_value"
    )
    space = skrub_ops.get_action_space(plan)
    assert "model" in space
    # each tuned HP becomes its own searchable, discretized dimension
    assert "model__RandomForest__n_estimators" in space
    assert space["model__RandomForest__n_estimators"][0] == 100
    assert space["model__RandomForest__n_estimators"][-1] == 500
    assert "model__HistGradientBoosting__learning_rate" in space


def test_hp_range_is_clipped_to_allowed_envelope():
    # LLM asks for an out-of-bounds learning rate; resolver clips to [0.01, 0.3].
    raw = {"model": [{"name": "HistGradientBoosting",
                      "params": {"learning_rate": {"float": [1e-6, 5.0], "log": True}}}]}
    df = _california()
    plan = skrub_ops.build_staged_plan(
        resolve_spec(raw, task_type="regression"), df, target="median_house_value"
    )
    vals = skrub_ops.get_action_space(plan)["model__HistGradientBoosting__learning_rate"]
    assert min(vals) >= 0.01 - 1e-9
    assert max(vals) <= 0.3 + 1e-9


def test_unknown_param_is_dropped_not_tuned():
    # not_a_real_param is not in the allowed tunable set -> ignored, no crash.
    raw = {"model": [{"name": "RandomForest", "params": {"not_a_real_param": {"int": [1, 9]}}}]}
    df = _california()
    plan = skrub_ops.build_staged_plan(
        resolve_spec(raw, task_type="regression"), df, target="median_house_value"
    )
    space = skrub_ops.get_action_space(plan)
    assert not any("not_a_real_param" in k for k in space)


def test_bare_names_still_work_without_params():
    spec = resolve_spec({"model": ["RandomForest"]}, task_type="regression")
    # bare name -> plain seeded estimator, no choose_* params
    assert spec["model"]["RandomForest"].get_params()["random_state"] == 42
