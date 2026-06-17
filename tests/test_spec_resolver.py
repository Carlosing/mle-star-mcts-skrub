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
