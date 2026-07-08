"""Tests for spec_resolver: LLM dotted-path spec -> seeded instance-spec (lazy)."""

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
    "clean_options": ["skip", "skrub.Cleaner"],
    "encoder_options": ["skrub.GapEncoder", "skrub.MinHashEncoder"],
    "stages": [
        {
            "name": "scale",
            "options": ["skip", "sklearn.preprocessing.StandardScaler"],
        }
    ],
    "model": [
        "sklearn.ensemble.HistGradientBoostingRegressor",
        "sklearn.ensemble.RandomForestRegressor",
    ],
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
    fenced = (
        '```json\n{"model": ["sklearn.ensemble.RandomForestRegressor"]}\n```'
    )
    assert parse_spec_json(fenced) == {
        "model": ["sklearn.ensemble.RandomForestRegressor"]
    }
    prose = 'Here you go:\n{"model": ["sklearn.ensemble.RandomForestRegressor"]}\nThanks.'
    assert parse_spec_json(prose) == {
        "model": ["sklearn.ensemble.RandomForestRegressor"]
    }


def test_resolve_paths_to_instances():
    spec = resolve_spec(RAW, task_type="regression")

    assert None in spec["clean_options"]
    assert any(isinstance(o, skrub.Cleaner) for o in spec["clean_options"])

    assert all(o is not None for o in spec["encoder_options"])
    assert any(isinstance(o, skrub.GapEncoder) for o in spec["encoder_options"])

    assert spec["stages"][0]["name"] == "scale"
    assert None in spec["stages"][0]["options"]
    assert any(
        isinstance(o, StandardScaler) for o in spec["stages"][0]["options"]
    )

    # model path-list -> {class_name: instance} dict
    assert set(spec["model"]) == {
        "HistGradientBoostingRegressor",
        "RandomForestRegressor",
    }
    assert isinstance(
        spec["model"]["HistGradientBoostingRegressor"],
        HistGradientBoostingRegressor,
    )


def test_classifier_path_imports_classifier():
    spec = resolve_spec(
        {"model": ["sklearn.ensemble.HistGradientBoostingClassifier"]},
        task_type="classification",
    )
    assert isinstance(
        spec["model"]["HistGradientBoostingClassifier"],
        HistGradientBoostingClassifier,
    )


def test_open_vocab_path_not_in_registry_uses_defaults():
    # MinMaxScaler is importable + allowlisted but NOT in the curated REGISTRY.
    from sklearn.preprocessing import MinMaxScaler

    spec = resolve_spec(
        {
            "stages": [
                {
                    "name": "scale",
                    "options": ["sklearn.preprocessing.MinMaxScaler"],
                }
            ],
            "model": ["sklearn.ensemble.RandomForestRegressor"],
        }
    )
    assert any(
        isinstance(o, MinMaxScaler) for o in spec["stages"][0]["options"]
    )


def test_non_allowlisted_root_is_rejected():
    assert unknown_operators({"model": ["os.system"]}) == ["os.system"]
    with pytest.raises(ValueError):
        resolve_spec({"model": ["os.system"]})


def test_unimportable_paths_dropped_and_reported():
    raw = {
        "encoder_options": [
            "skrub.GapEncoder",
            "sklearn.preprocessing.NotARealThing",
        ],
        "model": [
            "sklearn.ensemble.RandomForestRegressor",
            "sklearn.bogus.Nope",
        ],
    }
    reported = unknown_operators(raw)
    assert "sklearn.preprocessing.NotARealThing" in reported
    assert "sklearn.bogus.Nope" in reported

    spec = resolve_spec(raw)
    assert len(spec["encoder_options"]) == 1  # bad encoder dropped
    assert set(spec["model"]) == {"RandomForestRegressor"}  # bad model dropped


def test_strict_raises_on_unknown():
    with pytest.raises(ValueError):
        resolve_spec(
            {
                "model": ["sklearn.ensemble.RandomForestRegressor"],
                "encoder_options": ["sklearn.bogus.Nope"],
            },
            strict=True,
        )


def test_no_usable_model_raises():
    with pytest.raises(ValueError):
        resolve_spec({"model": ["sklearn.bogus.Nope"]})


def test_seeding_is_applied():
    spec = resolve_spec({"model": ["sklearn.ensemble.RandomForestRegressor"]})
    assert (
        spec["model"]["RandomForestRegressor"].get_params()["random_state"]
        == 42
    )


def test_resolved_spec_builds_a_plan_on_california():
    df = _california()
    spec = resolve_spec(RAW, task_type="regression")
    plan = skrub_ops.build_staged_plan(spec, df, target="median_house_value")
    space = skrub_ops.get_action_space(plan)
    assert "model" in space
    assert set(space["model"]) == {
        "HistGradientBoostingRegressor",
        "RandomForestRegressor",
    }


# --- hyperparameter search ---------------------------------------------------

HP_RAW = {
    "model": [
        {
            "name": "sklearn.ensemble.RandomForestRegressor",
            "params": {"n_estimators": {"int": [100, 500]}},
        },
        {
            "name": "sklearn.ensemble.HistGradientBoostingRegressor",
            "params": {"learning_rate": {"float": [0.01, 0.3], "log": True}},
        },
    ]
}


def test_tuned_hps_surface_in_action_space():
    df = _california()
    plan = skrub_ops.build_staged_plan(
        resolve_spec(HP_RAW, task_type="regression"),
        df,
        target="median_house_value",
    )
    space = skrub_ops.get_action_space(plan)
    assert "model" in space
    assert "model__RandomForestRegressor__n_estimators" in space
    assert space["model__RandomForestRegressor__n_estimators"][0] == 100
    assert space["model__RandomForestRegressor__n_estimators"][-1] == 500
    assert "model__HistGradientBoostingRegressor__learning_rate" in space


def test_hp_range_is_clipped_to_allowed_envelope():
    raw = {
        "model": [
            {
                "name": "sklearn.ensemble.HistGradientBoostingRegressor",
                "params": {
                    "learning_rate": {"float": [1e-6, 5.0], "log": True}
                },
            }
        ]
    }
    df = _california()
    plan = skrub_ops.build_staged_plan(
        resolve_spec(raw, task_type="regression"),
        df,
        target="median_house_value",
    )
    vals = skrub_ops.get_action_space(plan)[
        "model__HistGradientBoostingRegressor__learning_rate"
    ]
    assert min(vals) >= 0.01 - 1e-9
    assert max(vals) <= 0.3 + 1e-9


def test_unknown_param_is_dropped_not_tuned():
    raw = {
        "model": [
            {
                "name": "sklearn.ensemble.RandomForestRegressor",
                "params": {"not_a_real_param": {"int": [1, 9]}},
            }
        ]
    }
    df = _california()
    plan = skrub_ops.build_staged_plan(
        resolve_spec(raw, task_type="regression"),
        df,
        target="median_house_value",
    )
    space = skrub_ops.get_action_space(plan)
    assert not any("not_a_real_param" in k for k in space)


def test_bare_path_works_without_params():
    spec = resolve_spec({"model": ["sklearn.ensemble.RandomForestRegressor"]})
    assert (
        spec["model"]["RandomForestRegressor"].get_params()["random_state"]
        == 42
    )
