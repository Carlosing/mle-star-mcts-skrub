"""Integration tests for the outer search loop (Option 1 + Option 3), offline.

Real skrub + MCTS on california-housing; the LLM proposer is a stage-aware fake,
so no network. Covers tree persistence, ablation targeting, model-gated HP
canonicalization, and the headline: a run keeps an option not in the plan.
"""

import os

import pandas as pd

from machine_learning_engineering import spec_resolver
from machine_learning_engineering.skrub_ops import get_choice_gating
from machine_learning_engineering.search_loop import (
    _merge_raw_plans,
    run_search_loop,
)


def test_merge_raw_plans_is_strictly_additive():
    old = {
        "encoder_options": ["skrub.GapEncoder"],
        "stages": [{"name": "scale", "options": ["skip"]}],
        "model": ["sklearn.linear_model.Ridge"],
    }
    new = {
        "encoder_options": ["skrub.GapEncoder", "skrub.MinHashEncoder"],
        "stages": [
            {"name": "scale", "options": ["sklearn.preprocessing.RobustScaler"]},
            {"name": "feature_eng", "options": ["sklearn.preprocessing.PolynomialFeatures"]},
        ],
        "model": [
            {
                "name": "lightgbm.LGBMRegressor",
                "params": {"n_estimators": {"int": [100, 600]}},
            }
        ],
    }
    merged = _merge_raw_plans(old, new)
    # unions by path: the duplicate GapEncoder is dropped, the new encoder kept
    assert merged["encoder_options"] == [
        "skrub.GapEncoder",
        "skrub.MinHashEncoder",
    ]
    # a matching stage only grows its option menu; a new stage appends whole
    assert merged["stages"][0]["options"] == [
        "skip",
        "sklearn.preprocessing.RobustScaler",
    ]
    assert merged["stages"][1]["name"] == "feature_eng"
    # an injected model keeps its params (it enters the search tuned)
    assert merged["model"][0] == "sklearn.linear_model.Ridge"  # untouched
    assert merged["model"][1]["params"]  # tuned addition
    # the old plan object was not mutated
    assert old["stages"][0]["options"] == ["skip"]
    assert old["model"] == ["sklearn.linear_model.Ridge"]


def test_merge_raw_plans_never_modifies_existing_entries():
    old = {"model": ["sklearn.linear_model.Ridge"]}
    # a proposal re-listing an existing operator WITH params must be ignored:
    # mutating an existing entry would change its action-space label and
    # orphan already-scored tree states
    new = {
        "model": [
            {
                "name": "sklearn.linear_model.Ridge",
                "params": {"alpha": {"float": [0.001, 1000.0]}},
            }
        ]
    }
    assert _merge_raw_plans(old, new) == old


def test_merge_raw_plans_treats_missing_stages_as_no_change():
    old = {
        "encoder_options": ["skrub.GapEncoder"],
        "model": ["sklearn.linear_model.Ridge"],
    }
    assert _merge_raw_plans(old, {}) == old
    # a partial proposal touches only the stage it names
    merged = _merge_raw_plans(old, {"model": ["sklearn.linear_model.Lasso"]})
    assert merged["encoder_options"] == ["skrub.GapEncoder"]
    assert len(merged["model"]) == 2


def test_merge_raw_plans_appends_new_scoped_groups_and_assemble():
    old = {
        "scoped_encodings": [
            {"name": "g1", "cols": ["a"], "options": ["skrub.GapEncoder"]}
        ],
        "assemble": [{"name": "j1", "table": "aux", "operations": ["mean"]}],
        "model": ["sklearn.linear_model.Ridge"],
    }
    new = {
        "scoped_encodings": [
            {"name": "g1", "cols": ["IGNORED"], "options": ["skrub.MinHashEncoder"]},
            {"name": "g2", "cols": ["b"], "options": ["skrub.StringEncoder"], "additive": True},
        ],
        "assemble": [
            {"name": "j1", "table": "HIJACKED", "operations": ["max"]},
            {"name": "j2", "table": "aux", "operations": ["max"]},
        ],
    }
    merged = _merge_raw_plans(old, new)
    g1, g2 = merged["scoped_encodings"]
    assert g1["cols"] == ["a"]  # an existing group's cols never change
    assert g1["options"] == ["skrub.GapEncoder", "skrub.MinHashEncoder"]
    assert g2["name"] == "g2" and g2["additive"] is True
    j1, j2 = merged["assemble"]
    assert j1["table"] == "aux"  # an existing join config never changes
    assert j2["name"] == "j2"


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

# An extension plan not present in RAW: a tuned model + a new scale option.
_EXTENSION = {
    "model": [
        {
            "name": "sklearn.linear_model.Ridge",
            "params": {"alpha": {"float": [0.01, 100.0], "log": True}},
        }
    ],
    "stages": [
        {"name": "scale", "options": ["sklearn.preprocessing.RobustScaler"]}
    ],
}


def _spec():
    return spec_resolver.resolve_spec(RAW, task_type="regression")


def _resolve(plan_json):
    return spec_resolver.resolve_spec(plan_json, task_type="regression")


def test_tree_persists_across_outer_steps():
    res = run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=2,
        budget_per_step=6,
    )
    # backprop increments root.N once per iteration across all phases (no
    # reset): 2 slices of 6 + the ceil(12/4) focused-refinement bonus
    assert res["root"].N == 2 * 6 + 3
    # score cache holds one entry per distinct evaluated state
    assert len(res["score_cache"]) >= 1


def test_targeting_picks_an_operator_stage():
    res = run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=2,
        budget_per_step=8,
    )
    assert res["target_key"] is not None
    assert (
        "__" not in res["target_key"]
    )  # an operator/model stage, not an HP key


def test_run_keeps_an_option_not_in_the_plan():
    calls = {"n": 0}

    def fake_propose(plan_json, context):
        calls["n"] += 1
        return _EXTENSION

    res = run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=3,
        budget_per_step=8,
        propose=fake_propose,
        raw_spec=RAW,
        resolve=_resolve,
    )

    assert res["injected_options"]  # at least one new option was injected
    # the injected model + scale option live in the rebuilt action space
    assert "Ridge" in res["action_space"]["model"]
    assert any("RobustScaler" in lbl for lbl in res["action_space"]["scale"])
    # the injected model arrived TUNED: its HP became a new gated dimension
    gating = get_choice_gating(res["plan"])
    hp_dims = [k for k, v in gating.items() if v == ("model", "Ridge")]
    assert hp_dims and all(k in res["action_space"] for k in hp_dims)
    assert any(k in res["injected_options"] for k in hp_dims)
    # one LLM call per refinement step, never per rollout
    assert calls["n"] <= 3


def test_interleaved_propose_fires_between_every_slice():
    calls = []

    def fake_propose(plan_json, context):
        calls.append(context.get("target_stage"))
        return {}  # nothing to add: the search continues unchanged

    run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=4,
        budget_per_step=3,
        propose=fake_propose,
        raw_spec=RAW,
        resolve=_resolve,
    )
    # search 3 -> propose -> 3 -> propose -> 3 -> propose -> 3
    assert len(calls) == 3


def test_propose_needs_raw_spec_and_resolve():
    calls = {"n": 0}

    def fake_propose(plan_json, context):
        calls["n"] += 1
        return _EXTENSION

    # without the raw plan + resolver, Option 3 is off: no proposer call
    run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=2,
        budget_per_step=3,
        propose=fake_propose,
    )
    assert calls["n"] == 0


def test_retargeting_reruns_each_slice_and_can_be_disabled(monkeypatch):
    from machine_learning_engineering import search_loop as sl

    picks = {"n": 0}
    orig = sl.pick_target_node

    def counting_pick(ledger):
        picks["n"] += 1
        return orig(ledger)

    monkeypatch.setattr(sl, "pick_target_node", counting_pick)
    sl.run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=3,
        budget_per_step=4,
    )
    assert picks["n"] == 2  # after slice 0 and slice 1, never after the last
    picks["n"] = 0
    sl.run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=3,
        budget_per_step=4,
        retarget=False,
    )
    assert picks["n"] == 1  # the first pick is kept for the whole run


def test_score_cache_bounds_cv_calls_across_slices():
    res = run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=3,
        budget_per_step=4,
    )
    # each MCTS iteration evaluates at most one NEW state; the persisted cache
    # is exact dedup, so distinct CV evaluations never exceed the total budget
    # (3 slices of 4 + the ceil(12/4) bonus phase) plus the root
    assert len(res["score_cache"]) <= 3 * 4 + 3 + 1


# A single HP-tuned model. The lone encoder gains skrub's default companion
# (resolver repair), so the searchable dims are its gated HPs PLUS the 2-option
# encoder — the bonus phase refines both structure and HPs.
_HP_RAW = {
    "encoder_options": ["skrub.GapEncoder"],
    "model": [
        {
            "name": "sklearn.ensemble.HistGradientBoostingRegressor",
            "params": {
                "learning_rate": {"float": [0.01, 0.3], "log": True},
                "max_iter": {"int": [100, 600]},
                "max_depth": {"int": [2, 16]},
            },
        }
    ],
}


def test_bonus_phase_refines_incumbent_hps_after_main_budget():
    spec = spec_resolver.resolve_spec(_HP_RAW, task_type="regression")
    b = 6
    res = run_search_loop(
        spec, _california(), TARGET, scoring="r2", budget_per_step=b
    )
    # the bonus phase ran ceil(b/4) extra rollouts on the SAME persisted root
    assert res["root"].N == b + -(-b // 4)
    # the bonus phase refines single-edit neighbors — gated HPs AND structural
    # stages (the now-2-option encoder) — so every refined dim is a real search
    # dimension, and at least one gated HP was among them
    gating = get_choice_gating(res["plan"])
    assert res["refined_dims"]
    assert all(
        k in gating or k in res["action_space"] for k in res["refined_dims"]
    )
    assert any(k in gating for k in res["refined_dims"])
    # HP space was actually explored: some evaluated config sets a gated HP
    assert any(k in gating for key in res["score_cache"] for k in dict(key))


def test_bonus_phase_explores_structural_neighbors_and_off_switch():
    from machine_learning_engineering import search_loop as sl

    # bare spec (no tunable HPs): the old HP-only phase no-op'd here; the
    # focused phase now explores the incumbent's structural neighbors.
    # A 3-scale x 3-model space so budget 5 can't exhaust it before the bonus.
    struct_raw = {
        "encoder_options": ["skrub.GapEncoder"],
        "stages": [
            {
                "name": "scale",
                "options": [
                    "skip",
                    "sklearn.preprocessing.StandardScaler",
                    "sklearn.preprocessing.RobustScaler",
                ],
            }
        ],
        "model": [
            "sklearn.ensemble.HistGradientBoostingRegressor",
            "sklearn.ensemble.RandomForestRegressor",
            "sklearn.linear_model.Ridge",
        ],
    }
    res = sl.run_search_loop(
        spec_resolver.resolve_spec(struct_raw, task_type="regression"),
        _california(),
        TARGET,
        scoring="r2",
        budget_per_step=5,
    )
    assert res["root"].N == 5 + 2  # ceil(5/4) bonus rollouts ran
    assert res["refined_dims"]  # and they edited real dims...
    assert all(
        "__" not in k for k in res["refined_dims"]
    )  # ...all structural (scale/model/encoder), no HP dims exist here

    # hp_refine=False skips the bonus phase entirely
    spec = spec_resolver.resolve_spec(_HP_RAW, task_type="regression")
    off = sl.run_search_loop(
        spec,
        _california(),
        TARGET,
        scoring="r2",
        budget_per_step=6,
        hp_refine=False,
    )
    assert off["root"].N == 6 and off["refined_dims"] == []


def test_run_states_are_model_gated_canonical():
    res = run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=2,
        budget_per_step=8,
    )
    gating = get_choice_gating(res["plan"])
    # no cached state carries a hyperparameter for a non-selected model
    for key in res["score_cache"]:
        state = dict(key)
        for k in state:
            if k in gating:
                parent, activating = gating[k]
                assert state.get(parent) == activating


def test_booster_plan_rolls_out_at_full_n_jobs():
    """CV fold-parallelism is inter-process (joblib), orthogonal to the
    intra-process OpenMP double-init that segfaults boosters (that trigger is
    the estimator's own n_jobs, pinned to 1 in REGISTRY). A booster plan must
    therefore roll out end-to-end at the requested n_jobs without crashing."""
    import numpy as np
    import pandas as pd

    from machine_learning_engineering.spec_resolver import resolve_spec

    n = 200
    rs = np.random.RandomState(0)
    df = pd.DataFrame(
        {"a": rs.rand(n), "b": [f"t{i % 5}" for i in range(n)],
         "y": [i % 2 for i in range(n)]}
    )
    spec = resolve_spec(
        {"model": ["lightgbm.LGBMClassifier"]}, task_type="classification"
    )
    out = run_search_loop(
        spec, df, "y", scoring="accuracy", budget_per_step=6,
        stratify=True, hp_refine=False, n_jobs=6,
    )
    assert out["best_score"] > 0.0  # searched + fit boosters, no segfault


def test_proposal_injection_error_is_surfaced():
    """A proposal that merges but won't resolve/build is dropped so the search
    continues — but the reason is recorded, so an empty injected_options from a
    FAILED injection is distinguishable from one where nothing was proposed."""
    def fake_propose(plan_json, context):
        return _EXTENSION  # a real, mergeable extension (Ridge + RobustScaler)

    def failing_resolve(plan):
        raise ValueError("simulated unresolvable extension")

    res = run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=2,
        budget_per_step=6,
        propose=fake_propose,
        raw_spec=RAW,
        resolve=failing_resolve,
    )

    assert res["proposal_injection_error"] is not None
    assert "simulated unresolvable extension" in res["proposal_injection_error"]
    assert res["injected_options"] == []  # the failed injection was NOT applied


def test_no_proposal_injection_error_when_injection_succeeds():
    """Baseline: a resolvable proposal injects and leaves the error field None."""
    res = run_search_loop(
        _spec(),
        _california(),
        TARGET,
        scoring="r2",
        outer_steps=2,
        budget_per_step=6,
        propose=lambda plan_json, context: _EXTENSION,
        raw_spec=RAW,
        resolve=_resolve,
    )
    assert res["proposal_injection_error"] is None
    assert res["injected_options"]  # it did inject
