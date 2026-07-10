"""Tests for structure-aware rollout subsampling + seed-averaged rewards.

Pins the Upgrade-3 behavior: `_profile_subsample_n` sizes the rollout
subsample from the data profile (imbalance, high-cardinality text) instead of
row count alone, and `make_rollout_fn(n_subsample_seeds=)` averages the reward
over several seeded subsamples — the pure-code denoiser for tasks where a
single subsample draw made "more budget = worse config" possible.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

skrub = pytest.importorskip("skrub")

_BASE = os.path.join(
    os.path.dirname(__file__), "..", "machine_learning_engineering"
)
_spec = importlib.util.spec_from_file_location(
    "skrub_ops", os.path.join(_BASE, "skrub_ops.py")
)
skrub_ops = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = skrub_ops
_spec.loader.exec_module(skrub_ops)


def _imbalanced_df(n=10_000, pos_rate=0.01, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    return pd.DataFrame({"x": rng.normal(size=n) + y, "y": y})


# --- _profile_subsample_n -----------------------------------------------------


def test_profile_n_keeps_base_rule_on_plain_data():
    df = pd.DataFrame({"x": range(30_000), "y": range(30_000)})
    n, floor = skrub_ops._profile_subsample_n(df, "y", stratify=False)
    assert n == 500 and floor == 10  # max(500, 1%) with no profile pressure


def test_profile_n_never_exceeds_the_table():
    df = pd.DataFrame({"x": range(300), "y": range(300)})
    n, _ = skrub_ops._profile_subsample_n(df, "y", stratify=False)
    assert n == 300


def test_profile_n_grows_for_imbalance_and_respects_the_cap():
    # 1% positive: floor 40 real positives -> n grows toward 40/0.01 = 4000,
    # capped at 2000 for wall-clock
    df = _imbalanced_df(n=10_000, pos_rate=0.01)
    n, floor = skrub_ops._profile_subsample_n(df, "y", stratify=True)
    assert floor == 40
    assert n == 2000


def test_profile_n_floor_is_all_real_minority_rows_when_scarce():
    df = _imbalanced_df(n=10_000, pos_rate=0.0015)  # ~15 positives total
    minority = int(df["y"].sum())
    _, floor = skrub_ops._profile_subsample_n(df, "y", stratify=True)
    assert floor == min(minority, 40)


def test_profile_n_bumps_for_high_cardinality_text():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "text": [f"cat_{i % 200}" for i in range(5_000)],
            "x": rng.normal(size=5_000),
            "y": rng.normal(size=5_000),
        }
    )
    n, _ = skrub_ops._profile_subsample_n(df, "y", stratify=False)
    assert n == 1000  # > the 500 base: encoders need to see the vocabulary


def test_profile_n_ignores_the_target_cardinality():
    # a high-cardinality TARGET must not trigger the text bump
    df = pd.DataFrame(
        {"x": range(5_000), "y": [f"label_{i}" for i in range(5_000)]}
    )
    n, _ = skrub_ops._profile_subsample_n(df, "y", stratify=False)
    assert n == 500


# --- seed-averaged rollouts ---------------------------------------------------


@pytest.fixture(scope="module")
def reg_df():
    rng = np.random.default_rng(1)
    n = 1_200  # > the 500-row subsample, so different seeds draw differently
    x = rng.normal(size=n)
    return pd.DataFrame(
        {"x": x, "target": 2.0 * x + rng.normal(scale=1.0, size=n)}
    )


@pytest.fixture(scope="module")
def reg_plan(reg_df):
    from sklearn.linear_model import Ridge

    return skrub_ops.build_staged_plan({"model": {"Ridge": Ridge()}}, reg_df)


def test_seed_averaged_rollout_is_deterministic_and_distinct(reg_plan, reg_df):
    one = skrub_ops.make_rollout_fn(reg_plan, reg_df, seed=7)
    avg = skrub_ops.make_rollout_fn(
        reg_plan, reg_df, seed=7, n_subsample_seeds=3
    )
    r1, r3 = one({"model": "Ridge"}), avg({"model": "Ridge"})
    assert 0.0 < r1 <= 1.0 and 0.0 < r3 <= 1.0
    assert r3 == avg({"model": "Ridge"})  # deterministic -> cache-exact
    assert r3 != r1  # genuinely averages over different subsample draws


def test_seed_averaging_reduces_reward_variance(reg_plan, reg_df):
    """The point of the feature: across base seeds, the averaged reward
    varies less than the single-draw reward."""

    def rewards(k):
        return [
            skrub_ops.make_rollout_fn(
                reg_plan, reg_df, seed=s, n_subsample_seeds=k
            )({"model": "Ridge"})
            for s in range(30, 38)
        ]

    assert np.var(rewards(3)) < np.var(rewards(1))
