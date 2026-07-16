"""Per-rollout wall-clock cap: a slow config scores 0.0 within the budget.

The safety envelope for free-form hyperparameters is at the import level, so a
reckless range can make a single fit pathologically slow. `make_rollout_fn`'s
`timeout_s` guarantees a rollout returns in ~fixed time or 0.0, via a SIGALRM
`_time_limit` that raises a BaseException (so skrub/sklearn's per-fold
`error_score` handler can't swallow it and keep running).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("skrub")

from machine_learning_engineering import skrub_ops
from machine_learning_engineering.skrub_ops import (
    build_staged_plan,
    get_default_state,
    make_rollout_fn,
)


def _plan_and_df():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "target": (rng.normal(size=n) > 0).astype(int),
        }
    )
    spec = {
        "model": {
            "RF": RandomForestClassifier(n_estimators=300, random_state=42)
        }
    }
    return build_staged_plan(spec, df, target="target"), df


# --- the primitive ------------------------------------------------------------


def test_time_limit_raises_on_overrun_and_passes_when_generous():
    t0 = time.monotonic()
    with pytest.raises(skrub_ops._RolloutTimeout):
        with skrub_ops._time_limit(0.05):
            time.sleep(2.0)  # SIGALRM interrupts the sleep
    assert time.monotonic() - t0 < 1.0  # returned promptly, not after 2s

    ran = False
    with skrub_ops._time_limit(
        5.0
    ):  # generous: block completes, timer disarmed
        ran = True
    assert ran


def test_time_limit_noop_when_disabled():
    # None/0 disables the cap -> the block runs uncapped (no raise)
    with skrub_ops._time_limit(None):
        pass
    with skrub_ops._time_limit(0):
        pass


# --- integration: a real skrub rollout -----------------------------------------


def test_rollout_times_out_to_zero_but_completes_when_generous():
    plan, df = _plan_and_df()
    state = get_default_state(plan)

    # A microscopic cap: the CV can't finish in time -> the whole rollout is 0.0
    t0 = time.monotonic()
    timed_out = make_rollout_fn(plan, df, timeout_s=0.01)(state)
    assert timed_out == 0.0
    assert time.monotonic() - t0 < 5.0  # aborted, didn't run the full CV

    # The SAME config scores a real (non-zero) reward with a generous cap, so the
    # 0.0 above must have come from the timeout, not a broken config.
    ok = make_rollout_fn(plan, df, timeout_s=45.0, scoring="accuracy")(state)
    assert ok > 0.0
