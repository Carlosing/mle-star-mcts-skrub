"""Offline checks for the revived MLE-STAR runner's hard-cap machinery.

No network: we compose the real MLE-STAR root and exercise the cap/token
callbacks directly. Full end-to-end MLE-STAR needs a live provider + budget
(it writes and executes generated Python), which is out of scope for the
offline suite — here we prove the caps abort cleanly and the graph composes.
"""

import time
import types

import scripts.run_mlestar as R


def test_clamp_config_sets_conservative_knobs():
    cfg = R._clamp_config(
        "california-housing-prices", "./machine_learning_engineering/tasks/"
    )
    assert cfg.num_solutions == 1
    assert cfg.max_debug_round == 1
    assert cfg.outer_loop_round == 1
    assert cfg.task_name == "california-housing-prices"


def _find_llm(agent):
    if hasattr(agent, "model") and getattr(agent, "before_model_callback", None):
        return agent
    for child in getattr(agent, "sub_agents", None) or []:
        found = _find_llm(child)
        if found is not None:
            return found
    return None


def test_root_composes_and_caps_abort_cleanly():
    # ADK forbids re-parenting the sub-agent singletons, so the root is built
    # ONCE and both cap paths (max_calls, deadline) are exercised via the
    # mutable limits on the shared `counter` dict.
    R._clamp_config(
        "california-housing-prices", "./machine_learning_engineering/tasks/"
    )
    sink, counter = {}, {"calls": 0}
    root = R.build_capped_root(
        max_calls=2,
        deadline=time.perf_counter() + 9999,
        max_output_tokens=4096,
        token_sink=sink,
        counter=counter,
    )
    assert [a.name for a in root.sub_agents][:3] == [
        "initialization_agent",
        "refinement_agent",
        "ensemble_agent",
    ]
    leaf = _find_llm(root)
    assert leaf is not None
    cap = leaf.before_model_callback[0]

    ctx = types.SimpleNamespace(agent_name="probe")
    req = types.SimpleNamespace(
        config=types.SimpleNamespace(max_output_tokens=None)
    )
    # max_calls path
    cap(ctx, req)
    assert req.config.max_output_tokens == 4096  # per-call output bound applied
    cap(ctx, req)
    assert counter["calls"] == 2
    try:
        cap(ctx, req)
        assert False, "cap should have raised past max_calls"
    except R._CallBudgetExceeded:
        pass

    # deadline path: move the deadline into the past, reset the counter
    counter["calls"] = 0
    counter["deadline"] = time.perf_counter() - 1.0
    try:
        cap(ctx, req)
        assert False, "cap should have raised on the deadline"
    except R._CallBudgetExceeded:
        pass
    assert counter["calls"] == 0  # deadline check precedes the counter
