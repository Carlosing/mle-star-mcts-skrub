"""Sweep-harness tests — run_pipeline mocked, everything else real, offline."""

import json
import os

import pytest

from machine_learning_engineering import sweep


def _write_spec(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return str(path)


def test_load_sweep_spec_merges_defaults_and_expands_lists(tmp_path):
    path = _write_spec(tmp_path, {
        "defaults": {"task": "credit-fraud", "budget": 10},
        "runs": [{"c": [0.3, 0.5], "seed": [1, 2]}, {"top_k": 3}],
    })
    configs = sweep.load_sweep_spec(path)
    assert len(configs) == 5  # 2x2 expansion + 1 singleton
    assert all(cfg["task"] == "credit-fraud" and cfg["budget"] == 10
               for cfg in configs)
    assert {(cfg["c"], cfg["seed"]) for cfg in configs[:4]} == {
        (0.3, 1), (0.3, 2), (0.5, 1), (0.5, 2)
    }
    assert configs[4]["top_k"] == 3 and configs[4]["c"] == 0.5  # module default


def test_load_sweep_spec_applies_n_proposes_sugar(tmp_path):
    path = _write_spec(tmp_path, {
        "defaults": {"task": "t"},
        "runs": [{"n_proposes": 3}, {"n_proposes": 0}],
    })
    cfg, zero = sweep.load_sweep_spec(path)
    assert cfg["outer_steps"] == 4 and cfg["refine"] is True
    assert "n_proposes" not in cfg  # normalized away
    assert zero["outer_steps"] == 1 and zero["refine"] is False


def test_load_sweep_spec_rejects_unknown_keys(tmp_path):
    path = _write_spec(tmp_path, {
        "defaults": {"task": "t"}, "runs": [{"budgett": 5}]
    })
    with pytest.raises(ValueError, match="budgett"):
        sweep.load_sweep_spec(path)


def test_slugs_are_unique_and_filesystem_safe(tmp_path):
    path = _write_spec(tmp_path, {
        "defaults": {"task": "t"},
        "runs": [{"c": [0.3, 0.5], "seed": [1, 2], "n_proposes": [0, 1]}],
    })
    configs = sweep.load_sweep_spec(path)
    slugs = [sweep.slug(cfg) for cfg in configs]
    assert len(set(slugs)) == len(configs) == 8
    assert all(s.replace(".", "").replace("_", "").isalnum() for s in slugs)


def _fake_result(cfg, spec_raw):
    return {
        "best_search_score": 0.7 + cfg["c"] / 100,
        "report": {"scorer": "roc_auc", "score": 0.71},
        "ensemble": None,
        "used_fallback_spec": False,
        "reused_spec": spec_raw is not None,
        "llm_calls": 0 if spec_raw is not None else 2,
        "spec_raw": spec_raw or "FETCHED_SPEC",
    }


def test_sweep_fetches_spec_once_per_task_and_writes_artifacts(tmp_path, monkeypatch):
    fetches = []

    def fake_run_pipeline(task_name, spec_raw=None, out_dir=None, c=0.5, **kwargs):
        if spec_raw is None:
            fetches.append(task_name)
        os.makedirs(out_dir, exist_ok=True)
        return _fake_result({"c": c}, spec_raw)

    monkeypatch.setattr(sweep.pipeline, "run_pipeline", fake_run_pipeline)
    spec_path = _write_spec(tmp_path, {
        "defaults": {"budget": 5},
        "runs": [{"task": "a", "c": [0.3, 0.5]}, {"task": "b"}],
    })
    out = str(tmp_path / "out")
    monkeypatch.setattr(
        "sys.argv", ["sweep", spec_path, "--out", out, "--retry-wait", "0"]
    )
    sweep._main()

    assert fetches == ["a", "b"]  # one agent fetch per task, not per run
    with open(os.path.join(out, "sweep.csv"), encoding="utf-8") as f:
        rows = list(__import__("csv").DictReader(f))
    assert len(rows) == 3
    assert [r["status"] for r in rows] == ["ok"] * 3
    assert [r["llm_calls"] for r in rows] == ["2", "0", "2"]  # reuse after fetch
    assert os.path.exists(os.path.join(out, "sweep.md"))
    assert all(os.path.isdir(r["run_dir"]) for r in rows)


def test_sweep_retries_quota_once_then_continues(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky_run_pipeline(task_name, spec_raw=None, out_dir=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: rate limited")
        if task_name == "bad":
            raise RuntimeError("boom, not a quota problem")
        os.makedirs(out_dir, exist_ok=True)
        return _fake_result({"c": 0.5}, spec_raw)

    monkeypatch.setattr(sweep.pipeline, "run_pipeline", flaky_run_pipeline)
    monkeypatch.setattr(sweep.time, "sleep", lambda s: None)
    spec_path = _write_spec(tmp_path, {
        "runs": [{"task": "good"}, {"task": "bad"}],
    })
    out = str(tmp_path / "out")
    monkeypatch.setattr("sys.argv", ["sweep", spec_path, "--out", out])
    sweep._main()

    with open(os.path.join(out, "sweep.csv"), encoding="utf-8") as f:
        rows = list(__import__("csv").DictReader(f))
    good, bad = rows
    assert good["status"] == "ok"  # quota error -> one retry -> succeeded
    assert bad["status"] == "failed" and "boom" in bad["error"]
    assert os.path.exists(os.path.join(out, "sweep.md"))  # sweep still finished
