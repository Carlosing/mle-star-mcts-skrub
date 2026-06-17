"""Resolve an LLM-authored plan spec (operator NAME strings) into a
``build_staged_plan`` spec (seeded estimator instances) — allowed-list only.

The ``plan_author`` agent emits JSON naming operators per stage. This module
turns those names into real, seeded estimator instances using a CURATED
registry: no ``eval``, no dynamic import. A name outside the registry is dropped
(or raises, with ``strict=True``), so a hallucinated operator can never run.

Flow: ``parse_spec_json(text)`` -> ``resolve_spec(dict, task_type)`` -> a dict
ready for ``skrub_ops.build_staged_plan``.

Two format reconciliations happen here (the LLM shape != build_staged_plan's):
- option name lists -> instance lists, with "skip"/"none" -> None;
- the ``model`` name list -> a {label: instance} dict (what build_staged_plan
  wants), with the right Regressor/Classifier variant per ``task_type``.
"""

import json
import re

import skrub
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler

SEED = 42
_SKIP_TOKENS = {"skip", "none", "null", ""}


def _skrub_attr(name):
    """skrub class if present in the installed version, else None."""
    return getattr(skrub, name, None)


# Task-agnostic transformers: friendly name -> factory (callable -> instance).
# skrub classes are guarded so a version without one simply omits it.
_RAW_TRANSFORMERS = {
    "Cleaner": _skrub_attr("Cleaner"),
    "GapEncoder": _skrub_attr("GapEncoder"),
    "MinHashEncoder": _skrub_attr("MinHashEncoder"),
    "StringEncoder": _skrub_attr("StringEncoder"),
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "PolynomialFeatures": lambda: PolynomialFeatures(degree=2, include_bias=False),
    "PCA": PCA,
}
TRANSFORMER_REGISTRY = {k: v for k, v in _RAW_TRANSFORMERS.items() if v is not None}

# Task-aware models: friendly name -> {task_type: factory}.
MODEL_REGISTRY = {
    "HistGradientBoosting": {
        "regression": HistGradientBoostingRegressor,
        "classification": HistGradientBoostingClassifier,
    },
    "RandomForest": {
        "regression": RandomForestRegressor,
        "classification": RandomForestClassifier,
    },
    "Linear": {
        "regression": Ridge,
        "classification": lambda: LogisticRegression(max_iter=1000),
    },
}


# --- parsing -----------------------------------------------------------------


def parse_spec_json(raw):
    """Parse the LLM output into a dict, tolerating ```json fences and prose.

    Raises json.JSONDecodeError if no valid JSON object can be recovered (e.g. a
    MAX_TOKENS-truncated response) — the caller decides how to handle that.
    """
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if 0 <= start < end:
            return json.loads(s[start : end + 1])
        raise


# --- resolution --------------------------------------------------------------


def _ensure_seeded(est, seed=SEED):
    """Inject random_state=seed iff the estimator has it and it is unset."""
    try:
        params = est.get_params()
    except Exception:
        return est
    if params.get("random_state", "absent") is None:
        est.set_params(random_state=seed)
    return est


def _is_skip(name) -> bool:
    return name is None or (
        isinstance(name, str) and name.strip().lower() in _SKIP_TOKENS
    )


def _resolve_options(names, seed, allow_skip):
    """Map a list of option names to instances; "skip" -> a single None."""
    out = []
    skip_added = False
    for name in names or []:
        if _is_skip(name):
            if allow_skip and not skip_added:
                out.append(None)
                skip_added = True
            continue
        factory = TRANSFORMER_REGISTRY.get(name)
        if factory is not None:
            out.append(_ensure_seeded(factory(), seed))
        # unknown name -> silently dropped (see unknown_operators / strict=True)
    return out


def _iter_model_names(model):
    if model is None:
        return []
    if isinstance(model, dict):
        return list(model.keys())
    if isinstance(model, str):
        return [model]
    return list(model)


def resolve_spec(raw, task_type: str = "regression", seed: int = SEED, strict: bool = False) -> dict:
    """Turn an LLM spec (names) into a build_staged_plan spec (instances).

    Args:
      raw: spec as a dict or JSON string (fences/prose tolerated).
      task_type: "regression" or "classification" (selects model variant).
      strict: raise on any name outside the allowed list instead of dropping it.

    Returns a dict with instance-valued clean_options / encoder_options / stages
    and a {label: instance} ``model`` dict. ``assemble`` (AggJoiner config) is
    passed through unchanged. Raises ValueError if no known model survives.
    """
    spec = parse_spec_json(raw)
    if task_type not in ("regression", "classification"):
        raise ValueError(f"task_type must be regression|classification, got {task_type!r}")
    if strict:
        unknown = unknown_operators(spec, task_type)
        if unknown:
            raise ValueError(f"unknown operators (not in allowed list): {unknown}")

    out: dict = {}
    if spec.get("assemble"):
        out["assemble"] = spec["assemble"]  # config passthrough, not registry ops

    clean = _resolve_options(spec.get("clean_options"), seed, allow_skip=True)
    if clean:
        out["clean_options"] = clean

    enc = _resolve_options(spec.get("encoder_options"), seed, allow_skip=False)
    if enc:
        out["encoder_options"] = enc

    stages = []
    for stage in spec.get("stages", []) or []:
        opts = _resolve_options(stage.get("options"), seed, allow_skip=True)
        if opts:
            stages.append({"name": stage["name"], "options": opts})
    if stages:
        out["stages"] = stages

    models = {}
    for name in _iter_model_names(spec.get("model")):
        entry = MODEL_REGISTRY.get(name)
        if entry and task_type in entry:
            models[name] = _ensure_seeded(entry[task_type](), seed)
    if not models:
        raise ValueError(f"spec has no known model for task_type={task_type!r}")
    out["model"] = models
    return out


# --- introspection (for the prompt vocabulary + driver-side validation) ------


def unknown_operators(raw, task_type: str = "regression") -> list[str]:
    """Names in the spec that are NOT in the allowed list (would be dropped)."""
    spec = parse_spec_json(raw)
    unknown: list[str] = []

    def check(names):
        for n in names or []:
            if not _is_skip(n) and n not in TRANSFORMER_REGISTRY:
                unknown.append(n)

    check(spec.get("clean_options"))
    check(spec.get("encoder_options"))
    for stage in spec.get("stages", []) or []:
        check(stage.get("options"))
    for n in _iter_model_names(spec.get("model")):
        entry = MODEL_REGISTRY.get(n)
        if not entry or task_type not in entry:
            unknown.append(n)
    return unknown


def allowed_operators() -> dict:
    return {
        "transformers": sorted(TRANSFORMER_REGISTRY),
        "models": sorted(MODEL_REGISTRY),
    }


def format_allowed_for_prompt() -> str:
    """A line block listing the allowed names, for the plan_author instruction."""
    a = allowed_operators()
    return (
        "Use ONLY these operator names (exact spelling). For an optional stage, "
        'use "skip" to omit it.\n'
        f"  transformers: {', '.join(a['transformers'])}\n"
        f"  models: {', '.join(a['models'])}"
    )
