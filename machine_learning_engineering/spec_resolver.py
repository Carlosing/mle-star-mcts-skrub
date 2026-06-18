"""Resolve an LLM-authored plan spec into a ``build_staged_plan`` spec — with
hyperparameter search — via lazy, allowlisted imports.

The ``plan_author`` agent names operators by their FULL DOTTED IMPORT PATH, e.g.
``"sklearn.preprocessing.RobustScaler"`` or
``"sklearn.ensemble.RandomForestRegressor"``. Each operator is either a bare
path string or ``{"name": <path>, "params": {...}}`` to tune hyperparameters.

Imports are **lazy**: a class is imported (via ``importlib``) only when its path
is actually named, so this module loads without pulling sklearn — only ``skrub``
is needed eagerly (for the ``choose_*`` nodes). Safety: only paths under
``ALLOWLIST_ROOTS`` (sklearn, skrub) are importable, so a hallucinated path can
never import an arbitrary module. A path that can't be imported is simply
dropped (no special handling, by design).

Hyperparameters are still curated: ``REGISTRY`` holds vetted tunable bounds per
path. A tuned param becomes a ``skrub.choose_int`` / ``choose_float`` /
``choose_from`` node so it surfaces in ``get_action_space`` and MCTS searches it
(the CASH structure — HPs nested under each model). An operator that is
importable but not in ``REGISTRY`` is usable at its defaults (no HP search).

Flow: parse_spec_json(text) -> resolve_spec(dict) -> dict for build_staged_plan.
"""

import importlib
import json
import re

import skrub  # eager: needed for choose_* nodes (sklearn stays lazy)

SEED = 42
_SKIP_TOKENS = {"skip", "none", "null", ""}

# Only paths under these roots may be imported (blocks `import os`, etc.).
ALLOWLIST_ROOTS = ("sklearn", "skrub")


# --- tunable rule constructors (the allowed HP envelope) ---------------------


def _int(low, high, log=False):
    return {"type": "int", "low": low, "high": high, "log": log}


def _float(low, high, log=False):
    return {"type": "float", "low": low, "high": high, "log": log}


def _choice(options):
    return {"type": "choice", "options": list(options)}


# Shared model tunables (same param names across Regressor/Classifier variants).
_HGB_TUNABLE = {
    "learning_rate": _float(0.01, 0.3, log=True),
    "max_iter": _int(100, 600),
    "max_depth": _int(2, 16),
    "l2_regularization": _float(0.0, 1.0),
}
_RF_TUNABLE = {
    "n_estimators": _int(100, 500),
    "max_depth": _int(3, 30),
    "min_samples_leaf": _int(1, 10),
    "max_features": _choice(["sqrt", "log2", 1.0]),
}

# Curated registry keyed by dotted path: {kind, defaults, tunable}. `kind` only
# groups the prompt vocabulary. Classes are imported lazily, not referenced here.
REGISTRY = {
    # skrub transformers
    "skrub.Cleaner": {"kind": "transformer", "defaults": {}, "tunable": {}},
    "skrub.GapEncoder": {
        "kind": "transformer",
        "defaults": {},
        "tunable": {"n_components": _int(10, 50)},
    },
    "skrub.MinHashEncoder": {
        "kind": "transformer",
        "defaults": {},
        "tunable": {"n_components": _int(20, 80)},
    },
    "skrub.StringEncoder": {"kind": "transformer", "defaults": {}, "tunable": {}},
    # sklearn transformers
    "sklearn.preprocessing.StandardScaler": {
        "kind": "transformer",
        "defaults": {},
        "tunable": {},
    },
    "sklearn.preprocessing.RobustScaler": {
        "kind": "transformer",
        "defaults": {},
        "tunable": {},
    },
    "sklearn.preprocessing.PolynomialFeatures": {
        "kind": "transformer",
        "defaults": {"include_bias": False, "degree": 2},
        "tunable": {"degree": _int(2, 3)},
    },
    "sklearn.decomposition.PCA": {"kind": "transformer", "defaults": {}, "tunable": {}},
    # sklearn models — regression + classification variants
    "sklearn.ensemble.HistGradientBoostingRegressor": {
        "kind": "model",
        "defaults": {},
        "tunable": _HGB_TUNABLE,
    },
    "sklearn.ensemble.HistGradientBoostingClassifier": {
        "kind": "model",
        "defaults": {},
        "tunable": _HGB_TUNABLE,
    },
    "sklearn.ensemble.RandomForestRegressor": {
        "kind": "model",
        "defaults": {},
        "tunable": _RF_TUNABLE,
    },
    "sklearn.ensemble.RandomForestClassifier": {
        "kind": "model",
        "defaults": {},
        "tunable": _RF_TUNABLE,
    },
    "sklearn.linear_model.Ridge": {
        "kind": "model",
        "defaults": {},
        "tunable": {"alpha": _float(1e-3, 1e3, log=True)},
    },
    "sklearn.linear_model.LinearRegression": {
        "kind": "model",
        "defaults": {},
        "tunable": {},
    },
    "sklearn.linear_model.LogisticRegression": {
        "kind": "model",
        "defaults": {"max_iter": 1000},
        "tunable": {"C": _float(1e-3, 1e3, log=True)},
    },
}


# --- lazy, allowlisted class loading -----------------------------------------


def _load_class(path):
    """Import and return the class at a dotted path, or None.

    Returns None for non-strings, paths outside ALLOWLIST_ROOTS, or any import /
    attribute failure (the operator is then dropped — by design).
    """
    if not isinstance(path, str) or "." not in path:
        return None
    if path.split(".", 1)[0] not in ALLOWLIST_ROOTS:
        return None
    module_path, _, cls_name = path.rpartition(".")
    try:
        return getattr(importlib.import_module(module_path), cls_name)
    except (ImportError, AttributeError, ValueError):
        return None


# --- parsing -----------------------------------------------------------------


def parse_spec_json(raw):
    """Parse LLM output into a dict, tolerating ```json fences and prose."""
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


def _entry_name_params(item):
    """An option is a bare path, or {"name": <path>, "params": {...}}."""
    if isinstance(item, dict):
        return item.get("name"), item.get("params") or {}
    return item, {}


def _clip(rng, rule):
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        return rule["low"], rule["high"]
    try:
        low = max(float(rng[0]), float(rule["low"]))
        high = min(float(rng[1]), float(rule["high"]))
    except (TypeError, ValueError):
        return rule["low"], rule["high"]
    return low, high


def _build_choice(choice_name, llm_rule, rule):
    """Turn one allowed param + LLM range into a skrub choose_* node (or None)."""
    if not isinstance(llm_rule, dict):
        return None
    typ = rule["type"]
    if typ in ("int", "float"):
        low, high = _clip(llm_rule.get(typ) or llm_rule.get("range"), rule)
        if low is None or low >= high:
            return None
        log = bool(llm_rule.get("log", rule.get("log", False)))
        if typ == "int":
            return skrub.choose_int(int(low), int(high), log=log, name=choice_name)
        return skrub.choose_float(low, high, log=log, name=choice_name)
    if typ == "choice":
        allowed = rule["options"]
        wanted = llm_rule.get("choice") or llm_rule.get("options") or allowed
        opts = [o for o in wanted if o in allowed] or allowed
        return skrub.choose_from(opts, name=choice_name)
    return None


def _make(path, params, seed, context):
    """Lazily import `path` and instantiate it, wrapping tuned params in
    skrub choose_* nodes. Returns None if the class can't be imported/built."""
    cls = _load_class(path)
    if cls is None:
        return None
    entry = REGISTRY.get(path, {})
    tunable = entry.get("tunable", {})
    kwargs = dict(entry.get("defaults", {}))
    for pname, llm_rule in (params or {}).items():
        rule = tunable.get(pname)
        if rule is None:
            continue  # not a curated tunable for this operator -> ignored
        choice = _build_choice(f"{context}__{cls.__name__}__{pname}", llm_rule, rule)
        if choice is not None:
            kwargs[pname] = choice
    try:
        return _ensure_seeded(cls(**kwargs), seed)
    except Exception:
        return None


def _resolve_options(items, seed, allow_skip, context):
    out = []
    skip_added = False
    for item in items or []:
        path, params = _entry_name_params(item)
        if _is_skip(path):
            if allow_skip and not skip_added:
                out.append(None)
                skip_added = True
            continue
        inst = _make(path, params, seed, context)
        if inst is not None:
            out.append(inst)
    return out


def _iter_model_items(model):
    if model is None:
        return []
    if isinstance(model, (str, dict)):
        return [model]
    return list(model)


def resolve_spec(
    raw, task_type: str = "regression", seed: int = SEED, strict: bool = False
) -> dict:
    """Turn an LLM spec (dotted paths + HP ranges) into a build_staged_plan spec.

    Returns instance-valued clean_options / encoder_options / stages and a
    ``{class_name: instance}`` ``model`` dict, where tuned params are skrub
    choose_* nodes. ``assemble`` is passed through. ``task_type`` is advisory now
    (the LLM names the task-appropriate class). Raises if no model could be
    imported.
    """
    spec = parse_spec_json(raw)
    if task_type not in ("regression", "classification"):
        raise ValueError(
            f"task_type must be regression|classification, got {task_type!r}"
        )
    if strict:
        unknown = unknown_operators(spec)
        if unknown:
            raise ValueError(
                f"unknown operators (not importable / not allowed): {unknown}"
            )

    out: dict = {}
    if spec.get("assemble"):
        out["assemble"] = spec["assemble"]  # config passthrough

    clean = _resolve_options(spec.get("clean_options"), seed, True, "clean")
    if clean:
        out["clean_options"] = clean
    enc = _resolve_options(spec.get("encoder_options"), seed, False, "encoder")
    if enc:
        out["encoder_options"] = enc

    stages = []
    for stage in spec.get("stages", []) or []:
        opts = _resolve_options(
            stage.get("options"), seed, True, stage.get("name", "stage")
        )
        if opts:
            stages.append({"name": stage["name"], "options": opts})
    if stages:
        out["stages"] = stages

    models = {}
    for item in _iter_model_items(spec.get("model")):
        path, params = _entry_name_params(item)
        inst = _make(path, params, seed, "model")
        if inst is not None:
            models[type(inst).__name__] = inst
    if not models:
        raise ValueError("spec has no usable model (none could be imported)")
    out["model"] = models
    return out


# --- introspection (prompt vocabulary + driver-side validation) --------------


def unknown_operators(raw, task_type=None) -> list[str]:
    """Operator paths in the spec that cannot be imported (allowlist/typo)."""
    spec = parse_spec_json(raw)
    unknown: list[str] = []

    def check(items):
        for it in items or []:
            path, _ = _entry_name_params(it)
            if not _is_skip(path) and _load_class(path) is None:
                unknown.append(path)

    check(spec.get("clean_options"))
    check(spec.get("encoder_options"))
    for stage in spec.get("stages", []) or []:
        check(stage.get("options"))
    for it in _iter_model_items(spec.get("model")):
        path, _ = _entry_name_params(it)
        if _load_class(path) is None:
            unknown.append(path)
    return unknown


def allowed_operators() -> dict:
    return {
        "transformers": sorted(
            p for p, e in REGISTRY.items() if e["kind"] == "transformer"
        ),
        "models": sorted(p for p, e in REGISTRY.items() if e["kind"] == "model"),
    }


def _fmt_rule(p, r) -> str:
    if r["type"] in ("int", "float"):
        log = " log" if r.get("log") else ""
        return f"{p} {r['type']}[{r['low']},{r['high']}]{log}"
    return f"{p} choice{list(r['options'])}"


def _fmt_tunable(tunable) -> str:
    return ", ".join(_fmt_rule(p, r) for p, r in tunable.items())


def format_allowed_for_prompt() -> str:
    """List allowed dotted paths + their tunable hyperparameters for the prompt."""
    lines = [
        "Name operators by their FULL DOTTED IMPORT PATH (e.g. "
        '"sklearn.preprocessing.RobustScaler"). An operator is either a bare '
        'path, or {"name": <path>, "params": {...}} to tune hyperparameters. '
        'Param rules: {"int":[lo,hi]}, {"float":[lo,hi],"log":true}, '
        '{"choice":[...]}. For an optional stage use "skip". Choose the '
        "task-appropriate class (Regressor for regression, Classifier for "
        "classification). Only sklearn.* and skrub.* paths are allowed. These "
        "have curated, searchable hyperparameters:",
        "Transformers:",
    ]
    for path in sorted(p for p, e in REGISTRY.items() if e["kind"] == "transformer"):
        tun = _fmt_tunable(REGISTRY[path]["tunable"])
        lines.append(f"  {path}" + (f" (params: {tun})" if tun else ""))
    lines.append("Models:")
    for path in sorted(p for p, e in REGISTRY.items() if e["kind"] == "model"):
        tun = _fmt_tunable(REGISTRY[path]["tunable"])
        lines.append(f"  {path}" + (f" (params: {tun})" if tun else ""))
    return "\n".join(lines)
