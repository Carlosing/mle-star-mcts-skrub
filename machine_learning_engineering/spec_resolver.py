"""Resolve an LLM-authored plan spec into a ``build_staged_plan`` spec — with
hyperparameter search — allowed-list only.

The ``plan_author`` agent emits JSON. Each operator is either a bare name
(defaults) or an object ``{"name": X, "params": {...}}`` that tunes its
hyperparameters. Names AND hyperparameters are restricted to a curated registry:
no ``eval``, no dynamic import, and HP ranges are clipped into vetted bounds. A
tuned param becomes a ``skrub.choose_int`` / ``choose_float`` / ``choose_from``
node, so it surfaces in ``get_action_space`` and MCTS searches it alongside the
operator choice (the CASH structure — HPs nested under each model).

Param rule shapes the LLM may emit (type is authoritative from the registry):
    {"int":   [low, high]}            -> choose_int(low, high)
    {"float": [low, high], "log": true} -> choose_float(low, high, log=True)
    {"choice": ["a", "b"]}            -> choose_from(["a", "b"])
Ranges are clipped to the registry's allowed envelope; unknown params/operators
are dropped (or raise with strict=True).

Flow: parse_spec_json(text) -> resolve_spec(dict, task_type) -> dict ready for
skrub_ops.build_staged_plan. Note (CASH): HPs of a non-selected model still
appear as (inactive) search dimensions — correct but slightly wider; proper
conditional nesting is the deferred staged-expansion work (docs/pipeline-stages).
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


# --- tunable rule constructors (the allowed HP envelope) ---------------------


def _int(low, high, log=False):
    return {"type": "int", "low": low, "high": high, "log": log}


def _float(low, high, log=False):
    return {"type": "float", "low": low, "high": high, "log": log}


def _choice(options):
    return {"type": "choice", "options": list(options)}


def _skrub_attr(name):
    return getattr(skrub, name, None)


# Transformers: name -> {factory, defaults, tunable}. skrub classes guarded.
_TRANSFORMERS = {
    "Cleaner": {"factory": _skrub_attr("Cleaner"), "defaults": {}, "tunable": {}},
    "GapEncoder": {
        "factory": _skrub_attr("GapEncoder"),
        "defaults": {},
        "tunable": {"n_components": _int(10, 50)},
    },
    "MinHashEncoder": {
        "factory": _skrub_attr("MinHashEncoder"),
        "defaults": {},
        "tunable": {"n_components": _int(20, 80)},
    },
    "StringEncoder": {"factory": _skrub_attr("StringEncoder"), "defaults": {}, "tunable": {}},
    "StandardScaler": {"factory": StandardScaler, "defaults": {}, "tunable": {}},
    "RobustScaler": {"factory": RobustScaler, "defaults": {}, "tunable": {}},
    "PolynomialFeatures": {
        "factory": PolynomialFeatures,
        "defaults": {"include_bias": False, "degree": 2},
        "tunable": {"degree": _int(2, 3)},
    },
    "PCA": {"factory": PCA, "defaults": {}, "tunable": {}},
}
TRANSFORMER_REGISTRY = {k: v for k, v in _TRANSFORMERS.items() if v["factory"] is not None}

# Models: name -> {regression, classification, defaults, tunable}. `tunable` is
# flat when params are shared across tasks, or task-keyed when names differ.
MODEL_REGISTRY = {
    "HistGradientBoosting": {
        "regression": HistGradientBoostingRegressor,
        "classification": HistGradientBoostingClassifier,
        "defaults": {},
        "tunable": {
            "learning_rate": _float(0.01, 0.3, log=True),
            "max_iter": _int(100, 600),
            "max_depth": _int(2, 16),
            "l2_regularization": _float(0.0, 1.0),
        },
    },
    "RandomForest": {
        "regression": RandomForestRegressor,
        "classification": RandomForestClassifier,
        "defaults": {},
        "tunable": {
            "n_estimators": _int(100, 500),
            "max_depth": _int(3, 30),
            "min_samples_leaf": _int(1, 10),
            "max_features": _choice(["sqrt", "log2", 1.0]),
        },
    },
    "Linear": {
        "regression": Ridge,
        "classification": lambda **kw: LogisticRegression(max_iter=1000, **kw),
        "defaults": {},
        "tunable": {
            "regression": {"alpha": _float(1e-3, 1e3, log=True)},
            "classification": {"C": _float(1e-3, 1e3, log=True)},
        },
    },
}


# --- parsing -----------------------------------------------------------------


def parse_spec_json(raw):
    """Parse LLM output into a dict, tolerating ```json fences and prose.

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
    """An option is a bare name or {"name": X, "params": {...}}."""
    if isinstance(item, dict):
        return item.get("name"), item.get("params") or {}
    return item, {}


def _clip(rng, rule):
    """Clip an LLM [low, high] into the registry envelope; fall back to it."""
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
            return None  # collapsed/invalid range -> leave at default
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


def _make(factory, defaults, tunable, op_name, params, seed, context):
    """Instantiate an operator, wrapping tuned params in skrub choose_* nodes."""
    kwargs = dict(defaults)
    for pname, llm_rule in (params or {}).items():
        rule = tunable.get(pname)
        if rule is None:
            continue  # param not tunable / not allowed -> dropped
        choice = _build_choice(f"{context}__{op_name}__{pname}", llm_rule, rule)
        if choice is not None:
            kwargs[pname] = choice
    return _ensure_seeded(factory(**kwargs), seed)


def _model_tunable(entry, task_type):
    t = entry.get("tunable", {})
    if t and all(k in ("regression", "classification") for k in t):
        return t.get(task_type, {})
    return t


def _resolve_options(items, seed, allow_skip, context):
    out = []
    skip_added = False
    for item in items or []:
        name, params = _entry_name_params(item)
        if _is_skip(name):
            if allow_skip and not skip_added:
                out.append(None)
                skip_added = True
            continue
        entry = TRANSFORMER_REGISTRY.get(name)
        if entry is not None:
            out.append(
                _make(entry["factory"], entry["defaults"], entry["tunable"], name, params, seed, context)
            )
    return out


def _iter_model_items(model):
    if model is None:
        return []
    if isinstance(model, dict):
        return list(model.keys())
    if isinstance(model, (str, dict)):
        return [model]
    return list(model)


def resolve_spec(raw, task_type: str = "regression", seed: int = SEED, strict: bool = False) -> dict:
    """Turn an LLM spec (names + HP ranges) into a build_staged_plan spec.

    Returns instance-valued clean_options / encoder_options / stages and a
    {label: instance} ``model`` dict, where tuned params are skrub choose_*
    nodes. ``assemble`` is passed through. Raises if no known model survives.
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

    clean = _resolve_options(spec.get("clean_options"), seed, True, "clean")
    if clean:
        out["clean_options"] = clean
    enc = _resolve_options(spec.get("encoder_options"), seed, False, "encoder")
    if enc:
        out["encoder_options"] = enc

    stages = []
    for stage in spec.get("stages", []) or []:
        opts = _resolve_options(stage.get("options"), seed, True, stage.get("name", "stage"))
        if opts:
            stages.append({"name": stage["name"], "options": opts})
    if stages:
        out["stages"] = stages

    models = {}
    for item in _iter_model_items(spec.get("model")):
        name, params = _entry_name_params(item)
        entry = MODEL_REGISTRY.get(name)
        if entry and task_type in entry:
            models[name] = _make(
                entry[task_type], entry.get("defaults", {}),
                _model_tunable(entry, task_type), name, params, seed, "model",
            )
    if not models:
        raise ValueError(f"spec has no known model for task_type={task_type!r}")
    out["model"] = models
    return out


# --- introspection (prompt vocabulary + driver-side validation) --------------


def unknown_operators(raw, task_type: str = "regression") -> list[str]:
    """Operator names in the spec that are NOT in the allowed list."""
    spec = parse_spec_json(raw)
    unknown: list[str] = []

    def check(items):
        for it in items or []:
            name, _ = _entry_name_params(it)
            if not _is_skip(name) and name not in TRANSFORMER_REGISTRY:
                unknown.append(name)

    check(spec.get("clean_options"))
    check(spec.get("encoder_options"))
    for stage in spec.get("stages", []) or []:
        check(stage.get("options"))
    for it in _iter_model_items(spec.get("model")):
        name, _ = _entry_name_params(it)
        entry = MODEL_REGISTRY.get(name)
        if not entry or task_type not in entry:
            unknown.append(name)
    return unknown


def allowed_operators() -> dict:
    return {
        "transformers": sorted(TRANSFORMER_REGISTRY),
        "models": sorted(MODEL_REGISTRY),
    }


def _fmt_rule(p, r) -> str:
    if r["type"] in ("int", "float"):
        log = " log" if r.get("log") else ""
        return f"{p} {r['type']}[{r['low']},{r['high']}]{log}"
    return f"{p} choice{list(r['options'])}"


def _fmt_tunable(tunable) -> str:
    if not tunable:
        return ""
    if all(k in ("regression", "classification") for k in tunable):  # task-keyed
        parts = []
        for task, params in tunable.items():
            parts.append(
                "; ".join(_fmt_rule(p, r) for p, r in params.items()) + f" [{task}]"
            )
        return " | ".join(parts)
    return ", ".join(_fmt_rule(p, r) for p, r in tunable.items())


def format_allowed_for_prompt() -> str:
    """Lists allowed operators + their tunable hyperparameters for the prompt."""
    lines = [
        "Use ONLY these operators (exact names). An operator is either a bare "
        'name, or an object {"name": X, "params": {...}} to tune hyperparameters. '
        'Param rules: {"int":[lo,hi]}, {"float":[lo,hi],"log":true}, '
        '{"choice":[...]}. For an optional stage use "skip".',
        "Transformers:",
    ]
    for name in sorted(TRANSFORMER_REGISTRY):
        tun = _fmt_tunable(TRANSFORMER_REGISTRY[name]["tunable"])
        lines.append(f"  {name}" + (f" (params: {tun})" if tun else ""))
    lines.append("Models:")
    for name in sorted(MODEL_REGISTRY):
        tun = _fmt_tunable(MODEL_REGISTRY[name].get("tunable", {}))
        lines.append(f"  {name}" + (f" (params: {tun})" if tun else ""))
    return "\n".join(lines)
