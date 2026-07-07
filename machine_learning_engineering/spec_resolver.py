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
    "skrub.DatetimeEncoder": {
        "kind": "transformer",
        "defaults": {},
        "tunable": {"resolution": _choice(["month", "day", "hour"])},
    },
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


def _resolve_options(items, seed, allow_skip, context, priors_out=None, label_fn=repr):
    """Resolve option items to instances; optionally collect per-option priors.

    An option dict may carry ``"prior": 0.0-1.0`` (the LLM's confidence this
    option wins on this dataset). When ``priors_out`` is given, priors are
    stored under the label the option will get in the action space
    (``label_fn(instance)`` — repr for list-based stages, class name for
    dict-labeled ones).
    """
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
            _collect_prior(priors_out, item, label_fn(inst))
    return out


def _collect_prior(priors_out, item, label) -> None:
    """Store an option's ``prior`` under its action-space label (clipped 0-1)."""
    if priors_out is None or not isinstance(item, dict) or "prior" not in item:
        return
    try:
        priors_out[label] = min(1.0, max(0.0, float(item["prior"])))
    except (TypeError, ValueError):
        pass


def _iter_model_items(model):
    if model is None:
        return []
    if isinstance(model, (str, dict)):
        return [model]
    return list(model)


def _sanitize_name(name) -> str:
    """Slug a group/stage label so it is a safe choice-name suffix.

    Double underscores are collapsed — ``__`` marks hyperparameter dimensions
    in the action space, so a group name must never introduce one.

    Example:
        _sanitize_name("job title enc")  # -> "job_title_enc"
    """
    slug = re.sub(r"\W+", "_", str(name)).strip("_")
    return re.sub(r"_{2,}", "_", slug) or "group"


def _resolve_scoped(entries, seed, main_columns, priors: dict | None = None) -> list[dict]:
    """Validate scoped-encoding groups (the searchable *scope* stage).

    Keeps a group only if at least one of its columns exists in the main table
    and at least one option resolves through the allowlist. Column names the
    LLM invented are dropped (build_staged_plan re-checks against the actual
    dataframe, and the runtime selector is missing-tolerant on top). Two
    optional flags pass through: ``"position": "post_encode"`` (apply after
    the vectorizer; anything else falls back to the pre_encode default) and
    ``"additive": true`` (keep the original columns, concatenate the output).

    Example:
        _resolve_scoped(
            [{"name": "title enc", "cols": ["title", "ghost"], "additive": True,
              "options": ["skip", "skrub.GapEncoder"]}], 42, ["title", "x"])
        # -> [{"name": "title_enc", "cols": ["title"], "options": [<GapEncoder>],
        #      "additive": True}]
    """
    out, seen = [], set()
    for cfg in entries or []:
        if not isinstance(cfg, dict):
            continue
        cols = [
            c for c in (cfg.get("cols") or []) if main_columns is None or c in main_columns
        ]
        if not cols:
            continue
        name = _sanitize_name(cfg.get("name") or cols[0])
        if name in seen:
            continue
        group_priors: dict = {}
        options = [
            o
            for o in _resolve_options(
                cfg.get("options"),
                seed,
                False,
                f"scope_{name}",
                priors_out=group_priors if priors is not None else None,
                label_fn=lambda inst: type(inst).__name__,
            )
            if o is not None
        ]
        if not options:
            continue
        seen.add(name)
        group = {"name": name, "cols": cols, "options": options}
        if cfg.get("position") == "post_encode":
            group["position"] = "post_encode"
        if cfg.get("additive") is True:
            group["additive"] = True
        out.append(group)
        if priors is not None and group_priors:
            priors[f"scope_{name}"] = group_priors
    return out


# Aggregations skrub's AggJoiner supports (skrub._agg_joiner.SUPPORTED_OPS);
# "sum"/"median"/"mean"/"std" are numeric-only, enforced by skrub at fit.
_AGG_OPERATIONS = {"count", "mode", "min", "max", "sum", "median", "mean", "std"}


def _resolve_assemble(
    entries, aux_schemas, main_columns, priors: dict | None = None
) -> list[dict]:
    """Validate LLM assemble (AggJoiner) configs against the real table schemas.

    Same philosophy as HP clipping: hallucinated names are dropped, never
    executed. An entry survives only if its table is a known auxiliary table,
    its join keys exist in the respective tables, and at least one operation is
    supported; ``cols`` is intersected with the aux columns (dropped entirely
    if nothing survives, meaning "aggregate all"). Without ``aux_schemas``
    (single-table run) every entry is dropped.

    Example:
        _resolve_assemble(
            [{"table": "products", "key": "ID", "operations": ["mean", "bogus"]}],
            aux_schemas={"products": ["ID", "price"]}, main_columns=["ID", "x"])
        # -> [{"table": "products", "key": "ID", "operations": ["mean"], ...}]
    """
    out = []
    for cfg in entries or []:
        if not isinstance(cfg, dict):
            continue
        table = cfg.get("table")
        if not aux_schemas or table not in aux_schemas:
            continue
        aux_cols = set(aux_schemas[table])
        operations = [
            op for op in (cfg.get("operations") or []) if op in _AGG_OPERATIONS
        ]
        if not operations:
            continue
        key, main_key, aux_key = cfg.get("key"), cfg.get("main_key"), cfg.get("aux_key")
        if key is not None:
            if key not in aux_cols or (main_columns and key not in main_columns):
                continue
            keys = {"key": key}
        elif main_key is not None and aux_key is not None:
            if aux_key not in aux_cols or (main_columns and main_key not in main_columns):
                continue
            keys = {"main_key": main_key, "aux_key": aux_key}
        else:
            continue
        cleaned = {
            "name": cfg.get("name") or f"{table}_{'_'.join(operations)}",
            "table": table,
            "operations": operations,
            **keys,
        }
        cols = [c for c in (cfg.get("cols") or []) if c in aux_cols]
        if cols:
            cleaned["cols"] = cols
        out.append(cleaned)
        if priors is not None:
            assemble_priors = priors.setdefault("assemble", {})
            _collect_prior(assemble_priors, cfg, cleaned["name"])
            if not assemble_priors:
                priors.pop("assemble")
    return out


def resolve_spec(
    raw,
    task_type: str = "regression",
    seed: int = SEED,
    strict: bool = False,
    aux_schemas: dict[str, list[str]] | None = None,
    main_columns: list[str] | None = None,
) -> dict:
    """Turn an LLM spec (dotted paths + HP ranges) into a build_staged_plan spec.

    Returns instance-valued clean_options / encoder_options / stages and a
    ``{class_name: instance}`` ``model`` dict, where tuned params are skrub
    choose_* nodes. ``assemble`` entries are validated against ``aux_schemas``
    (``{table_name: [columns]}``) and ``main_columns`` — invalid tables / keys /
    operations / cols are dropped, and without ``aux_schemas`` the stage is
    dropped entirely (see ``_resolve_assemble``). ``scoped_encodings`` groups
    are validated the same way (see ``_resolve_scoped``). Any option carrying
    ``"prior": 0.0-1.0`` contributes to ``out["priors"]`` =
    ``{choice_name: {label: weight}}`` — consumed by the search loop's
    ``prior_fn``, ignored by ``build_staged_plan``. ``task_type`` is advisory
    now (the LLM names the task-appropriate class). Raises if no model could
    be imported.
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
    priors: dict[str, dict[str, float]] = {}
    assemble = _resolve_assemble(
        spec.get("assemble"), aux_schemas, main_columns, priors=priors
    )
    if assemble:
        out["assemble"] = assemble

    def _options_with_priors(items, allow_skip, context, label_fn=repr):
        stage_priors: dict = {}
        opts = _resolve_options(
            items, seed, allow_skip, context, priors_out=stage_priors, label_fn=label_fn
        )
        if stage_priors:
            priors[context] = stage_priors
        return opts

    clean = _options_with_priors(spec.get("clean_options"), True, "clean")
    if clean:
        out["clean_options"] = clean
    enc = _options_with_priors(spec.get("encoder_options"), False, "encoder")
    if enc:
        out["encoder_options"] = enc

    scoped = _resolve_scoped(
        spec.get("scoped_encodings"), seed, main_columns, priors=priors
    )
    if scoped:
        out["scoped_encodings"] = scoped

    stages = []
    for stage in spec.get("stages", []) or []:
        opts = _options_with_priors(
            stage.get("options"), True, stage.get("name", "stage")
        )
        if opts:
            stages.append({"name": stage["name"], "options": opts})
    if stages:
        out["stages"] = stages

    models = {}
    model_priors: dict = {}
    for item in _iter_model_items(spec.get("model")):
        path, params = _entry_name_params(item)
        inst = _make(path, params, seed, "model")
        if inst is not None:
            models[type(inst).__name__] = inst
            _collect_prior(model_priors, item, type(inst).__name__)
    if not models:
        raise ValueError("spec has no usable model (none could be imported)")
    out["model"] = models
    if model_priors:
        priors["model"] = model_priors
    if priors:
        out["priors"] = priors
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
