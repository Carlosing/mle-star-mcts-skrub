"""Resolve an LLM-authored plan spec into a ``build_staged_plan`` spec — with
hyperparameter search — via lazy, allowlisted imports.

The ``plan_author`` agent names operators by their FULL DOTTED IMPORT PATH, e.g.
``"sklearn.preprocessing.RobustScaler"`` or
``"sklearn.ensemble.RandomForestRegressor"``. Each operator is either a bare
path string or ``{"name": <path>, "params": {...}}`` to tune hyperparameters.

Imports are **lazy**: a class is imported (via ``importlib``) only when its path
is actually named, so this module loads without pulling sklearn — only ``skrub``
is needed eagerly (for the ``choose_*`` nodes). Safety: only paths under
``ALLOWLIST_ROOTS`` (sklearn, skrub, lightgbm, xgboost) are importable, so a
hallucinated path can never import an arbitrary module. A path that can't be imported is simply
dropped (no special handling, by design).

The safety envelope is at the IMPORT level, not the search level: any param the
LLM tunes is accepted with the range it gave (``_build_free_choice``), as long
as the operator's constructor actually accepts that param (``_accepts_param`` —
an unknown param is dropped individually, never dropping the operator) and it is
not an RNG-identity param (``_RNG_PARAMS``, seeded centrally for determinism).
``REGISTRY`` is now a set of *curated known-good* bounds: a param listed there is
still clipped to its vetted range (``_build_choice``), while any other param is
free-form. Either way a tuned param becomes a ``skrub.choose_int`` /
``choose_float`` / ``choose_from`` node so it surfaces in ``get_action_space``
and MCTS searches it (the CASH structure — HPs nested under each model).

Flow: parse_spec_json(text) -> resolve_spec(dict) -> dict for build_staged_plan.
"""

import importlib
import importlib.util
import inspect
import json
import math
import re

import skrub  # needed for choose_* nodes

SEED = 42
_SKIP_TOKENS = {"skip", "none", "null", ""}

# Only paths under these roots may be imported (blocks `import os`, etc.).
# lightgbm/xgboost are here for their sklearn-compatible boosters (the SOTA the
# data_analyst surfaces via web search); both seed via `random_state`, so
# `_ensure_seeded` keeps them deterministic.
ALLOWLIST_ROOTS = ("sklearn", "skrub", "lightgbm", "xgboost")


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
# Gradient-boosting libs (sklearn-compatible). Same param names across the
# Regressor/Classifier variants; kept modest so a single fit stays under the
# per-rollout wall-clock cap.
_LGBM_TUNABLE = {
    "n_estimators": _int(100, 1000),
    "learning_rate": _float(0.01, 0.3, log=True),
    "num_leaves": _int(15, 255),
    "max_depth": _int(3, 16),
    "colsample_bytree": _float(0.5, 1.0),
    "reg_lambda": _float(0.0, 5.0),
}
_XGB_TUNABLE = {
    "n_estimators": _int(100, 1000),
    "learning_rate": _float(0.01, 0.3, log=True),
    "max_depth": _int(2, 12),
    "subsample": _float(0.5, 1.0),
    "colsample_bytree": _float(0.5, 1.0),
    "reg_lambda": _float(0.0, 5.0),
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
    "skrub.StringEncoder": {
        "kind": "transformer",
        "defaults": {},
        "tunable": {},
    },
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
    # handle_unknown defaults to "error": a CV fold holding a category the train
    # fold never saw would raise. sparse_output is forced False (_SPARSE_PARAMS).
    "sklearn.preprocessing.OneHotEncoder": {
        "kind": "transformer",
        "defaults": {"handle_unknown": "infrequent_if_exist"},
        "tunable": {"min_frequency": _int(1, 10)},
    },
    "sklearn.decomposition.PCA": {
        "kind": "transformer",
        "defaults": {},
        "tunable": {},
    },
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
    # Gradient-boosting libraries. `n_jobs=1` is LOAD-BEARING, not a perf knob:
    # skrub's pipeline loads sklearn's bundled libomp, and a multi-threaded
    # lightgbm/xgboost fit in the same process loads a SECOND OpenMP runtime;
    # inside skrub's multi-fold CV that duplicate-libomp collision *segfaults*
    # deterministically on macOS-ARM (crashes the whole run, uncatchable by the
    # rollout try/except or the SIGALRM cap). Single-threaded native execution
    # sidesteps it and is harmless elsewhere — rollouts fit tiny subsamples. The
    # other defaults just quiet each library's per-fit logging.
    "lightgbm.LGBMRegressor": {
        "kind": "model",
        "defaults": {"verbose": -1, "n_jobs": 1},
        "tunable": _LGBM_TUNABLE,
    },
    "lightgbm.LGBMClassifier": {
        "kind": "model",
        "defaults": {"verbose": -1, "n_jobs": 1},
        "tunable": _LGBM_TUNABLE,
    },
    "xgboost.XGBRegressor": {
        "kind": "model",
        "defaults": {"verbosity": 0, "n_jobs": 1},
        "tunable": _XGB_TUNABLE,
    },
    "xgboost.XGBClassifier": {
        "kind": "model",
        "defaults": {"verbosity": 0, "n_jobs": 1},
        "tunable": _XGB_TUNABLE,
    },
}


# --- lazy, allowlisted class loading -----------------------------------------


# Operators that import fine but need an OPTIONAL runtime dependency at fit
# time. The allow-list roots (skrub, sklearn, …) are always installed, so a
# path like ``skrub.TextEncoder`` passes ``_load_class`` — but TextEncoder only
# works if ``sentence_transformers`` is present. Without this guard a plan that
# names such an operator crashes the whole run at plan-BUILD (outside the
# rollout's 0.0-on-failure net); with it the operator is dropped like any other
# unusable one, and the run continues on the rest of the menu.
_OPTIONAL_DEPS = {
    "skrub.TextEncoder": "sentence_transformers",
}


def _optional_dep_available(path) -> bool:
    """True unless ``path`` needs an optional dep that isn't importable."""
    module = _OPTIONAL_DEPS.get(path)
    if module is None:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _load_class(path):
    """Import and return the class at a dotted path, or None.

    Returns None for non-strings, paths outside ALLOWLIST_ROOTS, an operator
    whose optional runtime dependency is missing (``_OPTIONAL_DEPS``), or any
    import / attribute failure (the operator is then dropped — by design).
    """
    if not isinstance(path, str) or "." not in path:
        return None
    if path.split(".", 1)[0] not in ALLOWLIST_ROOTS:
        return None
    if not _optional_dep_available(path):
        return None
    module_path, _, cls_name = path.rpartition(".")
    try:
        return getattr(importlib.import_module(module_path), cls_name)
    except (ImportError, AttributeError, ValueError):
        return None


# --- parsing -----------------------------------------------------------------


# Top-level keys that make an object a *plan* rather than some inner fragment
# of one ({"float": [0.7, 1.0]} and {"name": ..., "params": ...} both parse).
_PLAN_KEYS = frozenset(
    {
        "model",
        "cleaner",
        "vectorizer",
        "stages",
        "scoped_encodings",
        "assemble",
    }
)


def _is_plan_shaped(obj) -> bool:
    """True if `obj` is a dict carrying at least one top-level plan key.

    Example:
        _is_plan_shaped({"model": ["sklearn.svm.SVC"]})   # -> True
        _is_plan_shaped({"float": [0.7, 1.0]})            # -> False (fragment)
    """
    return isinstance(obj, dict) and bool(_PLAN_KEYS & obj.keys())


def parse_spec_json(raw):
    """Parse LLM output into a plan dict, tolerating ```json fences and prose.

    Every candidate must be *plan-shaped*. A truncated response (the model hit
    its output-token cap mid-object) leaves valid JSON fragments scattered in
    the text, and a brace-scan will happily return one — `{"float": [0.7, 1.0]}`
    silently became "the extended plan" and Option 3 no-op'd with no error.
    A fragment now raises instead, so the caller can retry or fail loudly.

    Example:
        >>> parse_spec_json('Ranges like {"int": [1, 9]} are typical.\\n'
        ...                 '```json\\n{"model": ["sklearn.svm.SVC"]}\\n```')
        {'model': ['sklearn.svm.SVC']}
    """
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()

    candidates = []
    # A fenced block is the model's own delimiter for "this is the answer", but
    # a model may fence the plan AND fence an illustrative snippet after it, so
    # collect every block rather than trusting the first or the last.
    candidates += re.findall(r"```(?:[a-zA-Z0-9]*)\s*\n(.*?)```", s, re.S)

    stripped = s
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    candidates.append(stripped)

    # Unfenced: try each '{' as a start rather than only the first, so a brace
    # in the prose cannot poison the parse. Widest span first.
    end = s.rfind("}")
    candidates += [
        s[start : end + 1]
        for start, ch in enumerate(s)
        if ch == "{" and 0 <= start < end
    ]

    fragment_seen = False
    for cand in candidates:
        try:
            parsed = json.loads(cand.strip())
        except json.JSONDecodeError:
            continue
        if _is_plan_shaped(parsed):
            return parsed
        fragment_seen = True

    hint = (
        " (parsed a JSON fragment but no top-level plan key — the response was "
        "probably truncated at its output-token cap)"
        if fragment_seen
        else ""
    )
    raise json.JSONDecodeError(f"no plan-shaped JSON in LLM output{hint}", s, 0)


# --- resolution --------------------------------------------------------------

_XGB_CLASSIFIER_SHIM = None


def _xgb_classifier_shim():
    """An XGBClassifier that accepts arbitrary (e.g. string) target labels.

    xgboost >= 1.6 dropped its internal label encoding: ``fit`` requires ``y``
    to be integers ``0..n_classes-1`` and raises on string labels — so on a
    string-target task every XGB rollout silently scored 0.0 (and a plan whose
    *default* model was XGB failed to build at all under skrub's eager
    preview). The subclass keeps the class name ``XGBClassifier`` so
    action-space labels and gating names are unchanged; labels are encoded in
    sorted order (the sklearn ``classes_`` convention), so ``predict_proba``
    columns line up with ``skrub_ops._resolve_scoring`` and the ensemble's
    ``np.unique(y_true)`` fallback. Imported lazily (and cached) so the module
    stays importable without xgboost.

    Example:
        cls = _xgb_classifier_shim()
        cls(n_estimators=10).fit(X, ["no", "yes", "no"]).predict(X)
        # -> array(["no", "yes", ...])  (original labels, not codes)
    """
    global _XGB_CLASSIFIER_SHIM
    if _XGB_CLASSIFIER_SHIM is not None:
        return _XGB_CLASSIFIER_SHIM
    import numpy as np
    import xgboost

    class XGBClassifier(xgboost.XGBClassifier):
        def fit(self, X, y, **kwargs):
            y = np.asarray(y)
            labels = np.unique(y)
            # expose the original labels only AFTER the base fit: xgboost's
            # own fit validates np.unique(codes) against self.classes_
            self._label_classes = None
            fitted = super().fit(X, np.searchsorted(labels, y), **kwargs)
            self._label_classes = labels
            return fitted

        def predict(self, X, **kwargs):
            codes = np.asarray(super().predict(X, **kwargs)).astype(int)
            return self._label_classes[codes]

        @property
        def classes_(self):
            labels = getattr(self, "_label_classes", None)
            return super().classes_ if labels is None else labels

    _XGB_CLASSIFIER_SHIM = XGBClassifier
    return _XGB_CLASSIFIER_SHIM


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


def _json_options(opts):
    """Coerce list-valued choice options to tuples.

    JSON cannot express a tuple, so `{"choice": [[1, 1], [1, 2]]}` reaches us as
    lists. sklearn validates several params as tuples specifically
    (`ngram_range`, `RobustScaler.quantile_range`, `hidden_layer_sizes`, ...) and
    raises `InvalidParameterError` on a list — the rollout scores 0.0 and the
    option loses by forfeit rather than on merit. Params that accept a list
    accept a tuple too, so the coercion is safe in the other direction.

    Example:
        _json_options([[1, 1], [1, 2]])   # -> [(1, 1), (1, 2)]
        _json_options(["sqrt", "log2"])   # -> ['sqrt', 'log2']
    """
    return [tuple(o) if isinstance(o, list) else o for o in opts]


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
        if log and low <= 0:
            log = False  # skrub's choose_* rejects a log scale at low<=0; keep
            # the param on a linear scale rather than raising (an LLM commonly
            # pairs log=true with a 0.0 lower bound — e.g. learning_rate)
        if typ == "int":
            return skrub.choose_int(
                int(low), int(high), log=log, name=choice_name
            )
        return skrub.choose_float(low, high, log=log, name=choice_name)
    if typ == "choice":
        allowed = rule["options"]
        wanted = llm_rule.get("choice") or llm_rule.get("options") or allowed
        opts = [o for o in _json_options(wanted) if o in allowed] or allowed
        return skrub.choose_from(opts, name=choice_name)
    return None


# Params whose tuning would break the determinism invariant (seeded centrally
# in `_ensure_seeded`), so they are never exposed as search dimensions.
_RNG_PARAMS = frozenset({"random_state"})

# skrub's DataOps carry pandas frames, which cannot hold a sparse matrix, so a
# sparse-emitting transformer raises inside `build_staged_plan` and takes the
# whole run down. Forced to False on any operator that accepts them.
_SPARSE_PARAMS = frozenset({"sparse_output", "sparse"})

# Document-frequency thresholds on the sklearn text vectorizers. skrub eagerly
# PREVIEWS every `.skb.apply` on a SINGLE row, and `min_df >= 2` is impossible on
# one document (skrub uses a choose_int's midpoint for that preview, so a range
# like [1,5] previews at 3) -> sklearn raises "max_df corresponds to < documents
# than min_df" and build_staged_plan takes the run down. Dropped like
# _SPARSE_PARAMS; the vectorizer's own defaults (min_df=1, max_df=1.0) are
# 1-row-safe. These operators are usually also removed by the sparse-output
# screen (`_emits_dataframe`), but dropping the param keeps the 1-row build
# preview safe for anything that slips through. See docs/BUG_LEDGER.md.
_DOC_FREQ_PARAMS = frozenset({"min_df", "max_df"})


def _accepts_param(cls, pname) -> bool:
    """True if ``cls.__init__`` accepts ``pname`` (or has ``**kwargs``).

    Lets an arbitrary LLM-proposed hyperparameter be dropped *individually* when
    the operator has no such constructor argument, instead of failing the whole
    ``cls(**kwargs)`` and losing the operator. Permissive if the signature can't
    be introspected (the try/except in ``_make`` remains the final guard).

    Example:
        _accepts_param(RandomForestRegressor, "n_estimators")   # -> True
        _accepts_param(RandomForestRegressor, "learning_rate")  # -> False
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return pname in params


def _names_param(cls, pname) -> bool:
    """True only if ``cls.__init__`` names ``pname`` explicitly.

    Unlike ``_accepts_param`` this is False for a ``**kwargs`` constructor, so a
    param can be *injected* rather than merely tolerated. LGBM/XGB swallow any
    keyword, and forcing e.g. ``sparse_output`` on them would reach the booster.

    Example:
        _names_param(OneHotEncoder, "sparse_output")   # -> True
        _names_param(LGBMClassifier, "sparse_output")  # -> False (**kwargs)
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return pname in params


def _build_free_choice(choice_name, llm_rule):
    """Build a skrub choose_* node from the LLM's OWN rule (no curated bounds).

    Used when the LLM tunes a hyperparameter that has no ``REGISTRY`` entry: the
    range is taken AS GIVEN (the safety envelope is the import allow-list, not
    per-param clipping), with only structural sanity — two finite numeric bounds
    with ``low < high``, and log scales need ``low > 0``. The rule self-describes
    its type: ``{"int": [lo, hi]}``, ``{"float": [lo, hi], "log": true}`` (or the
    ``{"type": "int"|"float", "range": [...]}`` form), or ``{"choice": [...]}``.
    Returns None for a malformed rule (the param is then simply omitted).

    Example:
        _build_free_choice("model__Lasso__alpha", {"float": [1e-4, 10], "log": True})
        # -> skrub.choose_float(1e-4, 10, log=True, name="model__Lasso__alpha")
    """
    if not isinstance(llm_rule, dict):
        return None
    for typ in ("int", "float"):
        rng = llm_rule.get(typ)
        if rng is None and llm_rule.get("type") == typ:
            rng = llm_rule.get("range")
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        try:
            low, high = float(rng[0]), float(rng[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
            return None
        log = bool(llm_rule.get("log", False))
        if log and low <= 0:
            return None
        if typ == "int":
            return skrub.choose_int(
                int(low), int(high), log=log, name=choice_name
            )
        return skrub.choose_float(low, high, log=log, name=choice_name)
    opts = llm_rule.get("choice") or llm_rule.get("options")
    if isinstance(opts, list) and opts:
        return skrub.choose_from(_json_options(opts), name=choice_name)
    return None


# A tiny multi-word text sample for the sparse-output screen. Tokens are 2+
# chars (sklearn's default token_pattern drops single chars) and shared across
# rows so a vocabulary forms; sklearn's text vectorizers accept a bare list.
_PROBE_DOCS = [
    "alpha beta gamma delta",
    "beta gamma alpha epsilon",
    "alpha gamma delta zeta",
    "beta alpha gamma eta",
]


def _emits_dataframe(cls) -> bool:
    """False only if `cls` is a transformer that positively emits a non-pandas
    container (scipy sparse) on text input; True otherwise (keep).

    skrub's DataOps carry pandas frames; a column transformer whose
    ``fit_transform`` returns a scipy-sparse matrix — sklearn's text vectorizers
    (``TfidfVectorizer``/``CountVectorizer``/``HashingVectorizer``) — raises
    inside ``build_staged_plan`` and takes the whole run down, and (unlike
    ``sparse_output``) has no constructor flag to force dense. Output container
    type is a property of the CLASS, not its hyperparameters, so we probe a bare
    default instance — needing none of the tuned params, dodging choose_* /
    min_df quirks. Anything that can't consume the text probe (every numeric
    transformer, or a predictor with no ``fit_transform``) raises and is KEPT:
    the screen only removes operators it positively proves emit non-frames.
    skrub's own encoders always return frames (and probing ``skrub.TextEncoder``
    would download a model), so ``_make`` screens only sklearn-rooted operators.

    Example:
        _emits_dataframe(TfidfVectorizer)   # -> False (drop)
        _emits_dataframe(StandardScaler)    # -> True  (keep; raises on text)
    """
    import pandas as pd

    try:
        est = cls()
    except Exception:
        return True  # needs constructor args -> can't cheaply probe, keep
    try:
        if hasattr(est, "set_output"):
            est.set_output(transform="pandas")  # what skrub would request
    except Exception:
        pass
    try:
        out = est.fit_transform(_PROBE_DOCS)
    except Exception:
        return True  # can't consume the text probe -> not a sparse vectorizer
    return isinstance(out, pd.DataFrame)


def _make(path, params, seed, context):
    """Lazily import `path` and instantiate it, wrapping tuned params in skrub
    choose_* nodes. Returns None only if the class can't be imported/built.

    Hyperparameters may be *curated* (a ``REGISTRY`` tunable rule -> range
    clipped to vetted bounds) or *arbitrary* (no registry rule -> the LLM's own
    range is used as-is via ``_build_free_choice``; the safety envelope is the
    import allow-list, not per-param clipping). Two kinds of param are dropped
    *individually*, never dropping the operator: one the class does not accept
    (``_accepts_param``) and any RNG-identity param (``_RNG_PARAMS``, seeded
    centrally for determinism).
    """
    cls = _load_class(path)
    if cls is None:
        return None
    if path == "xgboost.XGBClassifier":
        cls = _xgb_classifier_shim()  # string-label safety, same class name
    # Sparse-output screen: an sklearn transformer that emits scipy-sparse (the
    # text vectorizers) cannot live in skrub's pandas DataOps and crashes
    # build_staged_plan. Drop it here instead. Only sklearn-rooted transformer
    # slots are screened — skrub's own encoders always emit frames, and models
    # (context "model") are predictors, not transformers.
    if (
        context != "model"
        and path.split(".", 1)[0] == "sklearn"
        and not _emits_dataframe(cls)
    ):
        return None
    entry = REGISTRY.get(path, {})
    tunable = entry.get("tunable", {})
    kwargs = dict(entry.get("defaults", {}))
    for pname, llm_rule in (params or {}).items():
        if (
            pname in _RNG_PARAMS
            or pname in _SPARSE_PARAMS
            or pname in _DOC_FREQ_PARAMS
        ):
            continue  # omit this param, keep building the operator
        if not _accepts_param(cls, pname):
            continue
        name = f"{context}__{cls.__name__}__{pname}"
        rule = tunable.get(pname)
        try:
            choice = (
                _build_choice(name, llm_rule, rule)
                if rule is not None
                else _build_free_choice(name, llm_rule)
            )
        except Exception:
            choice = None  # a malformed range (skrub choose_* rejected it)
            # drops just this param — never the operator, and never the whole
            # resolve (which would silently kill an entire Option-3 injection)
        if choice is not None:
            kwargs[pname] = choice
    for pname in _SPARSE_PARAMS:
        if _names_param(cls, pname):
            kwargs[pname] = False
    try:
        return _ensure_seeded(cls(**kwargs), seed)
    except Exception:
        return None


def _class_default(cls, pname):
    """The constructor default for ``pname``, or ``inspect._empty`` if none."""
    try:
        param = inspect.signature(cls.__init__).parameters.get(pname)
    except (ValueError, TypeError):
        return inspect._empty
    return param.default if param is not None else inspect._empty


def _default_first(cls, pname, rule):
    """Reorder a ``{"choice": [...]}`` rule so the class default is FIRST.

    A skrub ``choose_from`` defaults to its first option, and that option seeds
    the search's ROOT config — so putting the operator's own constructor default
    first keeps the root at stock behavior (the robust root) regardless of how
    the LLM ordered the list. No-op for non-choice rules or when the default is
    not among the options.

    Example:
        _default_first(skrub.Cleaner, "drop_if_constant", {"choice": [True, False]})
        # -> {"choice": [False, True]}   (Cleaner's default is False)
    """
    if not (isinstance(rule, dict) and "choice" in rule):
        return rule
    opts = list(rule["choice"])
    default = _class_default(cls, pname)
    if default is not inspect._empty and default in opts:
        opts = [default] + [o for o in opts if o != default]
    return {**rule, "choice": opts}


def _resolve_backbone(path, spec_obj, seed, context, priors=None):
    """Build an ALWAYS-ON backbone operator (``Cleaner`` / ``TableVectorizer``).

    Its searchable knobs come only from the LLM ``spec_obj`` — there is NO
    code-owned fallback menu: an unspecified knob simply keeps skrub's own
    constructor default, so the bare ``cls()`` is the robust root. ``spec_obj``
    is ``{"params": {<scalar HP-rules>}, "slots": {<slot>: [<operator paths>]}}``:

    - scalar ``params`` (e.g. ``cardinality_threshold``, ``drop_if_constant``)
      become ``choose_*`` nodes, HP-style; a ``choice`` list is reordered
      default-first so the root stays at stock behavior.
    - each ``slots`` entry (an estimator-valued slot like ``low_cardinality`` /
      ``high_cardinality`` / ``numeric`` / ``datetime``) resolves its option
      list to instances and, if >1, a ``choose_from`` for that slot.

    Unknown / unaccepted params and slots are dropped individually (never the
    operator); returns the instance, or None only if the class can't be imported.

    Example:
        _resolve_backbone("skrub.Cleaner",
            {"params": {"drop_if_constant": {"choice": [False, True]}}}, 42, "cleaner")
        # -> Cleaner(drop_if_constant=choose_from([False, True], name=...))
    """
    cls = _load_class(path)
    if cls is None:
        return None
    kwargs: dict = {}
    if isinstance(spec_obj, dict):
        for pname, rule in (spec_obj.get("params") or {}).items():
            if pname in _RNG_PARAMS or not _accepts_param(cls, pname):
                continue
            name = f"{context}__{cls.__name__}__{pname}"
            try:
                node = _build_free_choice(
                    name, _default_first(cls, pname, rule)
                )
            except Exception:
                node = None
            if node is not None:
                kwargs[pname] = node
        for slot, items in (spec_obj.get("slots") or {}).items():
            if not _accepts_param(cls, slot):
                continue
            name = f"{context}__{slot}"
            slot_priors: dict = {}
            insts = _resolve_options(
                items, seed, False, name, priors_out=slot_priors
            )
            if len(insts) == 1:
                kwargs[slot] = insts[0]
            elif len(insts) >= 2:
                kwargs[slot] = skrub.choose_from(insts, name=name)
                # priors only matter for a real (>=2 option) search dimension
                if priors is not None and slot_priors:
                    priors[name] = slot_priors
    try:
        return cls(**kwargs)
    except Exception:
        return None


def _resolve_options(
    items, seed, allow_skip, context, priors_out=None, label_fn=repr
):
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


def _resolve_scoped(
    entries, seed, main_columns, priors: dict | None = None
) -> list[dict]:
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
            c
            for c in (cfg.get("cols") or [])
            if main_columns is None or c in main_columns
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
_AGG_OPERATIONS = {
    "count",
    "mode",
    "min",
    "max",
    "sum",
    "median",
    "mean",
    "std",
}


def _resolve_assemble(
    entries, aux_schemas, main_columns, priors: dict | None = None
) -> list[dict]:
    """Validate LLM assemble (AggJoiner) configs against the real table schemas.

    Same philosophy as HP clipping: hallucinated names are dropped, never
    executed. An entry survives only if its table is a known auxiliary table and
    its join keys exist in the respective tables; ``cols`` is intersected with
    the aux columns (dropped entirely if nothing survives, meaning "aggregate
    all"). Operations that were GIVEN but are all unsupported drop the entry (a
    hallucination); operations left EMPTY default to ``["mean"]`` — the planner
    correctly emits none for a 1-to-1 lookup join, where mean of the single
    matched row is the identity, so the enrichment join still runs. Without
    ``aux_schemas`` (single-table run) every entry is dropped.

    Example:
        _resolve_assemble(
            [{"table": "products", "key": "ID", "operations": ["mean", "bogus"]}],
            aux_schemas={"products": ["ID", "price"]}, main_columns=["ID", "x"])
        # -> [{"table": "products", "key": "ID", "operations": ["mean"], ...}]
    """
    # A model sometimes emits a single dict instead of a list of them (the
    # schema wants a list). Wrap it so it validates rather than iterating its
    # keys — and, if its nested shape is still wrong, drops cleanly below
    # instead of raising `unhashable type: dict` on `op in _AGG_OPERATIONS`.
    if isinstance(entries, dict):
        entries = [entries]

    out = []
    for cfg in entries or []:
        if not isinstance(cfg, dict):
            continue
        operations_raw = cfg.get("operations") or []
        if not all(isinstance(op, str) for op in operations_raw):
            continue  # nested per-column groups etc. — not our flat schema
        table = cfg.get("table")
        if not aux_schemas or table not in aux_schemas:
            continue
        aux_cols = set(aux_schemas[table])
        operations = [op for op in operations_raw if op in _AGG_OPERATIONS]
        if not operations:
            if operations_raw:
                continue  # operations were given but ALL invalid -> hallucinated
            # None given: a 1-to-1 relational table (country-happiness: one
            # GDP/life-exp row per country) has nothing to aggregate, so the
            # planner rightly emits `operations: []` — but AggJoiner requires
            # one. Default to "mean": on a single matched row it is the identity
            # (the value itself) and stays valid for a genuine 1-to-many join,
            # so the feature-enrichment join is never silently dropped.
            operations = ["mean"]
        key, main_key, aux_key = (
            cfg.get("key"),
            cfg.get("main_key"),
            cfg.get("aux_key"),
        )
        if key is not None:
            if key not in aux_cols or (
                main_columns and key not in main_columns
            ):
                continue
            keys = {"key": key}
        elif main_key is not None and aux_key is not None:
            if aux_key not in aux_cols or (
                main_columns and main_key not in main_columns
            ):
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

    Returns the always-on ``cleaner`` / ``vectorizer`` backbone instances (their
    LLM knobs wrapped in choose_* nodes, bare defaults when unauthored),
    instance-valued ``stages``, and a ``{class_name: instance}`` ``model`` dict
    where tuned params are skrub choose_* nodes. ``assemble`` entries are
    validated against ``aux_schemas``
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
            items,
            seed,
            allow_skip,
            context,
            priors_out=stage_priors,
            label_fn=label_fn,
        )
        if stage_priors:
            priors[context] = stage_priors
        return opts

    # Always-on backbones. The pipeline ALWAYS runs a skrub.Cleaner then a
    # skrub.TableVectorizer; their searchable knobs come only from the LLM
    # ``cleaner`` / ``vectorizer`` specs (params + estimator slots, resolved like
    # hyperparameters). Any knob left unauthored keeps skrub's own default, so a
    # bare ``Cleaner()`` / ``TableVectorizer()`` is the robust root — there is no
    # code-owned fallback menu.
    out["cleaner"] = _resolve_backbone(
        "skrub.Cleaner", spec.get("cleaner"), seed, "cleaner", priors=priors
    )
    out["vectorizer"] = _resolve_backbone(
        "skrub.TableVectorizer",
        spec.get("vectorizer"),
        seed,
        "vectorizer",
        priors=priors,
    )

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

    # Diagnostics: a malformed section resolves to nothing and vanishes with no
    # error (the assemble dict-vs-list bug); a stage left with <2 options has no
    # choice node and is not searched. Both are silent quality losses — surface
    # them so the driver/summary can flag them. (The cleaner/vectorizer backbones
    # always resolve to at least a bare default, so they can't be "dropped".)
    dropped: list[str] = []
    single_option: list[str] = []
    _raw_present = {
        "assemble": spec.get("assemble"),
        "scoped_encodings": spec.get("scoped_encodings"),
    }
    for name, raw_val in _raw_present.items():
        if raw_val and not out.get(name):
            dropped.append(name)
    for stage in spec.get("stages", []) or []:
        nm = stage.get("name", "stage")
        if stage.get("options") and not any(
            s["name"] == nm for s in out.get("stages", [])
        ):
            dropped.append(f"stage:{nm}")
    for stage in out.get("stages", []):
        if len(stage.get("options") or []) < 2:
            single_option.append(f"stage:{stage['name']}")
    for grp in out.get("scoped_encodings", []):
        # each scoped group gets an implicit skip at build, so <1 real option
        # is the degenerate case
        if len(grp.get("options") or []) < 1:
            single_option.append(f"scope:{grp.get('name', '?')}")
    if dropped:
        out["dropped_sections"] = dropped
    if single_option:
        out["single_option_stages"] = single_option
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

    for key in ("cleaner", "vectorizer"):
        obj = spec.get(key)
        if isinstance(obj, dict):
            for items in (obj.get("slots") or {}).values():
                check(items)
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
        "models": sorted(
            p for p, e in REGISTRY.items() if e["kind"] == "model"
        ),
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
        '{"choice":[...]}. You may tune ANY constructor hyperparameter of an '
        "allowed class (not only the ones listed below), and you choose the "
        "ranges yourself — they are used AS GIVEN (only the import is "
        "allow-listed; ranges are NOT clipped). So keep ranges sensible and, "
        "IMPORTANTLY, avoid values that make a single fit extremely slow: the "
        "search cross-validates many configs under a wall-clock budget, so an "
        "oversized upper bound (e.g. n_estimators or max_iter in the tens of "
        "thousands, very large n_components, a high polynomial degree) wastes "
        "the whole budget on one config. Keep upper bounds to what trains in a "
        'few seconds on a subsample. For an optional stage use "skip". Choose '
        "the task-appropriate class (Regressor for regression, Classifier for "
        "classification). Only sklearn.*, skrub.*, lightgbm.* and xgboost.* "
        "paths are allowed (lightgbm/xgboost are gradient-boosting model "
        "families — often the strongest on tabular data). For FREE-TEXT columns "
        "use skrub's own encoders — skrub.StringEncoder (default, TF-IDF-like), "
        "skrub.TextEncoder (semantic embeddings), or skrub.GapEncoder / "
        "skrub.MinHashEncoder for high-cardinality strings. Do NOT use the raw "
        "sklearn text vectorizers (sklearn.feature_extraction.text.Tfidf/Count/"
        "HashingVectorizer): they emit scipy-sparse matrices the skrub pipeline "
        "cannot carry, so they are dropped. The "
        "following have curated known-good ranges you can use as a guide:",
        "Transformers:",
    ]
    for path in sorted(
        p for p, e in REGISTRY.items() if e["kind"] == "transformer"
    ):
        tun = _fmt_tunable(REGISTRY[path]["tunable"])
        lines.append(f"  {path}" + (f" (params: {tun})" if tun else ""))
    lines.append("Models:")
    for path in sorted(p for p, e in REGISTRY.items() if e["kind"] == "model"):
        tun = _fmt_tunable(REGISTRY[path]["tunable"])
        lines.append(f"  {path}" + (f" (params: {tun})" if tun else ""))
    return "\n".join(lines)
