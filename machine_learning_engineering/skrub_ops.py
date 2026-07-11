"""skrub DataOps introspection and evaluation wrappers (Track A + Track B glue).

Everything that touches the skrub API lives here, so the MCTS engine (mcts.py) stays pure and the extractor logic has one home.

Verified against skrub 0.9.0 (run tests/test_skrub_ops.py inside Docker). The public .skb API has no structured param-grid accessor and
make_learner() takes no params argument, so this module uses the internal `skrub._data_ops._evaluation` helpers (`choices`, `set_params`) — they are what make_grid_search itself is built on. If a skrub upgrade breaks anything, it breaks here and nowhere else.

Determinism requirements (UCT values never converge otherwise):
- estimators inside the plan must be seeded (random_state=42 in the fixture);
- rollout subsampling is done by passing a seeded df.sample() through cross_validate(environment=...) — skrub's own .skb.subsample(how="random" is NOT seeded in 0.9.
"""

import functools
import numbers
import re
import signal
import threading
from contextlib import contextmanager
from typing import Callable

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin, clone
from skrub import selectors as _selectors
from skrub._data_ops import _evaluation as _ev


class _RolloutTimeout(BaseException):
    """Signals that a rollout exceeded its wall-clock budget.

    A ``BaseException`` (not ``Exception``) **on purpose**: skrub/sklearn
    ``cross_validate`` catches ``Exception`` inside each fold's fit
    (``error_score`` / ``FitFailedWarning``), so an ``Exception`` raised by the
    timeout would only NaN a single fold and the CV would keep running. A
    ``BaseException`` escapes that handler and aborts the whole CV, so the
    rollout catches it explicitly and returns 0.0.
    """


@contextmanager
def _time_limit(seconds: float | None):
    """Best-effort wall-clock cap on the wrapped block via SIGALRM.

    Raises ``_RolloutTimeout`` if the block runs longer than ``seconds``. A
    no-op when ``seconds`` is falsy/non-positive, off the main thread, or on a
    platform without SIGALRM — there the block simply runs uncapped (the
    caller's error handling still applies). The alarm fires between Python-level
    iterations (e.g. an estimator's ``n_estimators`` / ``max_iter`` loop), so it
    catches the iteration-count runaways free-form HP ranges can create; it
    cannot preempt a single non-yielding C call.

    Example:
        with _time_limit(45):          # aborts to _RolloutTimeout after 45s
            plan.skb.cross_validate(...)
    """
    armable = (
        bool(seconds)
        and seconds > 0
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not armable:
        yield
        return

    def _fire(signum, frame):
        raise _RolloutTimeout

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)  # always disarm
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# Track A — introspection (replaces the AST-based extractor)
# ---------------------------------------------------------------------------


def get_choices(plan) -> dict[str, tuple[int, object]]:
    """All choice nodes in the DAG as {choice_name: (choice_id, choice_object)}.

    Discrete choices (choose_from) have .outcomes / .outcome_names;
    numeric ones (choose_int/choose_float) have .low / .high / .log / .to_int.
    Unnamed choices are keyed by their numeric id.

    Example:
        get_choices(plan)
        # -> {"encoder": (0, <Choice>), "n_trees": (2, <NumericChoice>),
        #     "model": (4, <Choice>)}
    """
    # Dictionary creation using choice_name if available, else falling back to choice_id as string.
    # Values are tuples of (choice_id, choice_object) for easy access to both.
    return {
        (c.name if c.name is not None else str(cid)): (cid, c)
        for cid, c in _ev.choices(plan).items()
    }


def _option_names(choice) -> list[str]:
    """Human-readable option labels for a discrete choice.

    Dict-based choose_from has explicit outcome_names (['GBM', 'RF']);
    list-based ones fall back to the repr ('GapEncoder()'), which matches
    what describe_defaults() reports as the current value.

    Example:
        _option_names(model_choice)    # dict   -> ["GBM", "RF"]
        _option_names(encoder_choice)  # list   -> ["GapEncoder()", "MinHashEncoder()"]
    """
    # Discrete choices have .outcomes and optionally .outcome_names;
    # Unsuitable for numeric choices, which don't have .outcomes and are handled separately in get_action_space.
    if getattr(choice, "outcome_names", None):
        return list(choice.outcome_names)
    return [str(outcome) for outcome in choice.outcomes]


def _numeric_default(choice):
    """skrub's default outcome for a numeric choice, or None if unavailable.

    `Choice.default` is a plain value on some skrub versions and a method on
    others; anything non-numeric means "no usable default". Tested with
    `numbers.Number`, not `(int, float)` — skrub returns `np.int64`, which is
    not a Python `int` (though `np.float64` *is* a `float`, so an isinstance
    check silently keeps the float dims and drops the integer ones).

    Example:
        _numeric_default(skrub.choose_int(100, 1000, name="n"))  # -> 550
    """
    default = getattr(choice, "default", None)
    if callable(default):
        try:
            default = default()
        except Exception:
            return None
    if isinstance(default, bool) or not isinstance(default, numbers.Number):
        return None
    return default


def get_action_space(plan, n_numeric_options: int = 4) -> dict[str, list]:
    """The authoritative MCTS action space: {choice_name: [options]}.

    Comes from the DAG, never from an LLM (anti-pattern #1). Numeric ranges
    are discretized into `n_numeric_options` values (geometric spacing for
    log-scale choices) because MCTS needs a finite action set.

    Input:  a built plan (DataOp).
    Output: {choice_name: [options]}, discrete options as labels, numeric as values.

    Example:
        get_action_space(plan)
        # -> {"encoder": ["GapEncoder()", "MinHashEncoder()"],
        #     "n_trees": [50, 100, 125, 150, 200],  # choose_int(50, 200) + default
        #     "lr": [0.01, 0.031, 0.055, 0.097, 0.3],  # geomspaced + default
        #     "model": ["GBM", "RF"]}
    """
    space: dict[str, list] = {}
    for name, (_, choice) in get_choices(plan).items():
        if hasattr(choice, "outcomes"):
            space[name] = _option_names(choice)
        else:
            spacing = np.geomspace if choice.log else np.linspace
            values = spacing(choice.low, choice.high, n_numeric_options)
            cast = (lambda v: int(round(v))) if choice.to_int else float
            options = {cast(v) for v in values}
            # linspace/geomspace hit the endpoints and skip the centre, but
            # `get_default_state` seeds the root from skrub's default — the
            # range midpoint. Without it the root is unreachable by `expand`,
            # so every HP child is a jump to a far grid corner and the search
            # can only ever leave the default, never refine around it.
            default = _numeric_default(choice)
            if default is not None and choice.low <= default <= choice.high:
                options.add(cast(default))
            space[name] = sorted(options)
    return space


def get_choice_gating(plan) -> dict[str, tuple[str, str]]:
    """Map each conditional (model-gated) choice -> (parent_name, activating_label).

    skrub already tracks hyperparameters nested under a model option as
    *conditional children* of that model outcome (``_evaluation.choice_graph``'s
    ``children`` map, e.g. ``{(2, 0): [0], (2, 1): [1]}``). ``get_action_space``
    lists every choice flat, so the MCTS layer otherwise wastes budget editing a
    non-selected model's HPs. This returns the gate — keyed by the same names
    ``get_action_space`` uses — so ``mcts.expand`` only edits an HP when its
    parent model is selected.

    Example:
        get_choice_gating(plan)
        # -> {"model__RandomForestRegressor__n_estimators":
        #         ("model", "RandomForestRegressor"), ...}
    """
    graph = _ev.choice_graph(plan)
    choices = graph["choices"]  # {id: Choice}
    children = graph[
        "children"
    ]  # {(parent_id, outcome_idx): [child_ids], None: [...]}

    def name_of(cid: int) -> str:
        c = choices[cid]
        return c.name if getattr(c, "name", None) is not None else str(cid)

    gating: dict[str, tuple[str, str]] = {}
    for parent_key, child_ids in children.items():
        if parent_key is None:
            continue  # top-level choices are never gated
        parent_id, outcome_idx = parent_key
        labels = _option_names(choices[parent_id])
        if outcome_idx >= len(labels):
            continue
        activating = labels[outcome_idx]
        parent_name = name_of(parent_id)
        for child_id in child_ids:
            gating[name_of(child_id)] = (parent_name, activating)
    return gating


def get_state(plan) -> dict:
    """Current configuration as a compact dict — the MCTS state S.

    Example:
        get_state(plan)
        # -> {"encoder": "GapEncoder()", "lr": 0.055, "n_trees": 125, "model": "GBM"}
    """
    return plan.skb.describe_defaults()


def _reset_choices(plan) -> None:
    """Revert every choice to its intrinsic default (undo prior set_params).

    `set_params` mutates choices in place and `describe_defaults()` reflects
    that mutation, so applying a *partial* state without resetting first would
    leak the previous rollout's values. Discrete `Choice` resets via
    `chosen_outcome_idx=None`, numeric `NumericChoice` via
    `chosen_outcome=None`.

    Example:
        apply_state(plan, {"model": "RF"}); _reset_choices(plan)
        get_state(plan)["model"]   # -> "GBM"  (back to the default)
    """
    for _, choice in _ev.choices(plan).items():
        if hasattr(choice, "chosen_outcome_idx"):
            choice.chosen_outcome_idx = None
        if hasattr(choice, "chosen_outcome"):
            choice.chosen_outcome = None


def get_default_state(plan) -> dict:
    """The plan's pristine default config as action-space *labels*.

    Resets the plan first (so the read isn't contaminated by a prior
    `apply_state`), then reads discrete defaults from `get_action_space`
    itself (index 0 — skrub's `Choice.default` is always `outcomes[0]`,
    verified against skrub 0.9's `_choosing.py`) and true numeric defaults
    from `describe_defaults()`. This is the "nothing chosen yet" baseline for
    staged construction. NOTE: resetting the plan is a deliberate side effect.

    Do NOT match discrete defaults against `describe_defaults()`'s own string:
    for an outcome that itself contains a nested choice (e.g. an
    HP-tuned `encoder_options` entry), skrub abbreviates it to
    `"ClassName(...)"`, which never matches `get_action_space`'s full-repr
    label — round-tripping through that string silently produces an
    unappliable root state (`apply_state` raises, every rollout scores 0.0).

    Example:
        get_default_state(staged_plan)
        # -> {"scale": "None", "feature_eng": "None", "model": "LogReg"}
    """
    _reset_choices(plan)
    space = get_action_space(plan)
    choices_by_name = get_choices(plan)
    out = {}
    # Iterate describe_defaults()'s own keys, not every choice in the plan:
    # a numeric HP nested under a currently-*inactive* gated stage (e.g. a
    # feature_eng option whose default is skip) has no entry here at all —
    # that's correct, gating drops it (see mcts.canonicalize).
    for name, val in plan.skb.describe_defaults().items():
        _, choice = choices_by_name[name]
        out[name] = space[name][0] if hasattr(choice, "outcomes") else val
    return out


def get_steps_summary(plan) -> str:
    """Human-readable pipeline summary, used in ablation summaries.

    Example:
        get_steps_summary(plan)
        # -> "1. data\n2. drop(...)\n3. TableVectorizer\n4. choose_from(GBM, RF)..."
    """
    return plan.skb.describe_steps()


def find_node(plan, name: str):
    """Locate a named node in the DAG (targeting).

    Example:
        find_node(plan, "encoder")  # -> the encoder DataOp node
        find_node(plan, "nope")     # -> None
    """
    return plan.skb.find(name)


# ---------------------------------------------------------------------------
# Staged plan construction (the LLM "rich plan" hand-off)
# ---------------------------------------------------------------------------


def _skip_first(options: list) -> list:
    """Reorder so `None` (skip) is the default (first) outcome of a stage.

    Example:
        _skip_first([StandardScaler(), None])  # -> [None, StandardScaler()]
    """
    if None in options:
        return [None] + [o for o in options if o is not None]
    return options


def _scope_selector(cols: list[str]):
    """Missing-tolerant skrub selector matching exactly the named columns.

    `selectors.cols()` raises when a name is absent, so a column dropped by an
    upstream stage (e.g. Cleaner) would zero the whole rollout. A union of
    exact-match regexes selects whichever of the columns still exist and
    silently skips the rest — the scoped stage degrades to a no-op instead of
    failing.

    Example:
        _scope_selector(["title", "ghost"]).expand(df)  # -> ["title"]
    """
    selector = _selectors.regex(re.escape(cols[0]) + r"\Z")
    for col in cols[1:]:
        selector = selector | _selectors.regex(re.escape(col) + r"\Z")
    return selector


def _scoped_options(options: list) -> dict:
    """Label scoped-encoding options by class name, with 'skip' as default.

    Example:
        _scoped_options([GapEncoder(), MinHashEncoder()])
        # -> {"skip": None, "GapEncoder": ..., "MinHashEncoder": ...}
    """
    labeled = {"skip": None}
    for inst in options:
        if inst is None:
            continue
        label = base = type(inst).__name__
        i = 2
        while label in labeled:
            label, i = f"{base}_{i}", i + 1
        labeled[label] = inst
    return labeled


def _suffix_column(col: str, suffix: str) -> str:
    """Rename helper for additive scope groups (module-level so it pickles).

    Example:
        _suffix_column("signup_month", "date_feats")  # -> "signup_month__date_feats"
    """
    return f"{col}__{suffix}"


def _additive_scoped_options(options: list) -> dict:
    """Label additive-group options; 'skip' emits ZERO columns (not passthrough).

    An additive group's output is concatenated back onto the table, so its
    skip outcome cannot be None — None means passthrough, which would
    duplicate the scoped columns. SelectCols([]) yields an empty frame, so
    skip degrades the concat to a no-op.

    Example:
        _additive_scoped_options([DatetimeEncoder()])
        # -> {"skip": SelectCols(cols=[]), "DatetimeEncoder": ...}
    """
    labeled = _scoped_options(options)
    labeled["skip"] = skrub.SelectCols(
        []
    )  # replaces None, keeps first position
    return labeled


def _safe_names(names) -> list[str]:
    """Sanitize column names to ``[A-Za-z0-9_]``, de-duplicating collisions.

    LightGBM (and older XGBoost) reject feature names containing JSON special
    characters (``[]{}":,`` ...), which skrub's encoders and ``AggJoiner``
    routinely emit (e.g. ``Physician_Specialty: Foo|Bar``). Every other
    character becomes ``_``; a resulting collision gets a numeric suffix so
    names stay unique (boosters reject duplicate names too).

    Example:
        _safe_names(["a|b", "a:b", "ok"])  # -> ["a_b", "a_b_1", "ok"]
    """
    used: set[str] = set()
    out: list[str] = []
    for n in names:
        base = re.sub(r"[^0-9A-Za-z_]", "_", str(n)) or "col"
        name, i = base, 1
        while name in used:
            name, i = f"{base}_{i}", i + 1
        used.add(name)
        out.append(name)
    return out


class _SanitizeColumns(BaseEstimator, TransformerMixin):
    """Rename all columns to booster-safe identifiers before the model fit.

    A fixed (non-searchable) whole-frame step inserted just before the model so
    an injected LightGBM/XGBoost can fit at all: skrub's encoded / aggregated
    column names carry JSON special characters that LightGBM rejects outright,
    silently scoring the rollout 0.0. Models that ignore feature names (HGB,
    sklearn RF / linear) are unaffected — values pass through untouched. See
    ``_safe_names``.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = (
            list(X.columns) if hasattr(X, "columns") else None
        )
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            X = X.copy()
            X.columns = _safe_names(list(X.columns))
        return X

    def get_feature_names_out(self, input_features=None):
        names = (
            self.feature_names_in_
            if self.feature_names_in_ is not None
            else input_features
        )
        return np.asarray(_safe_names(list(names)))


def _is_array_cell(value) -> bool:
    """True for a non-string sequence stored inside a dataframe cell.

    Example:
        _is_array_cell(np.array(["Y", "Z"]))  # -> True
        _is_array_cell("Y")                   # -> False
    """
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, (list, tuple, np.ndarray, pd.Series)) or isinstance(
        value, pd.api.extensions.ExtensionArray
    )


def _first_cell(value):
    """Collapse an array cell to its first element; pass scalars through."""
    if not _is_array_cell(value):
        return value
    return value[0] if len(value) else np.nan


class _ScalarizeAggregates(BaseEstimator, TransformerMixin):
    """Collapse array-valued cells left behind by ``AggJoiner(operations="mode")``.

    On a modal TIE, skrub's ``mode`` aggregation writes the whole tied array
    into the cell (``["Y", "Z"]``) instead of a scalar. The column becomes
    ``object`` dtype mixing scalars, arrays and NaN, and skrub's own
    ``TableVectorizer`` then dies inside ``CleanNullStrings`` (pandas
    ``replace`` does ``operator.eq`` on the array -> "truth value of an empty
    array is ambiguous"). Every CV fold fails, so the whole assemble option
    silently scores 0.0 — credit-fraud's ``basket_mode`` did exactly that.

    A fixed, non-searchable step right after the assemble stage (it adds no
    ``choose_from``, so the action space and gating are untouched). Ties are
    broken by taking the first element, which is deterministic: pandas' mode
    returns its values sorted.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = (
            list(X.columns) if hasattr(X, "columns") else None
        )
        return self

    def transform(self, X):
        if not hasattr(X, "columns"):
            return X
        out = None
        for col in X.columns:
            if X[col].dtype != object:
                continue  # only an object column can hold an array cell
            if not X[col].map(_is_array_cell).any():
                continue
            if out is None:
                out = X.copy()
            out[col] = out[col].map(_first_cell)
        return X if out is None else out

    def get_feature_names_out(self, input_features=None):
        names = (
            self.feature_names_in_
            if self.feature_names_in_ is not None
            else input_features
        )
        return np.asarray(list(names))


class _ChainedAggJoiner(BaseEstimator, TransformerMixin):
    """Apply several ``AggJoiner`` steps in sequence, composing their columns.

    The ``assemble`` stage is a single exclusive ``choose_from``, so on a
    multi-table task (country-happiness: gdp + life_expectancy + legal_rights)
    the search can pick only ONE table's aggregates and discards the rest —
    yet the task's whole signal is spread across all of them. This wraps every
    join into one option so "use all aux tables" becomes selectable alongside
    each single-table option and ``skip``. Each joiner adds its columns to the
    growing frame; the shared main key (e.g. ``Country``) survives every join,
    so the chain is well-defined. Joiners are cloned per fit for determinism
    across CV folds.

    Example:
        _ChainedAggJoiner([AggJoiner(gdp, ...), AggJoiner(life, ...)])
        # fit_transform(main) -> main + gdp aggregates + life aggregates
    """

    def __init__(self, joiners):
        self.joiners = joiners

    def fit(self, X, y=None):
        self.fitted_ = []
        current = X
        for joiner in self.joiners:
            fitted = clone(joiner)
            current = fitted.fit_transform(current)
            self.fitted_.append(fitted)
        return self

    def transform(self, X):
        current = X
        for fitted in self.fitted_:
            current = fitted.transform(current)
        return current


def _apply_scope_group(node, group: dict, cols: list[str]):
    """Insert one scoped-encoding group into the DAG at the current node.

    Replace mode (default): the chosen operator's output SUBSTITUTES the
    scoped columns (`.skb.apply(cols=...)`). Additive mode
    (``group["additive"]``): the operator runs on a selected copy of the
    columns and its output — suffixed ``__<group name>`` against collisions —
    is concatenated back by row index, so the originals survive. 'skip' is
    the default no-op in both modes; the selector is missing-tolerant.

    Example:
        node = _apply_scope_group(
            node, {"name": "date_feats", "additive": True,
                   "options": [DatetimeEncoder()]}, ["signup"])
        # -> node with 'signup' kept AND signup_*__date_feats appended
    """
    name = group["name"]
    if group.get("additive"):
        sub = node.skb.apply(skrub.SelectCols(_scope_selector(cols)))
        derived = sub.skb.apply(
            skrub.choose_from(
                _additive_scoped_options(list(group["options"])),
                name=f"scope_{name}",
            )
        )
        renamed = derived.rename(
            columns=functools.partial(_suffix_column, suffix=name)
        )
        return node.skb.concat([renamed], axis=1)
    return node.skb.apply(
        skrub.choose_from(
            _scoped_options(list(group["options"])), name=f"scope_{name}"
        ),
        cols=_scope_selector(cols),
    )


def build_staged_plan(
    spec: dict, df, target: str = "target", aux_tables: dict | None = None
):
    """Build a skrub plan from an ordered, per-stage menu of operators.

    This is the shape produced by the LLM plan author (adk_agent.plan_author)
    after `spec_resolver.resolve_spec`: instead of one fixed pipeline, the LLM
    proposes, *for each stage*, a list of candidate operators (optionally with
    hyperparameter ranges). Each stage becomes a named `choose_from`, so the
    whole menu flows through `get_action_space` / `apply_state` unchanged and
    MCTS can
    search the *construction* of the pipeline, not just hyperparameters.

    Stages are applied in canonical pipeline order:
    assemble (relational) -> clean -> scoped encodings (pre_encode) ->
    encode -> scoped encodings (post_encode) -> post-encoding stages ->
    model. See docs/pipeline-stages.md for the taxonomy.

    `spec` shape (operators are real estimator instances, not strings —
    translating LLM text to instances is a separate concern):

        {
          # optional: aggregate-join auxiliary tables (relational data).
          # Needs aux_tables={name: dataframe}. Each entry -> an AggJoiner;
          # a 'skip' option is added automatically as the default.
          "assemble": [
            {"name": "aux_mean", "table": "aux", "operations": ["mean"],
             "key": "id", "cols": ["v"]},
          ],
          # optional: cleaning / type coercion before encoding
          "clean_options": [None, Cleaner()],
          # optional: scope — apply a searchable operator to SPECIFIC columns.
          # Column names are validated against df and matched
          # missing-tolerantly at runtime (see _scope_selector); a 'skip'
          # default is added. Two optional axes per group:
          #   "position": "pre_encode" (default; before the TableVectorizer)
          #     or "post_encode" (after it — only names the vectorizer
          #     passes through unchanged, i.e. numeric columns, still match);
          #   "additive": True to KEEP the original columns and concatenate
          #     the operator's output (suffixed __<name>) by row index,
          #     instead of replacing them (see _apply_scope_group).
          "scoped_encodings": [
            {"name": "title_enc", "cols": ["job_title"],
             "options": [GapEncoder(), MinHashEncoder()]},
            {"name": "date_feats", "cols": ["signup"], "additive": True,
             "options": [DatetimeEncoder()]},
            {"name": "num_scale", "cols": ["age"], "position": "post_encode",
             "options": [StandardScaler()]},
          ],
          # optional: encoder choice inside the TableVectorizer
          "encoder_options": [GapEncoder(), MinHashEncoder()],
          # optional: post-encoding numeric stages
          "stages": [
            {"name": "scale",       "options": [None, StandardScaler()]},
            {"name": "feature_eng", "options": [None, PolynomialFeatures(2)]},
          ],
          "model": {"GBM": ..., "RF": ..., "LogReg": ...},       # required
        }

    Conventions:
    - `None` (or the 'skip' key in assemble) is forced to be the *default*
      outcome, so an undecided stage = not-yet-enriched.
    - The model `choose_from` has a real default (its first entry), so a
      partial pipeline is always runnable.
    - The assemble stage uses a *labeled dict* `choose_from` so options read as
      'aux_mean' etc. rather than the AggJoiner repr (which embeds the table),
      and is followed by a fixed `_ScalarizeAggregates` step that repairs the
      array-valued cells `mode` leaves on a tie.

    Example (minimal, single-table):
        spec = {"encoder_options": [skrub.GapEncoder(), skrub.MinHashEncoder()],
                "model": {"GBM": GradientBoostingClassifier(),
                          "RF": RandomForestClassifier()}}
        plan = build_staged_plan(spec, df)        # df has a "target" column
        get_action_space(plan)
        # -> {"encoder": ["GapEncoder()", "MinHashEncoder()"], "model": ["GBM", "RF"]}
    """
    data = skrub.var("data", df)
    aux_vars = {
        name: skrub.var(name, adf) for name, adf in (aux_tables or {}).items()
    }
    X = data.drop(columns=target).skb.mark_as_X()
    y = data[target].skb.mark_as_y()
    node = X

    # --- assemble (relational): aggregate-join auxiliary tables ---
    if spec.get("assemble"):
        joiners = {"skip": None}
        made = []
        for cfg in spec["assemble"]:
            label = (
                cfg.get("name")
                or f"{cfg['table']}_{'_'.join(cfg['operations'])}"
            )
            joiner = skrub.AggJoiner(
                aux_vars[cfg["table"]],
                cfg["operations"],
                key=cfg.get("key"),
                main_key=cfg.get("main_key"),
                aux_key=cfg.get("aux_key"),
                cols=cfg.get("cols"),
            )
            joiners[label] = joiner
            made.append(joiner)
        # multi-table tasks: let the search combine ALL aux joins, not just one
        # (assemble is exclusive, so without this the other tables are lost).
        if len(made) >= 2:
            joiners["all_aggregates"] = _ChainedAggJoiner(made)
        node = node.skb.apply(skrub.choose_from(joiners, name="assemble"))
        # `mode` aggregation leaves an array in the cell on a tie, which kills
        # the TableVectorizer downstream (see _ScalarizeAggregates). Fixed
        # step, invisible to the action space.
        node = node.skb.apply(_ScalarizeAggregates())

    # --- clean / coerce ---
    if spec.get("clean_options"):
        node = node.skb.apply(
            skrub.choose_from(_skip_first(spec["clean_options"]), name="clean")
        )

    # --- scope: searchable operator on an explicit column subset ---
    scoped_groups = spec.get("scoped_encodings", []) or []

    def _scope_pass(node, position: str):
        for group in scoped_groups:
            if group.get("position", "pre_encode") != position:
                continue
            cols = [
                c
                for c in group.get("cols", [])
                if c in df.columns and c != target
            ]
            if not cols:
                continue  # nothing valid to scope over -> group dropped
            node = _apply_scope_group(node, group, cols)
        return node

    node = _scope_pass(node, "pre_encode")

    # --- encode / vectorize ---
    # The vectorization backbone is ALWAYS a skrub.TableVectorizer (this has
    # been true since the project's start): it routes numeric/datetime/low-card
    # columns through skrub's built-in transformers, and the searchable
    # `encoder` stage only chooses its `high_cardinality` slot. With no
    # `encoder_options`, that slot silently uses skrub's own default
    # (StringEncoder in 0.9) — so "no encoder in the plan" is NOT "no encoding".
    # `spec_resolver` adds that default as a companion to a lone encoder option
    # (`_default_high_cardinality_encoder`) so a required stage stays searchable.
    enc_opts = spec.get("encoder_options")
    if enc_opts:
        vectorizer = skrub.TableVectorizer(
            high_cardinality=skrub.choose_from(enc_opts, name="encoder")
        )
    else:
        vectorizer = skrub.TableVectorizer()
    node = node.skb.apply(vectorizer)

    # --- scope, post-encode: e.g. scale specific (passthrough-named) columns ---
    node = _scope_pass(node, "post_encode")

    # --- post-encoding numeric stages (scale, feature-eng, select) ---
    for stage in spec.get("stages", []):
        node = node.skb.apply(
            skrub.choose_from(
                _skip_first(list(stage["options"])), name=stage["name"]
            )
        )

    # --- sanitize feature names so an injected booster can fit (see
    # _safe_names): a fixed, non-searchable whole-frame rename, invisible to the
    # action space / gating (it adds no `choose_from`). ---
    node = node.skb.apply(_SanitizeColumns())

    # --- model ---
    node = node.skb.apply(
        skrub.choose_from(spec["model"], name="model").as_data_op(),
        y=y,
    )
    return node


# ---------------------------------------------------------------------------
# Track B — applying a state and rolling it out
# ---------------------------------------------------------------------------


def apply_state(plan, state: dict) -> None:
    """Configure the plan to match a state dict exactly (in place).

    Every choice is first reset to its default, then `state` is applied, so a
    *partial* state means "these stages decided, everything else default" —
    no value leaks from a previous call. Discrete values are option labels
    ('GBM', 'GapEncoder()'), numeric values are plain numbers. Unknown names
    or labels raise ValueError so rollouts can score the config 0.0 instead of
    silently evaluating the wrong pipeline.

    Input:  plan + a (possibly partial) state dict of {choice_name: label/value}.
    Output: None — the plan is mutated in place.

    Example:
        apply_state(plan, {"model": "RF", "rf_trees": 200})  # plan now uses RF/200
        apply_state(plan, {"model": "GBM"})                  # RF reset; rest default
        apply_state(plan, {"model": "DoesNotExist"})         # raises ValueError
    """
    _reset_choices(plan)
    choices_by_name = get_choices(plan)
    params = {}
    for name, value in state.items():
        if name not in choices_by_name:
            raise ValueError(f"unknown choice name: {name!r}")
        cid, choice = choices_by_name[name]
        if hasattr(choice, "outcomes"):
            labels = _option_names(choice)
            if value not in labels:
                raise ValueError(f"unknown option {value!r} for {name!r}")
            params[cid] = labels.index(value)
        else:
            params[cid] = value
    _ev.set_params(plan, params)


def _single_var_name(plan) -> str:
    names = plan.skb.get_vars() if hasattr(plan.skb, "get_vars") else ["data"]
    if isinstance(names, dict):
        names = list(names)
    if len(names) != 1:
        raise ValueError(f"expected exactly one input var, found: {names}")
    return names[0]
    # Example: _single_var_name(single_table_plan) -> "data"
    #          (raises for a relational plan with >1 input var)


def _cv_kwarg(stratify: bool, seed: int, n_splits: int = 5) -> dict:
    """A `cross_validate(cv=...)` kwarg, stratified when the target is rare.

    skrub's `cross_validate` calls `check_cv(cv)` **without** `y`/`classifier`,
    so it never auto-detects classification — it always defaults to plain
    `KFold`. On an imbalanced target (e.g. credit-fraud's ~1% positive rate)
    a random fold of a small subsample can land on zero positives, and
    sklearn's scorer then fails that fold silently (NaN, not an exception) —
    NaN rewards then poison the score cache without ever crashing. Passing an
    explicit `StratifiedKFold` (which skrub forwards straight to `cv.split`)
    fixes this; `stratify=False` keeps the old (regression) behavior.

    `n_splits` lets callers shrink the fold count when even a stratified
    subsample holds fewer minority rows than folds (see `make_rollout_fn`).

    Example:
        _cv_kwarg(True, 42, n_splits=3)  # -> {"cv": StratifiedKFold(3, ...)}
        _cv_kwarg(False, 42)             # -> {}
    """
    if not stratify:
        return {}
    from sklearn.model_selection import StratifiedKFold

    return {
        "cv": StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    }


def _stratified_subsample(df, target: str, n: int, seed: int, floor: int = 10):
    """A seeded per-class subsample of `n` rows with a minority floor.

    On extreme imbalance (credit-fraud is ~1.25% positive) a plain
    `df.sample(n=500)` captures only ~6 minority rows; StratifiedKFold(5)
    then tests on 1–2 positives per fold, so rewards are near-noise and a
    bad draw degenerates folds to a single class (roc_auc -> NaN). Sampling
    per class with `quota = min(count, max(floor, proportional share))`
    guarantees each class at least `floor` real rows (all of them when the
    data has fewer) — **without replacement, never duplicated**, since a
    duplicated row landing in both a train and a test fold would leak.
    The floor slightly shifts prevalence vs the full data; that is harmless
    for rank-based scorers (roc_auc) and negligible for accuracy at
    floor=10. Same `(df, target, n, seed)` -> identical frame.

    Example:
        small = _stratified_subsample(df, "fraud", n=500, seed=42)
        small["fraud"].value_counts().min()  # >= 10 (or all minority rows)
    """
    counts = df[target].value_counts()
    labels = sorted(counts.index, key=lambda c: (counts[c], str(c)))
    quotas = {
        c: min(int(counts[c]), max(floor, round(n * counts[c] / len(df))))
        for c in labels
    }
    # Trim overshoot from the largest classes first, never below the floor.
    for c in reversed(labels):
        excess = sum(quotas.values()) - n
        if excess <= 0:
            break
        quotas[c] = max(min(int(counts[c]), floor), quotas[c] - excess)
    parts = [
        df[df[target] == c].sample(n=quotas[c], random_state=seed)
        for c in labels
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def _profile_subsample_n(
    df,
    target: str | None,
    stratify: bool,
    min_rows: int = 500,
    cap: int = 2000,
) -> tuple[int, int]:
    """Size the rollout subsample from the data *profile*, not just row count.

    Returns ``(n, floor)`` for `_stratified_subsample`. The old fixed rule
    (``max(min_rows, 1% of rows)`` with a minority floor of 10) made rewards
    near-noise on imbalanced data: credit-fraud at 1.25% positive gave ~6-10
    positives in 500 rows, so CV folds tested on 1-2 each — observed directly
    when a budget-80 run chased that noisy proxy to a WORSE held-out config
    than budget-20. Profile-aware sizing:

    - imbalance: target ``floor = min(minority_count, 40)`` REAL minority rows
      (~8 per fold at 5 folds) and grow ``n`` toward ``floor / prevalence`` so
      the majority keeps scale, capped at ``cap`` rows for wall-clock;
    - high cardinality: any non-target string column with > 100 uniques bumps
      ``n`` to >= 1000 (encoders can't estimate a vocabulary from 500 rows);
    - relational fan-out: aux tables join whole (never subsampled), so the
      main-table ``n`` is left alone.

    The cap only limits profile-driven *growth* — a huge table whose 1% base
    already exceeds it keeps the old base size.

    Example:
        _profile_subsample_n(fraud_df, "fraud", True)   # -> (2000, 40)
        _profile_subsample_n(numeric_df, "y", False)    # -> (500, 10)
    """
    n = max(min_rows, int(0.01 * len(df)))
    feats = df.drop(columns=[target]) if target in df.columns else df
    text = feats.select_dtypes(include=["object", "string", "category"])
    if any(text[c].nunique() > 100 for c in text.columns):
        n = max(n, min(cap, 1000))
    floor = 10
    if stratify and target is not None and target in df.columns:
        counts = df[target].value_counts()
        floor = int(min(counts.min(), 40))
        prevalence = float(counts.min()) / len(df)
        if prevalence > 0:
            n = max(n, min(cap, int(floor / prevalence)))
    return min(len(df), n), floor


def _fold_mean(scores) -> float:
    """Mean over per-fold CV scores, skipping scorer-failed (NaN) folds.

    A degenerate fold (e.g. single-class under roc_auc) yields NaN, not an
    exception; averaging over the remaining folds keeps an otherwise-fine
    config alive. All folds failed -> 0.0 (worst bounded reward).

    Example:
        _fold_mean([0.8, float("nan")])  # -> 0.8
        _fold_mean([float("nan")] * 5)   # -> 0.0
    """
    arr = np.asarray(scores, dtype=float)
    if np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmean(arr))


# Scorers already confined to [0, 1]: their raw value IS a valid UCT reward, so
# they pass through untouched and every historical baseline on them still holds.
# Anything else (r2, and `None` = the estimator's own .score(), which is R2 for a
# regressor) is unbounded BELOW and must be squashed — see `_bounded_reward`.
_UNIT_SCORERS = frozenset(
    {"roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1"}
)


def _bounded_reward(score: float, scoring) -> float:
    """Map a raw CV score onto the (0, 1] UCT reward scale, order-preserving.

    UCT compares Q/N against an exploration term scaled by ``c``, so rewards
    must be bounded and higher-is-better. `r2` is bounded ABOVE by 1 but
    unbounded below, and a single catastrophic config (r2 = -50) would drag
    every ancestor's Q with it. Clamping the negative half to 0.0 bounds it but
    throws the ordering away — on a genuinely hard regression task every config
    scores r2 < 0, the whole landscape flattens and UCT is left choosing at
    random (observed on flight-delays and movielens).

    So unbounded-below scorers get a strictly increasing squash instead:

        f(s) = 1 / (2 - s)      f(1) = 1, f(0) = 0.5, f(-1) = 1/3, f(-inf) -> 0

    Ordering is preserved everywhere, catastrophes compress toward 0 instead of
    diverging, and 0.5 reads as "no better than predicting the mean". The reward
    is only a SEARCH scale: the reported metric (`evaluate_full`) is untouched,
    and 0.0 stays reserved for a failed config — strictly worse than any real
    one, since the squash never reaches 0.

    Scorers already in [0, 1] (`_UNIT_SCORERS`) pass through, so classification
    rewards keep their raw meaning.

    Example:
        _bounded_reward(0.66, "roc_auc")  # -> 0.66   (already bounded)
        _bounded_reward(0.0, "r2")        # -> 0.5    (predicts the mean)
        _bounded_reward(-49.0, "r2")      # -> 0.02   (bad, but still ordered)
    """
    if scoring in _UNIT_SCORERS:
        return min(1.0, max(0.0, float(score)))
    return 1.0 / (2.0 - min(1.0, float(score)))


def reward_scale(scoring) -> str:
    """How `_bounded_reward` maps this scorer's raw score onto the UCT scale.

    For reporting: `best_search_score` is a REWARD, not the raw scorer value,
    whenever the scorer is unbounded below.

    Example:
        reward_scale("accuracy")  # -> "raw"
        reward_scale("r2")        # -> "1/(2 - r2)"
    """
    return "raw" if scoring in _UNIT_SCORERS else f"1/(2 - {scoring or 'score'})"


# Ranking scorers that consume class *probabilities*. skrub's learner reports
# `_estimator_type == "transformer"`, so sklearn's built-in scorer never reduces
# a binary `predict_proba` to its positive column — a classifier that lacks
# `decision_function` (RandomForest, LightGBM, XGBoost) then hands a 2-column
# array straight to `roc_auc_score` and every fold fails (NaN -> 0.0). Only
# `HistGradientBoosting`/linear models dodged it via `decision_function`. We
# swap the scorer name for a callable that does the reduction itself.
_PROBA_SCORERS = ("roc_auc", "average_precision")


def _resolve_scoring(scoring):
    """Map a scorer name to a skrub-learner-safe scorer (callable or the name).

    Probability-ranking metrics (`roc_auc`, `average_precision`) become a
    callable that pulls the positive-class column out of `predict_proba` before
    scoring, so a classifier exposing only `predict_proba` (no
    `decision_function`) scores correctly through skrub's transformer-typed
    learner. Every other scorer name (and callables / None) passes through
    unchanged — `predict`-based metrics (accuracy, f1, r2) already work.

    Example:
        _resolve_scoring("accuracy")  # -> "accuracy"  (unchanged)
        _resolve_scoring("roc_auc")   # -> <callable positive-column roc_auc>
    """
    if scoring not in _PROBA_SCORERS:
        return scoring
    from sklearn import metrics as skmetrics

    score_func = {
        "roc_auc": skmetrics.roc_auc_score,
        "average_precision": skmetrics.average_precision_score,
    }[scoring]

    def _scorer(estimator, X, y):
        try:
            proba = np.asarray(estimator.predict_proba(X))
        except AttributeError:
            # decision_function-only classifiers (LinearSVC, SGD-hinge, SVC
            # without probability=True): skrub's learner raises
            # AttributeError for predict_proba, which would NaN every fold
            # (silent 0.0). Both metrics rank by score, so raw margins work.
            scores = np.asarray(estimator.decision_function(X))
            if scoring == "roc_auc" and scores.ndim == 2:
                return float(
                    score_func(
                        y,
                        scores,
                        multi_class="ovr",
                        labels=getattr(estimator, "classes_", None),
                    )
                )
            return float(score_func(y, scores))
        if proba.ndim == 2 and proba.shape[1] == 2:
            return float(score_func(y, proba[:, 1]))
        if scoring == "roc_auc" and proba.ndim == 2 and proba.shape[1] > 2:
            return float(
                score_func(
                    y,
                    proba,
                    multi_class="ovr",
                    labels=getattr(estimator, "classes_", None),
                )
            )
        return float(score_func(y, proba))

    _scorer.__name__ = scoring
    return _scorer


def make_rollout_fn(
    plan,
    df,
    min_rows: int = 500,
    seed: int = 42,
    aux: dict | None = None,
    main_var: str | None = None,
    scoring: str | None = None,
    stratify: bool = False,
    target: str | None = None,
    timeout_s: float | None = 60.0,
    n_subsample_seeds: int = 1,
    n_jobs: int = 1,
) -> Callable[[dict], float]:
    """Build a rollout_fn(state) -> float for mcts.mcts_search.

    Evaluates a configuration on an adaptive seeded subsample of the MAIN
    table, sized from the data profile (`_profile_subsample_n`: 1%-of-rows
    base, grown for imbalance / high-cardinality text, capped at 2000 for
    wall-clock, always capped at the full table). The same state always
    scores the same. Failed configs return 0.0.

    Works for both flat (complete) and staged (partial) states because
    `apply_state` resets to defaults before applying — a partial state simply
    leaves the undecided stages at their defaults.

    For relational plans, pass `aux={var_name: dataframe}` (auxiliary tables
    are joined whole, not subsampled) and `main_var` to name the main table's
    var (defaults to the plan's single var).

    `scoring` is an sklearn scorer name forwarded to cross_validate. It must be
    **higher-is-better** (MCTS maximizes). None keeps the estimator's default
    .score() (R2 for regressors). The raw CV score is mapped onto the bounded
    (0, 1] UCT reward scale by `_bounded_reward`: scorers already in [0, 1]
    (accuracy, roc_auc, f1) pass through unchanged, while an unbounded-below
    scorer (r2) is squashed order-preservingly through `1 / (2 - s)`. A failed
    config returns exactly 0.0, which the squash never reaches — so failures
    rank strictly below every config that actually ran.

    `stratify=True` (pass for classification, especially imbalanced targets)
    uses a seeded `StratifiedKFold` instead of skrub's plain-`KFold` default —
    see `_cv_kwarg` for why this matters. Passing `target` as well upgrades
    the *subsample* to a stratified one with a minority floor
    (`_stratified_subsample`) and shrinks `n_splits` when the subsampled
    minority is smaller than the fold count; without `target` the subsample
    stays the plain seeded `df.sample` of before. Per-fold NaNs (degenerate
    folds) are skipped via `_fold_mean` instead of failing the config.

    `timeout_s` (default 60s) is a per-config wall-clock cap: a config whose CV
    exceeds it scores 0.0 like any other failure (via `_time_limit`), so a
    free-form hyperparameter range that makes a single fit pathologically slow
    can't stall the search. `None`/0 disables the cap. Best-effort — see
    `_time_limit` (main-thread + SIGALRM only; can't preempt a non-yielding C
    call). With `n_subsample_seeds > 1` the cap scales with the seed count
    (the whole averaged rollout gets `timeout_s * n_subsample_seeds`). Note the
    cap is wall-clock, so on a loaded machine a config near the limit can flip
    between its real reward and 0.0 — raising `timeout_s` or `n_jobs` restores
    determinism by giving slow encoders (GapEncoder on high-cardinality text)
    headroom under the cap.

    `n_jobs` (default 1) is forwarded to sklearn's `cross_validate` to run CV
    folds in parallel — a clean ~n-fold speedup on high-cardinality-text tasks.
    It is inter-process (joblib forks each fold into its own address space), so
    it is SAFE with lightgbm/xgboost: their macOS-ARM double-libomp segfault
    comes from intra-process OpenMP threads (the ESTIMATOR's own n_jobs, pinned
    to 1 in REGISTRY), not from forking folds. Determinism is preserved: folds
    are averaged, order-independent.

    `n_subsample_seeds` (default 1) averages the reward over that many
    distinct seeded subsamples (seeds `seed`, `seed+1`, ...) — a pure-code
    denoiser for imbalanced/high-variance tasks where a single subsample draw
    dominates the reward. Deterministic: the subsamples are fixed at build
    time, so the same state still always scores the same and the score cache
    stays exact.

    Input:  plan + training df; returns a closure rollout(state) -> float.

    Example:
        rollout = make_rollout_fn(plan, df)
        rollout({"model": "GBM"})           # -> 0.89   (bounded reward, r2 0.78)
        rollout({"model": "DoesNotExist"})  # -> 0.0    (failed config, no crash)
        # relational:
        rollout = make_rollout_fn(plan, main_df, aux={"aux": aux_df}, main_var="data")
    """
    if target is not None and target in df.columns and df[target].isna().any():
        # unlabeled rows: sklearn raises "Input y contains NaN" on every fit,
        # so a handful of missing targets would zero the WHOLE search silently
        # (classification only dodged it because value_counts drops NaN)
        df = df.dropna(subset=[target])
    strat = stratify and target is not None and target in df.columns
    n, floor = _profile_subsample_n(df, target, strat, min_rows=min_rows)
    seeds = [seed + i for i in range(max(1, int(n_subsample_seeds)))]
    if strat:
        smalls = [
            _stratified_subsample(df, target, n, s, floor=floor) for s in seeds
        ]
        minority = min(int(s[target].value_counts().min()) for s in smalls)
        n_splits = int(min(5, max(2, minority)))
    else:
        smalls = [df.sample(n=n, random_state=s) for s in seeds]
        n_splits = 5
    var_name = main_var or _single_var_name(plan)
    scorer = _resolve_scoring(scoring)
    total_timeout = timeout_s * len(smalls) if timeout_s else timeout_s

    def rollout(state: dict) -> float:
        try:
            apply_state(plan, state)
            with _time_limit(total_timeout):
                scores = []
                for small in smalls:
                    environment = {var_name: small, **(aux or {})}
                    cv_kwargs = {
                        "environment": environment,
                        **_cv_kwarg(stratify, seed, n_splits=n_splits),
                    }
                    if n_jobs != 1:
                        cv_kwargs["n_jobs"] = n_jobs
                    if scorer is not None:
                        cv_kwargs["scoring"] = scorer
                    result = plan.skb.cross_validate(**cv_kwargs)
                    folds = np.asarray(result["test_score"], dtype=float)
                    if np.all(np.isnan(folds)):
                        return 0.0  # every fold failed -> a failed config
                    scores.append(_fold_mean(folds))
            # map the raw score onto the bounded UCT scale; 0.0 stays reserved
            # for failures above, so a legitimate r2 of 0 reads as 0.5
            return _bounded_reward(sum(scores) / len(scores), scoring)
        except _RolloutTimeout:
            return (
                0.0  # BaseException — not covered by `except Exception` below
            )
        except Exception:
            return 0.0

    return rollout


def evaluate_full(
    plan,
    state: dict,
    df=None,
    aux: dict | None = None,
    main_var: str | None = None,
    scoring: str | None = None,
    stratify: bool = False,
) -> float:
    """Score a configuration on the FULL data — the final h(s), not a proxy.

    Unlike rollouts this propagates exceptions: by the time we evaluate a
    winner we want to know about failures, not paper over them. Pass `aux`
    (and `main_var`) for relational plans, as in `make_rollout_fn`. `scoring`
    is an sklearn scorer name (e.g. the task/report metric like
    "neg_root_mean_squared_error"); None keeps the estimator default.
    `stratify=True` for (imbalanced) classification — see `_cv_kwarg`.

    Example:
        evaluate_full(plan, {"model": "GBM"}, df)  # -> 0.88  (full-data mean CV score)
    """
    apply_state(plan, state)
    cv_kwargs: dict = dict(_cv_kwarg(stratify, seed=42))
    if df is not None:
        var_name = main_var or _single_var_name(plan)
        cv_kwargs["environment"] = {var_name: df, **(aux or {})}
    scorer = _resolve_scoring(scoring)
    if scorer is not None:
        cv_kwargs["scoring"] = scorer
    result = plan.skb.cross_validate(**cv_kwargs)
    return float(result["test_score"].mean())


# ---------------------------------------------------------------------------
# Track A — ablation over DAG choices
# ---------------------------------------------------------------------------


def run_ablation(
    plan, node_name: str, df, base_state: dict | None = None
) -> dict:
    """Score every option of one named choice node, all else held fixed.

    Returns {option: score}. This replaces MLE-STAR's code-block ablation:
    instead of commenting out code, we swap one DAG choice at a time.

    Example:
        run_ablation(plan, "model", df)  # -> {"GBM": 0.88, "RF": 0.87}
    """
    base_state = base_state if base_state is not None else get_state(plan)
    options = get_action_space(plan)[node_name]
    rollout = make_rollout_fn(plan, df)
    return {
        option: rollout({**base_state, node_name: option}) for option in options
    }


def pick_target_node(ablation_results: dict[str, dict]) -> str:
    """Deterministic stage targeting (Option 1): highest score variance wins.

    `ablation_results` maps node_name -> {option: score}. The stage whose
    options differ most is the most promising between-slice refinement target
    (`search_loop.run_search_loop` locks the rest via `target_key`). Targeting
    stays pure code by design — an LLM inside the search loop is the rejected
    anti-pattern, not a pending upgrade. The complementary *post-budget*
    targeting is `search_loop.run_search_loop`'s focused-refinement bonus
    phase (all single-edit neighbors of the incumbent), deterministic for the
    same reason.

    Example:
        pick_target_node({"model":   {"GBM": 0.88, "RF": 0.87},
                          "encoder": {"GapEncoder()": 0.70, "MinHashEncoder()": 0.90}})
        # -> "encoder"   (its options differ most -> highest variance)
    """

    def variance(scores: dict) -> float:
        values = list(scores.values())
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    return max(
        ablation_results, key=lambda name: variance(ablation_results[name])
    )
