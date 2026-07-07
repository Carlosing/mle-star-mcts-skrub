"""skrub DataOps introspection and evaluation wrappers (Track A + Track B glue).

Everything that touches the skrub API lives here, so the MCTS engine
(mcts.py) stays pure and the extractor logic has one home.

Verified against skrub 0.9.0 (run tests/test_skrub_ops.py inside Docker).
The public .skb API has no structured param-grid accessor and
make_learner() takes no params argument, so this module uses the internal
`skrub._data_ops._evaluation` helpers (`choices`, `set_params`) — they are
what make_grid_search itself is built on. If a skrub upgrade breaks
anything, it breaks here and nowhere else.

Determinism requirements (UCT values never converge otherwise):
- estimators inside the plan must be seeded (random_state=42 in the fixture);
- rollout subsampling is done by passing a seeded df.sample() through
  cross_validate(environment=...) — skrub's own .skb.subsample(how="random")
  is NOT seeded in 0.9.
"""

import functools
import re
import signal
import threading
from contextlib import contextmanager
from typing import Callable

import numpy as np
import pandas as pd
import skrub
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
        #     "n_trees": [50, 100, 150, 200],     # choose_int(50, 200) discretized
        #     "lr": [0.01, 0.031, 0.097, 0.3],    # choose_float(.01,.3,log) geomspaced
        #     "model": ["GBM", "RF"]}
    """
    space: dict[str, list] = {}
    for name, (_, choice) in get_choices(plan).items():
        if hasattr(choice, "outcomes"):
            space[name] = _option_names(choice)
        else:
            spacing = np.geomspace if choice.log else np.linspace
            values = spacing(choice.low, choice.high, n_numeric_options)
            if choice.to_int:
                values = sorted({int(round(v)) for v in values})
            else:
                values = [float(v) for v in values]
            space[name] = list(values)
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
    children = graph["children"]  # {(parent_id, outcome_idx): [child_ids], None: [...]}

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
    labeled["skip"] = skrub.SelectCols([])  # replaces None, keeps first position
    return labeled


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
      'aux_mean' etc. rather than the AggJoiner repr (which embeds the table).

    Example (minimal, single-table):
        spec = {"encoder_options": [skrub.GapEncoder(), skrub.MinHashEncoder()],
                "model": {"GBM": GradientBoostingClassifier(),
                          "RF": RandomForestClassifier()}}
        plan = build_staged_plan(spec, df)        # df has a "target" column
        get_action_space(plan)
        # -> {"encoder": ["GapEncoder()", "MinHashEncoder()"], "model": ["GBM", "RF"]}
    """
    data = skrub.var("data", df)
    aux_vars = {name: skrub.var(name, adf) for name, adf in (aux_tables or {}).items()}
    X = data.drop(columns=target).skb.mark_as_X()
    y = data[target].skb.mark_as_y()
    node = X

    # --- assemble (relational): aggregate-join auxiliary tables ---
    if spec.get("assemble"):
        joiners = {"skip": None}
        for cfg in spec["assemble"]:
            label = cfg.get("name") or f"{cfg['table']}_{'_'.join(cfg['operations'])}"
            joiners[label] = skrub.AggJoiner(
                aux_vars[cfg["table"]],
                cfg["operations"],
                key=cfg.get("key"),
                main_key=cfg.get("main_key"),
                aux_key=cfg.get("aux_key"),
                cols=cfg.get("cols"),
            )
        node = node.skb.apply(skrub.choose_from(joiners, name="assemble"))

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
                c for c in group.get("cols", []) if c in df.columns and c != target
            ]
            if not cols:
                continue  # nothing valid to scope over -> group dropped
            node = _apply_scope_group(node, group, cols)
        return node

    node = _scope_pass(node, "pre_encode")

    # --- encode / vectorize ---
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
            skrub.choose_from(_skip_first(list(stage["options"])), name=stage["name"])
        )

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

    return {"cv": StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)}


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
        df[df[target] == c].sample(n=quotas[c], random_state=seed) for c in labels
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


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
    timeout_s: float | None = 45.0,
) -> Callable[[dict], float]:
    """Build a rollout_fn(state) -> float for mcts.mcts_search.

    Evaluates a configuration on an adaptive seeded subsample of the MAIN
    table: n = max(min_rows, 1% of the data), capped at the full table. The
    same state always scores the same. Failed configs return 0.0.

    Works for both flat (complete) and staged (partial) states because
    `apply_state` resets to defaults before applying — a partial state simply
    leaves the undecided stages at their defaults.

    For relational plans, pass `aux={var_name: dataframe}` (auxiliary tables
    are joined whole, not subsampled) and `main_var` to name the main table's
    var (defaults to the plan's single var).

    `scoring` is an sklearn scorer name forwarded to cross_validate. It must be
    **higher-is-better** (MCTS maximizes) — use a bounded one like "r2" or
    "accuracy" so rewards stay on the same scale as the UCT exploration term.
    None keeps the estimator's default .score() (R2 for regressors).

    `stratify=True` (pass for classification, especially imbalanced targets)
    uses a seeded `StratifiedKFold` instead of skrub's plain-`KFold` default —
    see `_cv_kwarg` for why this matters. Passing `target` as well upgrades
    the *subsample* to a stratified one with a minority floor
    (`_stratified_subsample`) and shrinks `n_splits` when the subsampled
    minority is smaller than the fold count; without `target` the subsample
    stays the plain seeded `df.sample` of before. Per-fold NaNs (degenerate
    folds) are skipped via `_fold_mean` instead of failing the config.

    `timeout_s` (default 45s) is a per-config wall-clock cap: a config whose CV
    exceeds it scores 0.0 like any other failure (via `_time_limit`), so a
    free-form hyperparameter range that makes a single fit pathologically slow
    can't stall the search. `None`/0 disables the cap. Best-effort — see
    `_time_limit` (main-thread + SIGALRM only; can't preempt a non-yielding C
    call).

    Input:  plan + training df; returns a closure rollout(state) -> float.

    Example:
        rollout = make_rollout_fn(plan, df)
        rollout({"model": "GBM"})           # -> 0.88   (mean CV test score)
        rollout({"model": "DoesNotExist"})  # -> 0.0    (failed config, no crash)
        # relational:
        rollout = make_rollout_fn(plan, main_df, aux={"aux": aux_df}, main_var="data")
    """
    n = min(len(df), max(min_rows, int(0.01 * len(df))))
    if stratify and target is not None and target in df.columns:
        small = _stratified_subsample(df, target, n, seed)
        minority = int(small[target].value_counts().min())
        n_splits = int(min(5, max(2, minority)))
    else:
        small = df.sample(n=n, random_state=seed)
        n_splits = 5
    var_name = main_var or _single_var_name(plan)

    def rollout(state: dict) -> float:
        try:
            apply_state(plan, state)
            environment = {var_name: small, **(aux or {})}
            cv_kwargs = {
                "environment": environment,
                **_cv_kwarg(stratify, seed, n_splits=n_splits),
            }
            if scoring is not None:
                cv_kwargs["scoring"] = scoring
            with _time_limit(timeout_s):
                result = plan.skb.cross_validate(**cv_kwargs)
            return _fold_mean(result["test_score"])
        except _RolloutTimeout:
            return 0.0  # BaseException — not covered by `except Exception` below
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
    if scoring is not None:
        cv_kwargs["scoring"] = scoring
    result = plan.skb.cross_validate(**cv_kwargs)
    return float(result["test_score"].mean())


# ---------------------------------------------------------------------------
# Track A — ablation over DAG choices
# ---------------------------------------------------------------------------


def run_ablation(plan, node_name: str, df, base_state: dict | None = None) -> dict:
    """Score every option of one named choice node, all else held fixed.

    Returns {option: score}. This replaces MLE-STAR's code-block ablation:
    instead of commenting out code, we swap one DAG choice at a time.

    Example:
        run_ablation(plan, "model", df)  # -> {"GBM": 0.88, "RF": 0.87}
    """
    base_state = base_state if base_state is not None else get_state(plan)
    options = get_action_space(plan)[node_name]
    rollout = make_rollout_fn(plan, df)
    return {option: rollout({**base_state, node_name: option}) for option in options}


def pick_target_node(ablation_results: dict[str, dict]) -> str:
    """Deterministic stage targeting (Option 1): highest score variance wins.

    `ablation_results` maps node_name -> {option: score}. The stage whose
    options differ most is the most promising between-slice refinement target
    (`search_loop.run_search_loop` locks the rest via `target_key`). Targeting
    stays pure code by design — an LLM inside the search loop is the rejected
    anti-pattern, not a pending upgrade. The complementary *post-budget*
    targeting that focuses on the incumbent model's untested hyperparameters is
    `search_loop._incumbent_hp_targets` (the HP-refinement bonus phase), which
    is deterministic for the same reason.

    Example:
        pick_target_node({"model":   {"GBM": 0.88, "RF": 0.87},
                          "encoder": {"GapEncoder()": 0.70, "MinHashEncoder()": 0.90}})
        # -> "encoder"   (its options differ most -> highest variance)
    """

    def variance(scores: dict) -> float:
        values = list(scores.values())
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    return max(ablation_results, key=lambda name: variance(ablation_results[name]))
