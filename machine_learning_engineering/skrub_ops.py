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

from typing import Callable

import numpy as np
import skrub
from skrub._data_ops import _evaluation as _ev


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
    `apply_state`), then normalizes discrete defaults to the labels
    `get_action_space` uses (`None` -> `'None'`, etc.) while keeping true
    numeric defaults untouched. This is the "nothing chosen yet" baseline for
    staged construction. NOTE: resetting the plan is a deliberate side effect.

    Example:
        get_default_state(staged_plan)
        # -> {"scale": "None", "feature_eng": "None", "model": "LogReg"}
    """
    _reset_choices(plan)
    space = get_action_space(plan)
    out = {}
    for name, val in plan.skb.describe_defaults().items():
        opts = space.get(name, [])
        if val in opts:
            out[name] = val
        elif str(val) in opts:  # bare None -> 'None', etc.
            out[name] = str(val)
        else:
            out[name] = val  # numeric true default, not in the discretized opts
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
    assemble (relational) -> clean -> encode -> post-encoding stages -> model.
    See docs/pipeline-stages.md for the full taxonomy.

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

    # --- encode / vectorize ---
    enc_opts = spec.get("encoder_options")
    if enc_opts:
        vectorizer = skrub.TableVectorizer(
            high_cardinality=skrub.choose_from(enc_opts, name="encoder")
        )
    else:
        vectorizer = skrub.TableVectorizer()
    node = node.skb.apply(vectorizer)

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


def make_rollout_fn(
    plan,
    df,
    min_rows: int = 500,
    seed: int = 42,
    aux: dict | None = None,
    main_var: str | None = None,
    scoring: str | None = None,
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

    Input:  plan + training df; returns a closure rollout(state) -> float.

    Example:
        rollout = make_rollout_fn(plan, df)
        rollout({"model": "GBM"})           # -> 0.88   (mean CV test score)
        rollout({"model": "DoesNotExist"})  # -> 0.0    (failed config, no crash)
        # relational:
        rollout = make_rollout_fn(plan, main_df, aux={"aux": aux_df}, main_var="data")
    """
    n = min(len(df), max(min_rows, int(0.01 * len(df))))
    small = df.sample(n=n, random_state=seed)
    var_name = main_var or _single_var_name(plan)

    def rollout(state: dict) -> float:
        try:
            apply_state(plan, state)
            environment = {var_name: small, **(aux or {})}
            cv_kwargs = {"environment": environment}
            if scoring is not None:
                cv_kwargs["scoring"] = scoring
            result = plan.skb.cross_validate(**cv_kwargs)
            return float(result["test_score"].mean())
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
) -> float:
    """Score a configuration on the FULL data — the final h(s), not a proxy.

    Unlike rollouts this propagates exceptions: by the time we evaluate a
    winner we want to know about failures, not paper over them. Pass `aux`
    (and `main_var`) for relational plans, as in `make_rollout_fn`. `scoring`
    is an sklearn scorer name (e.g. the task/report metric like
    "neg_root_mean_squared_error"); None keeps the estimator default.

    Example:
        evaluate_full(plan, {"model": "GBM"}, df)  # -> 0.88  (full-data mean CV score)
    """
    apply_state(plan, state)
    cv_kwargs: dict = {}
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
    """Deterministic node-targeting stub: highest score variance wins.

    `ablation_results` maps node_name -> {option: score}. The node whose
    options differ most is the most promising refinement target. The LLM
    targeting upgrade (task A4) replaces this function.

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
