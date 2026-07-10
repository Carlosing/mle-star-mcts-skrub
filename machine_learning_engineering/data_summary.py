"""Compact, token-bounded EDA digest of a dataframe for the LLM planner.

The full dataframe never goes to the model — only this digest does. It is the
SELA-style hand-off: enough structure (dtypes, missingness, cardinality, example
values, a few head rows) for the analyst to reason about encoders / models,
without spending tokens on the whole table. Pandas-only and deterministic.
"""

import pandas as pd


def infer_task_type(
    df: pd.DataFrame, target: str, metric: str | None = None
) -> str:
    """Heuristic: numeric target with many distinct values -> regression.

    A ``metric`` that implies a task type overrules the heuristic *when the
    target's dtype allows it*: a numeric target with few distinct values (a
    0.5-5.0 star rating, a Likert scale) is regression if the task reports
    RMSE, even though the cardinality rule would call it classification —
    otherwise the whole run silently searches `accuracy` over Classifiers and
    reports RMSE. A non-numeric target is always classification, whatever the
    metric says.

    Example:
        infer_task_type(df, "rating")          # -> "classification"  (10 values)
        infer_task_type(df, "rating", "rmse")  # -> "regression"
        infer_task_type(df, "status", "rmse")  # -> "classification"  (strings)
    """
    from machine_learning_engineering import metrics as _metrics

    s = df[target]
    numeric = pd.api.types.is_numeric_dtype(s)
    declared = _metrics.metric_task_type(metric)
    if declared is not None and (numeric or declared == "classification"):
        return declared
    if numeric and s.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def _fmt(x) -> str:
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)


def _join_key_candidates(
    main: pd.DataFrame, aux: pd.DataFrame, max_pairs: int = 3
) -> list[tuple[str, str, float]]:
    """Rank (main_col, aux_col) pairs by how plausibly they join the tables.

    A pair qualifies if the columns share a dtype family and the aux values
    overlap the main values (sampled, so large tables stay cheap). Same-name
    pairs get a small bonus so the natural key sorts first.

    Example:
        _join_key_candidates(main, aux)  # -> [("id", "id", 1.0)]
    """
    main_vals = {
        c: set(main[c].dropna().unique()[:5000])
        for c in main.columns
        if main[c].nunique(dropna=True) > 1
    }
    pairs = []
    for ac in aux.columns:
        a_vals = set(aux[ac].dropna().unique()[:5000])
        if not a_vals:
            continue
        for mc, m_vals in main_vals.items():
            if pd.api.types.is_numeric_dtype(
                main[mc]
            ) != pd.api.types.is_numeric_dtype(aux[ac]):
                continue
            overlap = len(a_vals & m_vals) / len(a_vals)
            if overlap >= 0.5 or (mc == ac and overlap > 0):
                pairs.append(
                    (mc, ac, round(overlap + (0.01 if mc == ac else 0), 3))
                )
    pairs.sort(key=lambda p: -p[2])
    return [(mc, ac, min(ov, 1.0)) for mc, ac, ov in pairs[:max_pairs]]


def _aux_table_digest(
    name: str, adf: pd.DataFrame, main: pd.DataFrame
) -> list[str]:
    """Compact per-aux-table digest lines (schema + join-key candidates)."""
    lines = [
        f"Auxiliary table {name!r}: {adf.shape[0]} rows x {adf.shape[1]} columns."
    ]
    lines.append("  Columns:")
    for col in adf.columns:
        s = adf[col]
        lines.append(
            f"    - {col}: dtype={s.dtype}, cardinality={int(s.nunique(dropna=True))}"
        )
    candidates = _join_key_candidates(main, adf)
    if candidates:
        lines.append(
            "  Join-key candidates (main column <-> aux column, overlap):"
        )
        for mc, ac, ov in candidates:
            lines.append(f"    - {mc} <-> {ac} (overlap={ov:.0%})")
    return lines


def make_data_summary(
    df: pd.DataFrame,
    target: str,
    n_example_values: int = 5,
    n_head_rows: int = 5,
    aux_tables: dict[str, pd.DataFrame] | None = None,
    metric: str | None = None,
) -> str:
    """Return a compact text digest of ``df`` to feed the analyst agent.

    Includes shape, target, inferred task type, and per-column dtype / missing
    rate / cardinality / examples (for categoricals) or min-max-mean (numerics),
    plus the first few rows. With ``aux_tables={name: df}``, each auxiliary
    table gets a schema digest plus join-key candidates, so the planner can
    propose aggregate joins (the ``assemble`` stage) using real names.

    ``metric`` (the task's declared metric) disambiguates the task type the
    digest reports, so the planner is never told "classification" for a
    low-cardinality numeric target that the pipeline searches as regression.
    """
    n_rows, n_cols = df.shape
    lines = [
        f"Dataset: {n_rows} rows x {n_cols} columns. Target column: {target!r}.",
    ]
    if target in df.columns:
        lines.append(
            f"Inferred task type: {infer_task_type(df, target, metric)}."
        )
    lines.append("")
    lines.append("Columns:")
    for col in df.columns:
        s = df[col]
        miss = f"{s.isna().mean() * 100:.1f}%"
        card = int(s.nunique(dropna=True))
        tag = " (TARGET)" if col == target else ""
        if pd.api.types.is_numeric_dtype(s):
            desc = (
                f"min={_fmt(s.min())} max={_fmt(s.max())} mean={_fmt(s.mean())}"
            )
        else:
            vals = list(s.dropna().unique()[:n_example_values])
            desc = "examples=" + ", ".join(repr(str(v)) for v in vals)
        lines.append(
            f"  - {col}{tag}: dtype={s.dtype}, missing={miss}, "
            f"cardinality={card}, {desc}"
        )
    lines.append("")
    lines.append(f"First {n_head_rows} rows:")
    lines.append(df.head(n_head_rows).to_string(index=False))
    for name, adf in (aux_tables or {}).items():
        lines.append("")
        lines.extend(_aux_table_digest(name, adf, df))
    return "\n".join(lines)
