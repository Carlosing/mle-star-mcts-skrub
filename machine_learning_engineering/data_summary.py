"""Compact, token-bounded EDA digest of a dataframe for the LLM planner.

The full dataframe never goes to the model — only this digest does. It is the
SELA-style hand-off: enough structure (dtypes, missingness, cardinality, example
values, a few head rows) for the analyst to reason about encoders / models,
without spending tokens on the whole table. Pandas-only and deterministic.
"""

import pandas as pd


def infer_task_type(df: pd.DataFrame, target: str) -> str:
    """Heuristic: numeric target with many distinct values -> regression."""
    s = df[target]
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def _fmt(x) -> str:
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)


def make_data_summary(
    df: pd.DataFrame,
    target: str,
    n_example_values: int = 5,
    n_head_rows: int = 5,
) -> str:
    """Return a compact text digest of ``df`` to feed the analyst agent.

    Includes shape, target, inferred task type, and per-column dtype / missing
    rate / cardinality / examples (for categoricals) or min-max-mean (numerics),
    plus the first few rows.
    """
    n_rows, n_cols = df.shape
    lines = [
        f"Dataset: {n_rows} rows x {n_cols} columns. Target column: {target!r}.",
    ]
    if target in df.columns:
        lines.append(f"Inferred task type: {infer_task_type(df, target)}.")
    lines.append("")
    lines.append("Columns:")
    for col in df.columns:
        s = df[col]
        miss = f"{s.isna().mean() * 100:.1f}%"
        card = int(s.nunique(dropna=True))
        tag = " (TARGET)" if col == target else ""
        if pd.api.types.is_numeric_dtype(s):
            desc = f"min={_fmt(s.min())} max={_fmt(s.max())} mean={_fmt(s.mean())}"
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
    return "\n".join(lines)
