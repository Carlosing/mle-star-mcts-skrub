# Tabular pipeline stages (skrub) — the MCTS search space

The constructive MCTS searches over the *stages* of a tabular/relational
pipeline. This document is the canonical stage taxonomy: what each stage is,
which skrub API provides it, and whether `build_staged_plan`
([skrub_ops.py](../machine_learning_engineering/skrub_ops.py)) builds it today.
It is derived from the skrub 0.9 API reference and verified against the
installed library.

## Stage order

```
assemble (relational) → clean → scope (scoped encodings) → encode → feature-eng/scale → select → model → hyperparameters → post-process
```

A plan is built top-down in this order; MCTS decides one stage at a time
(constructive topology). Every stage except `encode` and `model` is optional,
with **skip as its default outcome**, so an undecided stage means
"not yet enriched."

## The stages

| # | Stage | skrub API | In `build_staged_plan`? | Notes |
|---|-------|-----------|:---:|-------|
| 0 | **Assemble** (relational) | `AggJoiner`, `MultiAggJoiner`, `AggTarget`, `Joiner`, `InterpolationJoiner`, `fuzzy_join` | ✅ (`AggJoiner`) | Build the feature table from multiple tables. Highest-leverage stage for relational data — the skrub differentiator vs. flat-table AutoML |
| 1 | **Clean / coerce** | `Cleaner`, `DropUninformative`, `deduplicate`, `ToDatetime`/`ToFloat`, `SquashingScaler` | ✅ (`clean_options`) | Parse nulls/dates, drop bad columns, dedupe categories, robust scaling |
| 2 | **Encode / vectorize** | `TableVectorizer` + `GapEncoder`/`MinHashEncoder`/`StringEncoder`/`TextEncoder`/`SimilarityEncoder`/`DatetimeEncoder`/`ToCategorical` | ✅ (`encoder_options`) | Per-column-type encoding; the encoder is a searchable choice |
| 3 | **Scope** | `.skb.apply(estimator, cols=<selector>)`; selectors: `regex`, `cols`, `numeric`, `cardinality_below`, … | ✅ (`scoped_encodings`) | Apply a *searchable* encoder to an explicit column subset (before the TableVectorizer, which still handles the rest). Column names are validated at resolve/build time; the runtime selector is a union of exact-match regexes, so a column dropped upstream degrades the group to a no-op (`skrub_ops._scope_selector` — `selectors.cols()` would raise) |
| 4 | **Feature-eng / scale** | `apply(PolynomialFeatures/PCA/…)`, `DatetimeEncoder`, `SquashingScaler`/`StandardScaler`, `apply_func`, `deferred` | ✅ (`stages`) | skrub has no large FE library — FE = `apply` any sklearn transformer + custom funcs |
| 5 | **Select features** | `SelectCols`, `DropCols`, `DropUninformative`, sklearn selectors | ✅ (`stages`) | Optional feature selection |
| 6 | **Model** | `choose_from` over estimators via `apply` | ✅ (`model`) | Required; has a real default so partial pipelines run |
| 7 | **Hyperparameters** | `choose_int`/`choose_float`/`choose_from` | ✅ (`spec_resolver`) | Per-operator HP ranges in the JSON plan become nested `choose_*` nodes that MCTS searches. CASH note: HPs of a non-selected model are inactive search dims; conditional (model-gated) nesting is the remaining refinement |
| 8 | **Post-process / ensemble** | `concat` (stacking), `if_else`, `match`, `apply_func` | ❌ future | Combine feature sets / predictions; conditional branches |

Not pipeline stages, but relevant: `TableReport` and `column_associations` are
**EDA** utilities — useful as *input to the LLM planner* (à la SELA's EDA
stage), not as runtime transforms.

## Spec schema (the LLM "rich plan" hand-off)

`build_staged_plan(spec, df, aux_tables=...)` consumes this shape (operators are
real estimator instances; translating LLM text → instances is a separate
concern):

```python
spec = {
    # 0. assemble (needs aux_tables={name: df}); a 'skip' option is auto-added
    "assemble": [
        {"name": "aux_mean", "table": "aux", "operations": ["mean"],
         "key": "id", "cols": ["v"]},
    ],
    # 1. clean
    "clean_options": [None, Cleaner()],
    # 3. scope — searchable encoder on an explicit column subset (skip default)
    "scoped_encodings": [
        {"name": "title_enc", "cols": ["job_title"],
         "options": [GapEncoder(), MinHashEncoder()]},
    ],
    # 2. encode (the TableVectorizer still handles all unscoped columns)
    "encoder_options": [GapEncoder(), MinHashEncoder()],
    # 4–5. post-encoding numeric stages
    "stages": [
        {"name": "scale",       "options": [None, StandardScaler()]},
        {"name": "feature_eng", "options": [None, PolynomialFeatures(2)]},
    ],
    # 6. model (required)
    "model": {"GBM": ..., "RF": ..., "LogReg": ...},
}
```

The assemble stage uses a **labeled dict** `choose_from` internally so options
read as `aux_mean` rather than the `AggJoiner` repr (which embeds the whole aux
table). Every other optional stage uses `None` = skip as its default.

## Cautions

- **`AggTarget` aggregates the target** → leakage source. It must be computed
  inside the CV fold, which is exactly why skrub's `mark_as_X`/`mark_as_y` (and
  the brief's `A_leakage` agent) exist. Treat target-based aggregation as a
  guarded, leakage-checked action.
- **MCTS is weak on *continuous* hyperparameters.** Mosaic and auto-sklearn
  couple structural search with Bayesian optimization for the HP level; we
  discretize `choose_int`/`choose_float` instead. A BO hand-off for the HP
  stage is the established upgrade path if discretization proves coarse.
- **Relational rollouts:** auxiliary tables are joined whole (not subsampled);
  only the main table is subsampled. Pass `aux=` and `main_var=` to
  `make_rollout_fn`/`evaluate_full`.

## Status

Built and tested (see [test_staged.py](../tests/test_staged.py),
[test_scope_stage.py](../tests/test_scope_stage.py),
[test_spec_resolver.py](../tests/test_spec_resolver.py)): assemble
(`AggJoiner`), clean, **scope** (`scoped_encodings`), encode, scale,
feature-eng, model, **hyperparameter search** (per-operator `choose_*` from
the JSON plan), and **conditional (model-gated) HP nesting**. The relational
assemble stage is demonstrated to lift a near-chance score when the target
depends on an auxiliary-table aggregate, and is wired end-to-end (multi-table
`load_task`, aux digests, `plan_author` assemble configs — see
[test_relational_pipeline.py](../tests/test_relational_pipeline.py)). Future:
the `post-process` (stacking) stage.
