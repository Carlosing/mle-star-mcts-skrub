"""Realistic canned agent outputs for the open-payments classification task.

Mirrors california_agent_io but for a binary-classification, categorical-rich
dataset (status: allowed/disallowed). Encoder is pinned to MinHashEncoder to keep
the offline test fast (GapEncoder is slow on full-data eval).
"""

# data_analyst -> state["dataset_analysis"]
DATASET_ANALYSIS = """\
Task: binary classification, target 'status' (allowed vs disallowed payment),
scored by accuracy.

Column types:
- High-cardinality categorical columns (company / physician / drug identifiers)
  dominate — a strong case for MinHashEncoder (fast) or GapEncoder.
- A few low-cardinality categoricals and numeric amounts.

Options worth searching per stage:
- clean: skip vs skrub.Cleaner.
- encode: MinHashEncoder on the high-cardinality columns.
- model: HistGradientBoostingClassifier (strong tabular default),
  RandomForestClassifier, LogisticRegression (linear baseline on encoded cols).
"""

# plan_author -> state["skrub_spec_raw"]  (classification; dotted paths; valid JSON)
SKRUB_SPEC_RAW = """\
{
  "clean_options": ["skip", "skrub.Cleaner"],
  "encoder_options": ["skrub.MinHashEncoder"],
  "model": [
    {"name": "sklearn.ensemble.HistGradientBoostingClassifier", 
      "params": {"learning_rate": {"float": [0.01, 0.3], "log": true},"max_iter": {"int": [100, 300]}}},
    {"name": "sklearn.ensemble.RandomForestClassifier", "params": {"n_estimators": {"int": [100, 300]}}},
    "sklearn.linear_model.LogisticRegression"
  ]
}
"""

# Convenience: the two model turns in call order, for a scripted fake model.
RESPONSES = [DATASET_ANALYSIS, SKRUB_SPEC_RAW]
