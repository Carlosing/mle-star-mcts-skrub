"""Stored LLM answer exercising ARBITRARY (non-curated) hyperparameter tuning.

Authored by acting as the plan_author LLM against the *updated* instruction
(which now lets the model tune any constructor hyperparameter of an allow-listed
class, with ranges used as given, while warning against training-time-blowing
upper bounds). Offline tests resolve this to confirm free-form HP acceptance and
the per-param safety nets, with no API call — same practice as
``california_agent_io`` / ``open_payments_agent_io``.

Coverage baked into the plan below:
- HistGradientBoostingRegressor — a curated HP (learning_rate, clipped) AND an
  arbitrary-but-valid one (max_leaf_nodes, free-form).
- RandomForestRegressor — a curated HP (n_estimators) + an arbitrary valid one
  (min_impurity_decrease) + an INVALID one (learning_rate: RF has no such arg)
  that must be dropped WITHOUT dropping the RF operator.
- Lasso — NOT in the REGISTRY at all: a fully free-form HP (alpha) plus a
  random_state tuning attempt that must be refused (determinism invariant).
- PCA feature-eng stage — a free-form n_components on a non-curated transformer.
Ranges are kept modest (fast to fit) as the updated prompt instructs.
"""

# plan_author -> state["skrub_spec_raw"]  (dotted paths + arbitrary HPs; fenced)
SKRUB_SPEC_RAW = """\
```json
{
  "stages": [
    {"name": "feature_eng", "options": ["skip",
      {"name": "sklearn.decomposition.PCA", "params": {"n_components": {"int": [2, 6]}}}]}
  ],
  "model": [
    {"name": "sklearn.ensemble.HistGradientBoostingRegressor", "params": {
       "learning_rate": {"float": [0.01, 0.3], "log": true},
       "max_leaf_nodes": {"int": [15, 63]}}},
    {"name": "sklearn.ensemble.RandomForestRegressor", "params": {
       "n_estimators": {"int": [100, 400]},
       "min_impurity_decrease": {"float": [0.0, 0.05]},
       "learning_rate": {"float": [0.01, 0.3]}}},
    {"name": "sklearn.linear_model.Lasso", "params": {
       "alpha": {"float": [0.0001, 10.0], "log": true},
       "random_state": {"int": [0, 999]}}}
  ]
}
```
"""
