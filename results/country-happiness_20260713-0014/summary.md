# Run: country-happiness  (regression, metric=root_mean_squared_error)
model: openai/qwen3.5-397b-a17b  |  budget: 60  |  fallback spec: False
LLM calls: 4  |  tokens: 41,614 (prompt 13,583 + completion 28,031)

## ⚠ Plan quality warnings
- **dropped sections** (present in the plan but malformed, so resolved to nothing and not searched): ['scoped_encodings']

## Task
# Task

Predict the happiness_score.

This is a relational task: aggregate the auxiliary table(s) per main row before modelling (aux_gdp.csv on main.Country = gdp.Country Name; aux_life_expectancy.csv on main.Country = life_expectancy.Country Name; aux_legal_rights.csv on main.Country = legal_rights.Country Name).

# Metric

root_mean_squared_error

# Dataset

Regression task (117 training rows, 1 feature columns). train.csv has the features and the 'happiness_score' column; test.csv has the features only.

Relational, credit-fraud shaped: the main table is reduced to (Country, happiness_score) — every leaky decomposition column was dropped — so ALL signal must arrive through aggregate joins on three World-Bank aux tables. Only 146 rows, and the join is by country NAME across differently-named key columns (Country <-> 'Country Name'), so misses are silent.

## 1. Data report (analyst output, sent to the planner agent)
# Task Analysis & Pipeline Design Options

## 1. Task Type & Target
- **Task:** Regression.
- **Target Column:** `happiness_score`.
- **Distribution:** Continuous float (min=2404, max=7821, mean=5548). Note the magnitude is atypical for standard happiness indices (usually 0–10); scaling the target may improve model convergence depending on the algorithm.
- **Class Balance:** N/A (Regression).

## 2. Column Profile & Cardinality
- **`Country` (Main):** High-cardinality categorical (cardinality=117, rows=117). Effectively a unique identifier per row in the main table. **Do not one-hot encode**; it will cause perfect multicollinearity/overfitting. Use strictly as a join key.
- **`happiness_score` (Target):** Numeric.
- **Auxiliary Features (Post-Join):**
  - `GDP per capita`: Numeric (USD), likely skewed distribution.
  - `Strength of legal rights index`: Numeric (0–12), ordinal-like.
  - `Life expectancy`: Numeric (years), likely normally distributed.
- **Missingness:** Main table has 0% missing. However, auxiliary tables have 89 rows vs. 117 main rows. A join will introduce ~24% missingness for rows without auxiliary data.

## 3. Preprocessing, Encoders & Models
- **Cleaning Steps:**
  - **Join Strategy:** Search between `Left Join` (keep all 117 rows, impute missing aux data) vs. `Inner Join` (drop ~28 rows, lose data but avoid imputation bias). Given the tiny sample size, `Left Join` + Imputation is preferred to retain data.
  - **Imputation:** Search `SimpleImputer(strategy='mean')` vs. `IterativeImputer` vs. `KNNImputer`.
  - **Scaling:** Search `StandardScaler` vs. `RobustScaler` (for GDP outliers) on numeric features. Consider `TargetTransformer` (log or standard) on `happiness_score`.
- **Encoders:**
  - No categorical encoders needed for `Country` (drop after join).
  - If any residual categoricals exist after join, use `MinHashEncoder` (high cardinality) but unlikely here.
- **Candidate Model Families:**
  - **Linear Models:** `Ridge`, `Lasso`, `ElasticNet` (Strong regularization required due to n=117).
  - **Tree Ensembles:** `RandomForestRegressor` (limit `max_depth` to 3–5, `min_samples_leaf` ≥ 5), `XGBoost` (very low `max_depth`, high `reg_alpha`).
  - **Baseline:** `DummyRegressor(strategy='mean')`.
  - *Note:* Deep learning or complex boosting is likely to overfit given n=117.

## 4. Specific Column Operators
- **`Country` → `TableJoiner`:** Must be joined to auxiliary tables before feature extraction. Drop the `Country` column after joining to prevent leakage/overfitting.
- **Numeric Features → `Imputer` + `Scaler`:** Pipeline must handle missing values introduced by the join before scaling.
- **`GDP per capita` → `LogTransformer`:** GDP is often log-normally distributed; search log-transform vs. raw scale.

## 5. Relational & Auxiliary Opportunities
- **Join Keys:** `Country` (main) ↔ `Country Name` (aux).
- **Tables to Join:**
  1.  `gdp`: Highly predictive of happiness.
  2.  `life_expectancy`: Highly predictive of happiness.
  3.  `legal_rights`: Potentially weaker signal, but useful for regularization.
- **Aggregations:** Not applicable (1-to-1 relationship after join).
- **Feature Engineering:** Consider interaction terms post-join (e.g., `GDP * Life expectancy`) but limit complexity due to small sample size.

## 6. Critical Constraints for Planner
- **Sample Size:** n=117 is extremely small. Cross-validation must be `LeaveOneOut` or `KFold(n_splits=5)` with shuffling.
- **Overfitting Risk:** High. Pipeline search should prioritize regularization strength and feature selection (e.g., `SelectKBest`) over model complexity.
- **Data Leakage:** Ensure `Country` is not encoded as a feature; it is only a key.

## 2. Generated plan (planner output)
```json
{
  "assemble": [
    {
      "name": "join_gdp",
      "table": "gdp",
      "main_key": "Country",
      "aux_key": "Country Name",
      "operations": ["mean"],
      "cols": ["GDP per capita (current US$)"]
    },
    {
      "name": "join_life_expectancy",
      "table": "life_expectancy",
      "main_key": "Country",
      "aux_key": "Country Name",
      "operations": ["mean"],
      "cols": ["Life expectancy at birth, total (years)"]
    },
    {
      "name": "join_legal_rights",
      "table": "legal_rights",
      "main_key": "Country",
      "aux_key": "Country Name",
      "operations": ["mean"],
      "cols": ["Strength of legal rights index (0=weak to 12=strong)"]
    }
  ],
  "cleaner": {
    "params": {
      "drop_if_constant": {"choice": [false, true]},
      "drop_if_unique": {"choice": [false, true]},
      "parse_numbers": {"choice": [false, true]}
    }
  },
  "vectorizer": {
    "slots": {
      "high_cardinality": ["skrub.MinHashEncoder", "skrub.GapEncoder"],
      "low_cardinality": ["sklearn.preprocessing.OneHotEncoder"]
    }
  },
  "scoped_encodings": [
    {
      "name": "gdp_log_transform",
      "cols": ["GDP per capita (current US$)"],
      "options": [
        "skip",
        {"name": "sklearn.preprocessing.PowerTransformer", "params": {"standardize": {"choice": [false, true]}, "method": {"choice": ["yeo-johnson", "box-cox"]}}}
      ],
      "position": "post_encode",
      "additive": false
    }
  ],
  "stages": [
    {
      "name": "impute",
      "options": [
        "skip",
        {"name": "sklearn.impute.SimpleImputer", "params": {"strategy": {"choice": ["mean", "median"]}}},
        {"name": "sklearn.impute.KNNImputer", "params": {"n_neighbors": {"int": [3, 7]}}}
      ]
    },
    {
      "name": "scale",
      "options": [
        "skip",
        "sklearn.preprocessing.StandardScaler",
        "sklearn.preprocessing.RobustScaler",
        {"name": "sklearn.preprocessing.PowerTransformer", "params": {"standardize": {"choice": [false, true]}}}
      ]
    },
    {
      "name": "feature_eng",
      "options": [
        "skip",
        {"name": "sklearn.preprocessing.PolynomialFeatures", "params": {"degree": {"int": [2, 3]}, "interaction_only": {"choice": [false, true]}}}
      ]
    }
  ],
  "model": [
    {
      "name": "sklearn.linear_model.Ridge",
      "prior": 0.8,
      "params": {
        "alpha": {"float": [0.001, 1000.0], "log": true},
        "fit_intercept": {"choice": [true, false]}
      }
    },
    {
      "name": "sklearn.ensemble.HistGradientBoostingRegressor",
      "prior": 0.6,
      "params": {
        "learning_rate": {"float": [0.01, 0.3], "log": true},
        "max_iter": {"int": [100, 400]},
        "max_depth": {"int": [2, 8]},
        "l2_regularization": {"float": [0.0, 1.0]},
        "min_samples_leaf": {"int": [5, 20]}
      }
    },
    {
      "name": "sklearn.ensemble.RandomForestRegressor",
      "prior": 0.5,
      "params": {
        "n_estimators": {"int": [100, 300]},
        "max_depth": {"int": [2, 6]},
        "min_samples_leaf": {"int": [5, 15]},
        "max_features": {"choice": ["sqrt", "log2", 1.0]}
      }
    },
    {
      "name": "xgboost.XGBRegressor",
      "prior": 0.5,
      "params": {
        "n_estimators": {"int": [100, 300]},
        "learning_rate": {"float": [0.01, 0.3], "log": true},
        "max_depth": {"int": [2, 6]},
        "reg_lambda": {"float": [0.1, 10.0], "log": true},
        "subsample": {"float": [0.5, 1.0]}
      }
    }
  ]
}
```

## 3. Best configuration + score (MCTS incumbent)
```json
{
  "assemble": "all_aggregates",
  "cleaner__Cleaner__drop_if_constant": "False",
  "cleaner__Cleaner__drop_if_unique": "True",
  "cleaner__Cleaner__parse_numbers": "False",
  "vectorizer__high_cardinality": "MinHashEncoder()",
  "impute": "None",
  "scale": "QuantileTransformer(n_quantiles=choose_int(10, 100, name='scale__Quant...__n_quantiles'),\n                    output_distribution=choose_from(['normal', 'uniform'], name='scale__Quant..._distribution'),\n                    random_state=42)",
  "feature_eng": "None",
  "model": "HistGradientBoostingRegressor"
}
```
- search reward (r2, scale=1/(2 - r2)): 0.6808935363547687
- report metric (neg_root_mean_squared_error): -915.801719379763
- top-3 ensemble (neg_root_mean_squared_error): -753.8692 vs individuals ['-753.8692', '-753.8692', '-753.8692']
- targeted stage (Option 1): model
- focused-refinement bonus phase edited: ['assemble', 'cleaner__Cleaner__drop_if_constant', 'cleaner__Cleaner__drop_if_unique', 'cleaner__Cleaner__parse_numbers', 'feature_eng', 'feature_eng__PCA__n_components', 'feature_eng__PCA__svd_solver', 'feature_eng__PolynomialFeatures__degree', 'feature_eng__PolynomialFeatures__interaction_only', 'feature_eng__SplineTransformer__degree', 'feature_eng__SplineTransformer__include_bias', 'feature_eng__SplineTransformer__n_knots', 'impute', 'impute__KNNImputer__n_neighbors', 'impute__SimpleImputer__strategy', 'model', 'model__BayesianRidge__alpha_1', 'model__BayesianRidge__alpha_2', 'model__BayesianRidge__fit_intercept', 'model__BayesianRidge__lambda_1', 'model__BayesianRidge__lambda_2', 'model__ElasticNet__alpha', 'model__ElasticNet__fit_intercept', 'model__ElasticNet__l1_ratio', 'model__ExtraTreesRegressor__max_depth', 'model__ExtraTreesRegressor__max_features', 'model__ExtraTreesRegressor__min_samples_leaf', 'model__ExtraTreesRegressor__n_estimators', 'model__GaussianProcessRegressor__alpha', 'model__GaussianProcessRegressor__n_restarts_optimizer', 'model__LGBMRegressor__learning_rate', 'model__LGBMRegressor__n_estimators', 'model__LGBMRegressor__num_leaves', 'model__LGBMRegressor__reg_lambda', 'model__LGBMRegressor__subsample', 'model__Ridge__alpha', 'model__Ridge__fit_intercept', 'model__SVR__C', 'model__SVR__epsilon', 'model__SVR__kernel', 'scale', 'scale__KBinsDiscretizer__encode', 'scale__KBinsDiscretizer__n_bins', 'scale__KBinsDiscretizer__strategy', 'scale__PowerTransformer__standardize', 'scale__QuantileTransformer__n_quantiles', 'scale__QuantileTransformer__output_distribution', 'vectorizer__high_cardinality', 'vectorizer__low_cardinality']
- injected options not in the original plan (Option 3): ['scale__QuantileTransformer__n_quantiles', 'scale__QuantileTransformer__output_distribution', 'feature_eng__SplineTransformer__degree', 'feature_eng__SplineTransformer__include_bias', 'feature_eng__SplineTransformer__n_knots', 'model__LGBMRegressor__learning_rate', 'model__LGBMRegressor__n_estimators', 'model__LGBMRegressor__num_leaves', 'model__LGBMRegressor__reg_lambda', 'model__LGBMRegressor__subsample', 'model__ElasticNet__alpha', 'model__ElasticNet__fit_intercept', 'model__ElasticNet__l1_ratio', 'vectorizer__high_cardinality:StringEncoder(random_state=42)', "scale:QuantileTransformer(n_quantiles=choose_int(10, 100, name='scale__Quant...__n_quantiles'),\n                    output_distribution=choose_from(['normal', 'uniform'], name='scale__Quant..._distribution'),\n                    random_state=42)", "feature_eng:SplineTransformer(degree=choose_int(2, 3, name='feature_eng_...ormer__degree'),\n                  include_bias=choose_from([False, True], name='feature_eng_..._include_bias'),\n                  n_knots=choose_int(3, 10, name='feature_eng_...rmer__n_knots'))", 'model:LGBMRegressor', 'model:ElasticNet', 'vectorizer__low_cardinality', 'scale__KBinsDiscretizer__encode', 'scale__KBinsDiscretizer__n_bins', 'scale__KBinsDiscretizer__strategy', 'feature_eng__PCA__n_components', 'feature_eng__PCA__svd_solver', 'model__GaussianProcessRegressor__alpha', 'model__GaussianProcessRegressor__n_restarts_optimizer', 'model__SVR__C', 'model__SVR__epsilon', 'model__SVR__kernel', 'model__BayesianRidge__alpha_1', 'model__BayesianRidge__alpha_2', 'model__BayesianRidge__fit_intercept', 'model__BayesianRidge__lambda_1', 'model__BayesianRidge__lambda_2', 'model__ExtraTreesRegressor__max_depth', 'model__ExtraTreesRegressor__max_features', 'model__ExtraTreesRegressor__min_samples_leaf', 'model__ExtraTreesRegressor__n_estimators', 'vectorizer__high_cardinality:SimilarityEncoder()', "scale:KBinsDiscretizer(encode=choose_from(['ordinal', 'onehot'], name='scale__KBins...tizer__encode'),\n                 n_bins=choose_int(5, 20, name='scale__KBins...tizer__n_bins'),\n                 random_state=42,\n                 strategy=choose_from(['uniform', 'quantile', 'kmeans'], name='scale__KBins...zer__strategy'))", "feature_eng:PCA(n_components=choose_from([0.9, 0.95, 0.99, 'mle'], name='feature_eng_..._n_components'),\n    random_state=42,\n    svd_solver=choose_from(['auto', 'full', 'arpack', 'randomized'], name='feature_eng__PCA__svd_solver'))", 'model:GaussianProcessRegressor', 'model:SVR', 'model:BayesianRidge', 'model:ExtraTreesRegressor']

## Appendix — MCTS search space
```json
{
  "assemble": [
    "skip",
    "join_gdp",
    "join_life_expectancy",
    "join_legal_rights",
    "all_aggregates"
  ],
  "cleaner__Cleaner__drop_if_constant": [
    "False",
    "True"
  ],
  "cleaner__Cleaner__drop_if_unique": [
    "False",
    "True"
  ],
  "cleaner__Cleaner__parse_numbers": [
    "False",
    "True"
  ],
  "vectorizer__high_cardinality": [
    "MinHashEncoder()",
    "GapEncoder(random_state=42)",
    "StringEncoder(random_state=42)",
    "SimilarityEncoder()"
  ],
  "vectorizer__low_cardinality": [
    "OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False)",
    "OrdinalEncoder()"
  ],
  "impute__SimpleImputer__strategy": [
    "mean",
    "median"
  ],
  "impute__KNNImputer__n_neighbors": [
    3,
    4,
    5,
    6,
    7
  ],
  "impute": [
    "None",
    "SimpleImputer(strategy=choose_from(['mean', 'median'], name='impute__Simp...ter__strategy'))",
    "KNNImputer(n_neighbors=choose_int(3, 7, name='impute__KNNI...__n_neighbors'))"
  ],
  "scale__PowerTransformer__standardize": [
    "False",
    "True"
  ],
  "scale__QuantileTransformer__n_quantiles": [
    10,
    40,
    55,
    70,
    100
  ],
  "scale__QuantileTransformer__output_distribution": [
    "normal",
    "uniform"
  ],
  "scale__KBinsDiscretizer__encode": [
    "ordinal",
    "onehot"
  ],
  "scale__KBinsDiscretizer__n_bins": [
    5,
    10,
    12,
    15,
    20
  ],
  "scale__KBinsDiscretizer__strategy": [
    "uniform",
    "quantile",
    "kmeans"
  ],
  "scale": [
    "None",
    "StandardScaler()",
    "RobustScaler()",
    "PowerTransformer(standardize=choose_from([False, True], name='scale__Power...__standardize'))",
    "QuantileTransformer(n_quantiles=choose_int(10, 100, name='scale__Quant...__n_quantiles'),\n                    output_distribution=choose_from(['normal', 'uniform'], name='scale__Quant..._distribution'),\n                    random_state=42)",
    "KBinsDiscretizer(encode=choose_from(['ordinal', 'onehot'], name='scale__KBins...tizer__encode'),\n                 n_bins=choose_int(5, 20, name='scale__KBins...tizer__n_bins'),\n                 random_state=42,\n                 strategy=choose_from(['uniform', 'quantile', 'kmeans'], name='scale__KBins...zer__strategy'))"
  ],
  "feature_eng__PolynomialFeatures__degree": [
    2,
    3
  ],
  "feature_eng__PolynomialFeatures__interaction_only": [
    "False",
    "True"
  ],
  "feature_eng__SplineTransformer__degree": [
    2,
    3
  ],
  "feature_eng__SplineTransformer__include_bias": [
    "False",
    "True"
  ],
  "feature_eng__SplineTransformer__n_knots": [
    3,
    5,
    6,
    8,
    10
  ],
  "feature_eng__PCA__n_components": [
    "0.9",
    "0.95",
    "0.99",
    "mle"
  ],
  "feature_eng__PCA__svd_solver": [
    "auto",
    "full",
    "arpack",
    "randomized"
  ],
  "feature_eng": [
    "None",
    "PolynomialFeatures(degree=choose_int(2, 3, name='feature_eng_...tures__degree'),\n                   include_bias=False,\n                   interaction_only=choose_from([False, True], name='feature_eng_...eraction_only'))",
    "SplineTransformer(degree=choose_int(2, 3, name='feature_eng_...ormer__degree'),\n                  include_bias=choose_from([False, True], name='feature_eng_..._include_bias'),\n                  n_knots=choose_int(3, 10, name='feature_eng_...rmer__n_knots'))",
    "PCA(n_components=choose_from([0.9, 0.95, 0.99, 'mle'], name='feature_eng_..._n_components'),\n    random_state=42,\n    svd_solver=choose_from(['auto', 'full', 'arpack', 'randomized'], name='feature_eng__PCA__svd_solver'))"
  ],
  "model__Ridge__alpha": [
    0.001,
    0.1,
    1.0,
    10.0,
    1000.0
  ],
  "model__Ridge__fit_intercept": [
    "True",
    "False"
  ],
  "model__HistGradientBoostingRegressor__l2_regularization": [
    0.0,
    0.3333333333333333,
    0.5,
    0.6666666666666666,
    1.0
  ],
  "model__HistGradientBoostingRegressor__learning_rate": [
    0.01,
    0.031072325059538584,
    0.05477225575051663,
    0.09654893846056294,
    0.3
  ],
  "model__HistGradientBoostingRegressor__max_depth": [
    2,
    4,
    5,
    6,
    8
  ],
  "model__HistGradientBoostingRegressor__max_iter": [
    100,
    200,
    250,
    300,
    400
  ],
  "model__HistGradientBoostingRegressor__min_samples_leaf": [
    5,
    10,
    12,
    15,
    20
  ],
  "model__RandomForestRegressor__max_depth": [
    3,
    4,
    5,
    6
  ],
  "model__RandomForestRegressor__max_features": [
    "sqrt",
    "log2",
    "1.0"
  ],
  "model__RandomForestRegressor__min_samples_leaf": [
    5,
    7,
    8,
    10
  ],
  "model__RandomForestRegressor__n_estimators": [
    100,
    167,
    200,
    233,
    300
  ],
  "model__XGBRegressor__learning_rate": [
    0.01,
    0.031072325059538584,
    0.05477225575051663,
    0.09654893846056294,
    0.3
  ],
  "model__XGBRegressor__max_depth": [
    2,
    3,
    4,
    5,
    6
  ],
  "model__XGBRegressor__n_estimators": [
    100,
    167,
    200,
    233,
    300
  ],
  "model__XGBRegressor__reg_lambda": [
    0.1,
    0.36840314986403866,
    0.7071067811865476,
    1.3572088082974532,
    5.0
  ],
  "model__XGBRegressor__subsample": [
    0.5,
    0.6666666666666666,
    0.75,
    0.8333333333333333,
    1.0
  ],
  "model__LGBMRegressor__learning_rate": [
    0.01,
    0.027144176165949066,
    0.0447213595499958,
    0.07368062997280773,
    0.2
  ],
  "model__LGBMRegressor__n_estimators": [
    100,
    233,
    300,
    367,
    500
  ],
  "model__LGBMRegressor__num_leaves": [
    15,
    31,
    39,
    47,
    63
  ],
  "model__LGBMRegressor__reg_lambda": [
    0.0,
    1.6666666666666667,
    2.5,
    3.3333333333333335,
    5.0
  ],
  "model__LGBMRegressor__subsample": [
    0.6,
    0.7333333333333333,
    0.8,
    0.8666666666666667,
    1.0
  ],
  "model__ElasticNet__alpha": [
    0.001,
    0.046415888336127795,
    0.31622776601683805,
    2.1544346900318843,
    100.0
  ],
  "model__ElasticNet__fit_intercept": [
    "True",
    "False"
  ],
  "model__ElasticNet__l1_ratio": [
    0.1,
    0.3666666666666667,
    0.5,
    0.6333333333333333,
    0.9
  ],
  "model__GaussianProcessRegressor__alpha": [
    1e-06,
    0.0001,
    0.0010000000000000002,
    0.01,
    1.0
  ],
  "model__GaussianProcessRegressor__n_restarts_optimizer": [
    0,
    2,
    3,
    5
  ],
  "model__SVR__C": [
    0.1,
    1.0,
    3.1622776601683804,
    10.0,
    100.0
  ],
  "model__SVR__epsilon": [
    0.01,
    0.03684031498640387,
    0.07071067811865477,
    0.13572088082974532,
    0.5
  ],
  "model__SVR__kernel": [
    "rbf",
    "linear",
    "poly"
  ],
  "model__BayesianRidge__alpha_1": [
    1e-09,
    1e-07,
    1.0000000000000004e-06,
    1e-05,
    0.001
  ],
  "model__BayesianRidge__alpha_2": [
    1e-09,
    1e-07,
    1.0000000000000004e-06,
    1e-05,
    0.001
  ],
  "model__BayesianRidge__fit_intercept": [
    "True",
    "False"
  ],
  "model__BayesianRidge__lambda_1": [
    1e-09,
    1e-07,
    1.0000000000000004e-06,
    1e-05,
    0.001
  ],
  "model__BayesianRidge__lambda_2": [
    1e-09,
    1e-07,
    1.0000000000000004e-06,
    1e-05,
    0.001
  ],
  "model__ExtraTreesRegressor__max_depth": [
    2,
    3,
    4,
    5,
    6
  ],
  "model__ExtraTreesRegressor__max_features": [
    "sqrt",
    "log2",
    "1.0"
  ],
  "model__ExtraTreesRegressor__min_samples_leaf": [
    5,
    8,
    10,
    12,
    15
  ],
  "model__ExtraTreesRegressor__n_estimators": [
    100,
    167,
    200,
    233,
    300
  ],
  "model": [
    "Ridge",
    "HistGradientBoostingRegressor",
    "RandomForestRegressor",
    "XGBRegressor",
    "LGBMRegressor",
    "ElasticNet",
    "GaussianProcessRegressor",
    "SVR",
    "BayesianRidge",
    "ExtraTreesRegressor"
  ]
}
```

## Appendix — data digest (sent to the analyst)
```
Dataset: 117 rows x 2 columns. Target column: 'happiness_score'.
Inferred task type: regression.

Columns:
  - Country: dtype=object, missing=0.0%, cardinality=117, examples='Finland', 'Denmark', 'Iceland', 'Switzerland', 'Netherlands'
  - happiness_score (TARGET): dtype=float64, missing=0.0%, cardinality=115, min=2404 max=7821 mean=5548

First 5 rows:
    Country  happiness_score
    Finland           7821.0
    Denmark           7636.0
    Iceland           7557.0
Switzerland           7512.0
Netherlands           7415.0

Auxiliary table 'gdp': 89 rows x 2 columns.
  Columns:
    - Country Name: dtype=object, cardinality=89
    - GDP per capita (current US$): dtype=float64, cardinality=89
  Join-key candidates (main column <-> aux column, overlap):
    - Country <-> Country Name (overlap=100%)

Auxiliary table 'legal_rights': 89 rows x 2 columns.
  Columns:
    - Country Name: dtype=object, cardinality=89
    - Strength of legal rights index (0=weak to 12=strong): dtype=float64, cardinality=13
  Join-key candidates (main column <-> aux column, overlap):
    - Country <-> Country Name (overlap=100%)

Auxiliary table 'life_expectancy': 89 rows x 2 columns.
  Columns:
    - Country Name: dtype=object, cardinality=89
    - Life expectancy at birth, total (years): dtype=float64, cardinality=89
  Join-key candidates (main column <-> aux column, overlap):
    - Country <-> Country Name (overlap=100%)
```
