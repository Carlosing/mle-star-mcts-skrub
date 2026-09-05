# Run: traffic-violations  (classification, metric=accuracy)
model: openai/qwen3.5-397b-a17b  |  budget: 40  |  fallback spec: False
LLM calls: 4  |  tokens: 50,195 (prompt 29,667 + completion 20,528)

## Task
# Task

Predict the violation_type.

# Metric

accuracy

# Dataset

Classification task (12001 training rows, 42 feature columns). train.csv has the features and the 'violation_type' column; test.csv has the features only.

Multi-class (4 classes, heavily imbalanced tail). 40+ mixed columns: dirty free text, dates, times, lat/long, many binary Yes/No flags. The widest feature space of any staged task.

## 1. Data report (analyst output, sent to the planner agent)
# Dataset Analysis for Tabular Classification Pipeline

## Task Type & Target
- **Task:** Multi-class classification
- **Target:** `violation_type` (4 classes)

## Class Balance Assessment
**Severely imbalanced:**
| Class | Frequency | Count (approx) |
|-------|-----------|----------------|
| Warning | 50.0% | ~6,000 |
| Citation | 45.1% | ~5,400 |
| ESERO | 4.8% | ~576 |
| SERO | 0.1% | ~12 |

**Implications:** SERO is extremely rare (only ~12 samples). This will cause:
- Poor recall on minority classes without intervention
- Need for **class weighting** or **focal loss**
- Consider **macro-F1** or **balanced accuracy** as primary metrics (not accuracy)
- Option: merge SERO+ESERO into single "rare" category, or use hierarchical classification

## Column Type Classification

| Type | Columns | Notes |
|------|---------|-------|
| **Drop (ID/Constant)** | `seqid` (11921 unique), `agency` (1 unique), `geolocation` (redundant w/ lat/long) | No predictive value |
| **High-cardinality categorical** | `location` (9589), `model` (1398), `description` (1042), `charge` (401), `driver_city` (667), `search_reason_for_stop` (275), `make` (343), `date_of_stop` (2714), `time_of_stop` (1427) | Need specialized encoding |
| **Low-cardinality categorical** | `subagency` (7), `state` (53), `vehicletype` (19), `color` (24), `race` (6), `gender` (3), `arrest_type` (18), `article` (3), `driver_state` (44), `dl_state` (54) | OneHot or Ordinal |
| **Binary flags** | `accident`, `belts`, `personal_injury`, `property_damage`, `fatal`, `commercial_license`, `hazmat`, `commercial_vehicle`, `alcohol`, `work_zone`, `contributed_to_accident`, `search_conducted` | Direct encoding |
| **High-missing search features** | `search_disposition` (95.6%), `search_reason` (95.6%), `search_type` (95.6%), `search_arrest_reason` (96.9%), `search_outcome` (39.6%) | Consider dropping or "missing" as category |
| **Numeric** | `latitude`, `longitude`, `year` | Scaling for linear models |
| **Datetime** | `date_of_stop`, `time_of_stop` | Parse and expand |

## Recommended Preprocessing Options

### Encoders to Search
| Column Group | Encoder Options |
|--------------|-----------------|
| High-cardinality (`location`, `model`, `charge`, `driver_city`) | **MinHashEncoder** (fast) vs **GapEncoder** (semantic) |
| Text-like (`description`) | **GapEncoder** (10-50 components) or **TF-IDF + TruncatedSVD** |
| Low-cardinality categoricals | **OneHotEncoder** (drop='first') or **OrdinalEncoder** |
| Binary flags | **OrdinalEncoder** (0/1) |
| Datetime | **DatetimeEncoder** (extract hour, dayofweek, month, is_weekend, etc.) |

### Cleaning Steps
1. **Drop:** `seqid`, `agency`, `geolocation`
2. **Parse datetime:** `date_of_stop` → datetime64, `time_of_stop` → timedelta
3. **Handle missing:** Search columns have 95%+ missing—encode "missing" as explicit category rather than imputation
4. **Fix `year`:** Contains invalid values (min=0, max=2913)—clip to reasonable range (1950-2024) or treat outliers

### Feature Engineering Candidates
- **Temporal:** hour_of_day, day_of_week, is_weekend, month, year_from_date
- **Geographic:** haversine distance from centroid, lat/long binning, or keep raw for tree models
- **Interaction:** `search_conducted` × `search_outcome` (when both present)
- **Vehicle age:** 2024 - `year` (after cleaning)

### Scaling
- **Tree models (XGBoost/LightGBM/CatBoost):** No scaling needed
- **Linear models (LogisticRegression):** StandardScaler on numeric features post-encoding

## Candidate Model Families
1. **LightGBM / XGBoost** — Best overall for mixed tabular data, handles missing natively
2. **CatBoost** — Excellent for high-cardinality categoricals without extensive preprocessing
3. **RandomForest** — Robust baseline, handles class imbalance with `class_weight='balanced'`
4. **LogisticRegression** — Interpretable baseline with proper encoding pipeline

## Specific Columns Deserving Dedicated Operators

| Column | Operator | Rationale |
|--------|----------|-----------|
| `date_of_stop` + `time_of_stop` | **DatetimeEncoder** (keep raw + expand) | Temporal patterns in stops likely predictive (time of day, day of week) |
| `description` | **GapEncoder** (20-30 components) | Free-text violation descriptions contain semantic signal not captured by `charge` codes |
| `location` | **MinHashEncoder** (30-50 components) | 9589 unique locations—too high for OneHot, semantic similarity useful |
| `latitude`/`longitude` | **No transform** for trees; **KBinsDiscretizer** for linear models | Geographic signal already numeric |
| `year` | **Outlier clipping** + **VehicleAge** feature | Invalid values (0, 2913) need cleaning; vehicle age more predictive than year |

## Class Imbalance Handling Options
- **class_weight='balanced'** in sklearn models
- **scale_pos_weight** in XGBoost/LightGBM (per-class)
- **Focal Loss** for deep learning approaches
- **SMOTE** or **ADASYN** (caution with high-cardinality features)
- **Threshold tuning** per class post-calibration

## Relational/Auxiliary Table Opportunities
**None identified** — This is a single flat table. However:
- Could create **aggregate features** by grouping (e.g., violation rates by `subagency`, `driver_city`, or `charge`)
- Consider **target encoding** for high-cardinality columns (`location`, `charge`) with proper cross-validation to avoid leakage

## Pipeline Search Space Summary
| Stage | Options to Search |
|-------|-------------------|
| High-card encoding | MinHashEncoder(30/50/100) vs GapEncoder(10/20/30) |
| Text encoding | GapEncoder(20/30) vs TF-IDF(1000)+SVD(50) |
| Datetime features | Hour+DOW vs Full expansion (hour,dow,month,is_weekend,hour_sin/cos) |
| Missing strategy | Keep as category vs Drop high-missing (>90%) columns |
| Model | LightGBM vs CatBoost vs RandomForest |
| Class weighting | 'balanced' vs custom weights vs focal loss |
| Metric | macro_f1 vs balanced_accuracy vs weighted_f1 |

## 2. Generated plan (planner output)
```json
{
  "cleaner": {
    "params": {
      "drop_if_constant": {"choice": [false, true]},
      "drop_if_unique": {"choice": [false, true]},
      "drop_high_missing": {"float": [0.9, 0.99]}
    }
  },
  "vectorizer": {
    "params": {
      "cardinality_threshold": {"int": [10, 50]}
    },
    "slots": {
      "high_cardinality": [
        {"name": "skrub.MinHashEncoder", "params": {"n_components": {"int": [30, 80]}}},
        {"name": "skrub.GapEncoder", "params": {"n_components": {"int": [20, 50]}}}
      ],
      "low_cardinality": [
        "sklearn.preprocessing.OneHotEncoder",
        "sklearn.preprocessing.OrdinalEncoder"
      ],
      "datetime": [
        {"name": "skrub.DatetimeEncoder", "params": {"resolution": {"choice": ["hour", "day", "month"]}, "add_weekday": {"choice": [false, true]}}}
      ],
      "numeric": [
        "sklearn.preprocessing.StandardScaler",
        "sklearn.preprocessing.RobustScaler"
      ]
    }
  },
  "scoped_encodings": [
    {
      "name": "description_text",
      "cols": ["description"],
      "options": [
        {"name": "skrub.GapEncoder", "params": {"n_components": {"int": [20, 40]}}},
        {"name": "skrub.StringEncoder"}
      ],
      "position": "pre_encode",
      "additive": false
    },
    {
      "name": "location_encoding",
      "cols": ["location"],
      "options": [
        {"name": "skrub.MinHashEncoder", "params": {"n_components": {"int": [30, 60]}}},
        {"name": "skrub.GapEncoder", "params": {"n_components": {"int": [20, 40]}}}
      ],
      "position": "pre_encode",
      "additive": false
    },
    {
      "name": "datetime_expansion",
      "cols": ["date_of_stop", "time_of_stop"],
      "options": [
        {"name": "skrub.DatetimeEncoder", "params": {"resolution": {"choice": ["hour", "day"]}, "add_weekday": {"choice": [false, true]}, "add_total_seconds": {"choice": [false, true]}}}
      ],
      "position": "pre_encode",
      "additive": true
    }
  ],
  "stages": [
    {
      "name": "scale",
      "options": [
        "skip",
        "sklearn.preprocessing.StandardScaler",
        "sklearn.preprocessing.RobustScaler"
      ]
    },
    {
      "name": "feature_eng",
      "options": [
        "skip",
        {"name": "sklearn.preprocessing.PolynomialFeatures", "params": {"degree": {"int": [2, 3]}, "interaction_only": {"choice": [true, false]}}}
      ]
    }
  ],
  "model": [
    {
      "name": "lightgbm.LGBMClassifier",
      "prior": 0.85,
      "params": {
        "n_estimators": {"int": [100, 500]},
        "learning_rate": {"float": [0.01, 0.3], "log": true},
        "num_leaves": {"int": [15, 127]},
        "max_depth": {"int": [3, 12]},
        "colsample_bytree": {"float": [0.5, 1.0]},
        "reg_lambda": {"float": [0.0, 5.0]},
        "class_weight": {"choice": ["balanced", "None"]}
      }
    },
    {
      "name": "xgboost.XGBClassifier",
      "prior": 0.8,
      "params": {
        "n_estimators": {"int": [100, 500]},
        "learning_rate": {"float": [0.01, 0.3], "log": true},
        "max_depth": {"int": [2, 10]},
        "subsample": {"float": [0.5, 1.0]},
        "colsample_bytree": {"float": [0.5, 1.0]},
        "reg_lambda": {"float": [0.0, 5.0]},
        "scale_pos_weight": {"float": [1.0, 100.0], "log": true}
      }
    },
    {
      "name": "sklearn.ensemble.HistGradientBoostingClassifier",
      "prior": 0.7,
      "params": {
        "learning_rate": {"float": [0.01, 0.3], "log": true},
        "max_iter": {"int": [100, 500]},
        "max_depth": {"int": [2, 12]},
        "l2_regularization": {"float": [0.0, 1.0]},
        "class_weight": {"choice": ["balanced", "None"]}
      }
    },
    {
      "name": "sklearn.ensemble.RandomForestClassifier",
      "prior": 0.5,
      "params": {
        "n_estimators": {"int": [100, 400]},
        "max_depth": {"int": [3, 20]},
        "min_samples_leaf": {"int": [1, 10]},
        "max_features": {"choice": ["sqrt", "log2", 1.0]},
        "class_weight": {"choice": ["balanced", "None"]}
      }
    },
    {
      "name": "sklearn.linear_model.LogisticRegression",
      "prior": 0.3,
      "params": {
        "C": {"float": [0.001, 1000.0], "log": true},
        "class_weight": {"choice": ["balanced", "None"]},
        "max_iter": {"int": [100, 500]}
      }
    }
  ]
}
```

## 3. Best configuration + score (MCTS incumbent)
```json
{
  "cleaner__Cleaner__drop_if_constant": "False",
  "cleaner__Cleaner__drop_if_unique": "False",
  "scope_description_text": "skip",
  "scope_location_encoding": "skip",
  "scope_datetime_expansion": "skip",
  "vectorizer__TableVectorizer__cardinality_threshold": 30,
  "vectorizer__datetime__DatetimeEncoder__add_weekday": "False",
  "vectorizer__datetime__DatetimeEncoder__resolution": "hour",
  "vectorizer__high_cardinality": "GapEncoder(n_components=choose_int(20, 50, name='vectorizer__..._n_components'),\n           random_state=42)",
  "vectorizer__low_cardinality": "OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False)",
  "vectorizer__numeric": "StandardScaler()",
  "scale": "None",
  "feature_eng": "None",
  "model__LGBMClassifier__class_weight": "balanced",
  "model__LGBMClassifier__colsample_bytree": 0.5,
  "model__LGBMClassifier__learning_rate": 0.01,
  "model__LGBMClassifier__max_depth": 8,
  "model__LGBMClassifier__n_estimators": 300,
  "model__LGBMClassifier__num_leaves": 71,
  "model__LGBMClassifier__reg_lambda": 2.5,
  "model": "LGBMClassifier",
  "vectorizer__high_cardinality__GapEncoder__n_components": 20
}
```
- search reward (accuracy, scale=raw): 0.8825000000000001
- report metric (accuracy): 0.8851763154241288
- top-3 ensemble (accuracy): 0.8887 vs individuals ['0.8853', '0.8867', '0.8783']
- targeted stage (Option 1): model
- focused-refinement bonus phase edited: ['cleaner__Cleaner__drop_if_constant', 'cleaner__Cleaner__drop_if_unique', 'feature_eng', 'model', 'model__LGBMClassifier__class_weight', 'model__LGBMClassifier__colsample_bytree', 'model__LGBMClassifier__learning_rate', 'model__LGBMClassifier__max_depth', 'model__LGBMClassifier__n_estimators', 'model__LGBMClassifier__num_leaves', 'model__LGBMClassifier__reg_lambda', 'scale', 'scope_charge_encoding', 'scope_charge_encoding__MinHashEncoder__n_components', 'scope_datetime_expansion', 'scope_datetime_expansion__DatetimeEncoder__add_total_seconds', 'scope_datetime_expansion__DatetimeEncoder__add_weekday', 'scope_datetime_expansion__DatetimeEncoder__resolution', 'scope_description_text', 'scope_description_text__GapEncoder__n_components', 'scope_description_text__MinHashEncoder__n_components', 'scope_geolocation_parsing', 'scope_geolocation_parsing__GapEncoder__n_components', 'scope_geolocation_parsing__MinHashEncoder__n_components', 'scope_location_encoding', 'scope_location_encoding__GapEncoder__n_components', 'scope_location_encoding__MinHashEncoder__n_components', 'scope_search_flags_interaction', 'scope_search_reason_stop_encoding', 'scope_search_reason_stop_encoding__GapEncoder__n_components', 'vectorizer__TableVectorizer__cardinality_threshold', 'vectorizer__datetime__DatetimeEncoder__add_weekday', 'vectorizer__datetime__DatetimeEncoder__resolution', 'vectorizer__high_cardinality', 'vectorizer__high_cardinality__GapEncoder__n_components', 'vectorizer__high_cardinality__MinHashEncoder__n_components', 'vectorizer__low_cardinality', 'vectorizer__numeric']
- injected options not in the original plan (Option 3): ['scope_geolocation_parsing__GapEncoder__n_components', 'scope_geolocation_parsing__MinHashEncoder__n_components', 'scope_geolocation_parsing', 'scope_search_flags_interaction__PolynomialFeatures__include_bias', 'scope_search_flags_interaction__PolynomialFeatures__interaction_only', 'scope_search_flags_interaction', 'feature_eng__PCA__n_components', 'feature_eng__PCA__whiten', 'vectorizer__numeric:MinMaxScaler()', 'scale:MinMaxScaler()', "feature_eng:PCA(n_components=choose_float(0.5, 0.99, name='feature_eng_..._n_components'),\n    random_state=42,\n    whiten=choose_from([False, True], name='feature_eng__PCA__whiten'))", 'scope_description_text__MinHashEncoder__n_components', 'scope_charge_encoding__MinHashEncoder__n_components', 'scope_charge_encoding', 'scope_search_reason_stop_encoding__GapEncoder__n_components', 'scope_search_reason_stop_encoding', 'vectorizer__numeric__KBinsDiscretizer__encode', 'vectorizer__numeric__KBinsDiscretizer__n_bins', 'vectorizer__numeric__KBinsDiscretizer__strategy', 'feature_eng__TruncatedSVD__n_components', 'model__ExtraTreesClassifier__class_weight', 'model__ExtraTreesClassifier__max_depth', 'model__ExtraTreesClassifier__max_features', 'model__ExtraTreesClassifier__min_samples_leaf', 'model__ExtraTreesClassifier__n_estimators', 'scope_description_text:MinHashEncoder', "vectorizer__numeric:KBinsDiscretizer(encode=choose_from(['onehot', 'ordinal'], name='vectorizer__...tizer__encode'),\n                 n_bins=choose_int(5, 50, name='vectorizer__...tizer__n_bins'),\n                 random_state=42,\n                 strategy=choose_from(['quantile', 'uniform', 'kmeans'], name='vectorizer__...zer__strategy'))", "feature_eng:TruncatedSVD(n_components=choose_int(10, 100, name='feature_eng_..._n_components'),\n             random_state=42)", 'model:ExtraTreesClassifier']

## Appendix — MCTS search space
```json
{
  "cleaner__Cleaner__drop_if_constant": [
    "False",
    "True"
  ],
  "cleaner__Cleaner__drop_if_unique": [
    "False",
    "True"
  ],
  "scope_description_text__GapEncoder__n_components": [
    20,
    27,
    30,
    33,
    40
  ],
  "scope_description_text__MinHashEncoder__n_components": [
    30,
    40,
    45,
    50,
    60
  ],
  "scope_description_text": [
    "skip",
    "GapEncoder",
    "StringEncoder",
    "MinHashEncoder"
  ],
  "scope_location_encoding__MinHashEncoder__n_components": [
    30,
    40,
    45,
    50,
    60
  ],
  "scope_location_encoding__GapEncoder__n_components": [
    20,
    27,
    30,
    33,
    40
  ],
  "scope_location_encoding": [
    "skip",
    "MinHashEncoder",
    "GapEncoder"
  ],
  "scope_datetime_expansion__DatetimeEncoder__add_total_seconds": [
    "False",
    "True"
  ],
  "scope_datetime_expansion__DatetimeEncoder__add_weekday": [
    "False",
    "True"
  ],
  "scope_datetime_expansion__DatetimeEncoder__resolution": [
    "hour",
    "day"
  ],
  "scope_datetime_expansion": [
    "skip",
    "DatetimeEncoder"
  ],
  "scope_geolocation_parsing__GapEncoder__n_components": [
    20,
    30,
    35,
    40,
    50
  ],
  "scope_geolocation_parsing__MinHashEncoder__n_components": [
    30,
    40,
    45,
    50,
    60
  ],
  "scope_geolocation_parsing": [
    "skip",
    "GapEncoder",
    "MinHashEncoder"
  ],
  "scope_charge_encoding__MinHashEncoder__n_components": [
    30,
    40,
    45,
    50,
    60
  ],
  "scope_charge_encoding": [
    "skip",
    "MinHashEncoder"
  ],
  "scope_search_reason_stop_encoding__GapEncoder__n_components": [
    20,
    27,
    30,
    33,
    40
  ],
  "scope_search_reason_stop_encoding": [
    "skip",
    "GapEncoder"
  ],
  "vectorizer__TableVectorizer__cardinality_threshold": [
    10,
    23,
    30,
    37,
    50
  ],
  "vectorizer__datetime__DatetimeEncoder__add_weekday": [
    "False",
    "True"
  ],
  "vectorizer__datetime__DatetimeEncoder__resolution": [
    "hour",
    "day",
    "month"
  ],
  "vectorizer__high_cardinality__MinHashEncoder__n_components": [
    30,
    47,
    55,
    63,
    80
  ],
  "vectorizer__high_cardinality__GapEncoder__n_components": [
    20,
    30,
    35,
    40,
    50
  ],
  "vectorizer__high_cardinality": [
    "MinHashEncoder(n_components=choose_int(30, 80, name='vectorizer__..._n_components'))",
    "GapEncoder(n_components=choose_int(20, 50, name='vectorizer__..._n_components'),\n           random_state=42)"
  ],
  "vectorizer__low_cardinality": [
    "OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False)",
    "OrdinalEncoder()"
  ],
  "vectorizer__numeric__KBinsDiscretizer__encode": [
    "onehot",
    "ordinal"
  ],
  "vectorizer__numeric__KBinsDiscretizer__n_bins": [
    5,
    20,
    28,
    35,
    50
  ],
  "vectorizer__numeric__KBinsDiscretizer__strategy": [
    "quantile",
    "uniform",
    "kmeans"
  ],
  "vectorizer__numeric": [
    "StandardScaler()",
    "RobustScaler()",
    "MinMaxScaler()",
    "KBinsDiscretizer(encode=choose_from(['onehot', 'ordinal'], name='vectorizer__...tizer__encode'),\n                 n_bins=choose_int(5, 50, name='vectorizer__...tizer__n_bins'),\n                 random_state=42,\n                 strategy=choose_from(['quantile', 'uniform', 'kmeans'], name='vectorizer__...zer__strategy'))"
  ],
  "scope_search_flags_interaction__PolynomialFeatures__include_bias": [
    "False",
    "True"
  ],
  "scope_search_flags_interaction__PolynomialFeatures__interaction_only": [
    "True",
    "False"
  ],
  "scope_search_flags_interaction": [
    "skip",
    "PolynomialFeatures"
  ],
  "scale": [
    "None",
    "StandardScaler()",
    "RobustScaler()",
    "MinMaxScaler()"
  ],
  "feature_eng__PolynomialFeatures__degree": [
    2,
    3
  ],
  "feature_eng__PolynomialFeatures__interaction_only": [
    "True",
    "False"
  ],
  "feature_eng__PCA__n_components": [
    0.5,
    0.6633333333333333,
    0.745,
    0.8266666666666667,
    0.99
  ],
  "feature_eng__PCA__whiten": [
    "False",
    "True"
  ],
  "feature_eng__TruncatedSVD__n_components": [
    10,
    40,
    55,
    70,
    100
  ],
  "feature_eng": [
    "None",
    "PolynomialFeatures(degree=choose_int(2, 3, name='feature_eng_...tures__degree'),\n                   include_bias=False,\n                   interaction_only=choose_from([True, False], name='feature_eng_...eraction_only'))",
    "PCA(n_components=choose_float(0.5, 0.99, name='feature_eng_..._n_components'),\n    random_state=42,\n    whiten=choose_from([False, True], name='feature_eng__PCA__whiten'))",
    "TruncatedSVD(n_components=choose_int(10, 100, name='feature_eng_..._n_components'),\n             random_state=42)"
  ],
  "model__LGBMClassifier__class_weight": [
    "balanced",
    "None"
  ],
  "model__LGBMClassifier__colsample_bytree": [
    0.5,
    0.6666666666666666,
    0.75,
    0.8333333333333333,
    1.0
  ],
  "model__LGBMClassifier__learning_rate": [
    0.01,
    0.031072325059538584,
    0.05477225575051663,
    0.09654893846056294,
    0.3
  ],
  "model__LGBMClassifier__max_depth": [
    3,
    6,
    8,
    9,
    12
  ],
  "model__LGBMClassifier__n_estimators": [
    100,
    233,
    300,
    367,
    500
  ],
  "model__LGBMClassifier__num_leaves": [
    15,
    52,
    71,
    90,
    127
  ],
  "model__LGBMClassifier__reg_lambda": [
    0.0,
    1.6666666666666667,
    2.5,
    3.3333333333333335,
    5.0
  ],
  "model__XGBClassifier__colsample_bytree": [
    0.5,
    0.6666666666666666,
    0.75,
    0.8333333333333333,
    1.0
  ],
  "model__XGBClassifier__learning_rate": [
    0.01,
    0.031072325059538584,
    0.05477225575051663,
    0.09654893846056294,
    0.3
  ],
  "model__XGBClassifier__max_depth": [
    2,
    5,
    6,
    7,
    10
  ],
  "model__XGBClassifier__n_estimators": [
    100,
    233,
    300,
    367,
    500
  ],
  "model__XGBClassifier__reg_lambda": [
    0.0,
    1.6666666666666667,
    2.5,
    3.3333333333333335,
    5.0
  ],
  "model__XGBClassifier__scale_pos_weight": [
    1.0,
    4.641588833612778,
    10.000000000000002,
    21.544346900318832,
    100.0
  ],
  "model__XGBClassifier__subsample": [
    0.5,
    0.6666666666666666,
    0.75,
    0.8333333333333333,
    1.0
  ],
  "model__HistGradientBoostingClassifier__class_weight": [
    "balanced",
    "None"
  ],
  "model__HistGradientBoostingClassifier__l2_regularization": [
    0.0,
    0.3333333333333333,
    0.5,
    0.6666666666666666,
    1.0
  ],
  "model__HistGradientBoostingClassifier__learning_rate": [
    0.01,
    0.031072325059538584,
    0.05477225575051663,
    0.09654893846056294,
    0.3
  ],
  "model__HistGradientBoostingClassifier__max_depth": [
    2,
    5,
    7,
    9,
    12
  ],
  "model__HistGradientBoostingClassifier__max_iter": [
    100,
    233,
    300,
    367,
    500
  ],
  "model__RandomForestClassifier__class_weight": [
    "balanced",
    "None"
  ],
  "model__RandomForestClassifier__max_depth": [
    3,
    9,
    12,
    14,
    20
  ],
  "model__RandomForestClassifier__max_features": [
    "sqrt",
    "log2",
    "1.0"
  ],
  "model__RandomForestClassifier__min_samples_leaf": [
    1,
    4,
    6,
    7,
    10
  ],
  "model__RandomForestClassifier__n_estimators": [
    100,
    200,
    250,
    300,
    400
  ],
  "model__LogisticRegression__C": [
    0.001,
    0.1,
    1.0,
    10.0,
    1000.0
  ],
  "model__LogisticRegression__class_weight": [
    "balanced",
    "None"
  ],
  "model__LogisticRegression__max_iter": [
    100,
    233,
    300,
    367,
    500
  ],
  "model__ExtraTreesClassifier__class_weight": [
    "balanced",
    "None"
  ],
  "model__ExtraTreesClassifier__max_depth": [
    3,
    9,
    12,
    14,
    20
  ],
  "model__ExtraTreesClassifier__max_features": [
    "sqrt",
    "log2",
    "1.0"
  ],
  "model__ExtraTreesClassifier__min_samples_leaf": [
    1,
    4,
    6,
    7,
    10
  ],
  "model__ExtraTreesClassifier__n_estimators": [
    100,
    200,
    250,
    300,
    400
  ],
  "model": [
    "LGBMClassifier",
    "XGBClassifier",
    "HistGradientBoostingClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
    "ExtraTreesClassifier"
  ]
}
```

## Appendix — data digest (sent to the analyst)
```
Dataset: 12001 rows x 43 columns. Target column: 'violation_type'.
Inferred task type: classification.

Columns:
  - seqid: dtype=object, missing=0.0%, cardinality=11921, examples='fd3ee8f9-05f0-4d57-bdb6-9498d303a94c', '3ca2ee89-9e21-4a47-b583-7c426d6b3137', 'a8a3c188-0cd1-4006-9436-883f04c8f061', 'a239bc53-3581-44a6-8cda-35ac5c5df855', 'fdd85f93-0e0c-4d79-9681-9e1b1fdd9bf4'
  - date_of_stop: dtype=object, missing=0.0%, cardinality=2714, examples='07/23/2019', '07/11/2015', '04/10/2016', '03/17/2015', '08/11/2016'
  - time_of_stop: dtype=object, missing=0.0%, cardinality=1427, examples='00:12:00', '18:00:00', '18:38:00', '21:38:00', '08:14:00'
  - agency: dtype=object, missing=0.0%, cardinality=1, examples='MCP'
  - subagency: dtype=object, missing=0.0%, cardinality=7, examples='3rd District, Silver Spring', '4th District, Wheaton', 'Headquarters and Special Operations', '5th District, Germantown', '6th District, Gaithersburg / Montgomery Village'
  - description: dtype=object, missing=0.0%, cardinality=1042, examples='DRIVER FAILURE TO USE SIGNAL LAMP BEFORE TURN', 'OPERATING VEHICLE ON HIGHWAY WITH UNAUTHORIZED WINDOW TINTING MATERIAL', 'DRIVING VEHICLE ON HIGHWAY WITHOUT CURRENT REGISTRATION PLATES AND VALIDATION TABS', 'PERSON DRIVING MOTOR VEHICLE WHILE LICENSE SUSPENDED UNDER 17-106, 26-204, 26-206, 27-103', 'EXCEEDING MAXIMUM SPEED: 56 MPH IN A POSTED 35 MPH ZONE'
  - location: dtype=object, missing=0.0%, cardinality=9589, examples='PINEY BRANCH RD. AT DALE DR.', 'FENTON ST/WAYNE AVE', 'N/B NORBECK RD @ AVERY RD', 'CENTERWAY RD/CLUB HOUSE RD', '26400 BLK OF WOODFIELD RD'
  - latitude: dtype=float64, missing=0.0%, cardinality=10590, min=0 max=39.7 mean=36.16
  - longitude: dtype=float64, missing=0.0%, cardinality=10681, min=-77.43 max=0 mean=-71.34
  - accident: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - belts: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - personal_injury: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - property_damage: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - fatal: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - commercial_license: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - hazmat: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - commercial_vehicle: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - alcohol: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - work_zone: dtype=object, missing=0.0%, cardinality=2, examples='No', 'Yes'
  - search_conducted: dtype=object, missing=38.6%, cardinality=2, examples='Yes', 'No'
  - search_disposition: dtype=object, missing=95.6%, cardinality=4, examples='Nothing', 'Property Only', 'Contraband and Property', 'Contraband Only'
  - search_outcome: dtype=object, missing=39.6%, cardinality=4, examples='Arrest', 'Citation', 'Warning', 'SERO'
  - search_reason: dtype=object, missing=95.6%, cardinality=6, examples='Probable Cause', 'Incident to Arrest', 'Consensual', 'Exigent Circumstances', 'Other'
  - search_reason_for_stop: dtype=object, missing=38.6%, cardinality=275, examples='13-411(c1)', '21-201(a1)', '21-1124.2(d2)', '23-104', '22-201.1'
  - search_type: dtype=object, missing=95.6%, cardinality=3, examples='Both', 'Property', 'Person'
  - search_arrest_reason: dtype=object, missing=96.9%, cardinality=4, examples='Stop', 'Warrant', 'Other', 'Search'
  - state: dtype=object, missing=0.0%, cardinality=53, examples='MD', 'WA', 'VA', 'XX', 'MB'
  - vehicletype: dtype=object, missing=0.0%, cardinality=19, examples='02 - Automobile', '03 - Station Wagon', '29 - Unknown', '05 - Light Duty Truck', '01 - Motorcycle'
  - year: dtype=float64, missing=0.7%, cardinality=55, min=0 max=2913 mean=2006
  - make: dtype=object, missing=0.7%, cardinality=343, examples='ACURA', 'NISSAN', 'SUBARU', 'HOND', 'TOYOTA'
  - model: dtype=object, missing=0.7%, cardinality=1398, examples='TL', 'MAXIMA', 'IMPREZA', 'TK', 'CAMRY'
  - color: dtype=object, missing=1.2%, cardinality=24, examples='SILVER', 'WHITE', 'BLACK', 'GREEN', 'BLUE'
  - charge: dtype=object, missing=0.0%, cardinality=401, examples='21-605(a)', '22-406(i1)', '13-411(d)', '16-303(h)', '21-801.1'
  - article: dtype=object, missing=4.9%, cardinality=3, examples='Transportation Article', 'Maryland Rules', 'BR'
  - contributed_to_accident: dtype=bool, missing=0.0%, cardinality=2, min=0 max=1 mean=0.02091
  - race: dtype=object, missing=0.0%, cardinality=6, examples='BLACK', 'WHITE', 'ASIAN', 'OTHER', 'HISPANIC'
  - gender: dtype=object, missing=0.0%, cardinality=3, examples='M', 'F', 'U'
  - driver_city: dtype=object, missing=0.0%, cardinality=667, examples='SILVER SPRING', 'LANHAM', 'BELLINGHA', 'GERMANTOWN', 'MOUNT AIRY'
  - driver_state: dtype=object, missing=0.0%, cardinality=44, examples='MD', 'WA', 'VA', 'WV', 'DC'
  - dl_state: dtype=object, missing=0.1%, cardinality=54, examples='MD', 'WA', 'VA', 'XX', 'WV'
  - arrest_type: dtype=object, missing=0.0%, cardinality=18, examples='A - Marked Patrol', 'B - Unmarked Patrol', 'G - Marked Moving Radar (Stationary)', 'Q - Marked Laser', 'E - Marked Stationary Radar'
  - geolocation: dtype=object, missing=0.0%, cardinality=10988, examples='(39.04653, -76.9952916666667)', '(38.9938683333333, -77.0240883333333)', '(39.0915, -77.1258583333333)', '(39.17332, -77.2013716666667)', '(39.2844416666667, -77.201995)'
  - violation_type (TARGET): dtype=object, missing=0.0%, cardinality=4, examples='Citation', 'ESERO', 'SERO', 'Warning'
      class balance: 'Warning'=50.0%, 'Citation'=45.1%, 'ESERO'=4.8%, 'SERO'=0.1%

First 5 rows:
                               seqid date_of_stop time_of_stop agency                           subagency                                                                               description                     location  latitude  longitude accident belts personal_injury property_damage fatal commercial_license hazmat commercial_vehicle alcohol work_zone search_conducted search_disposition search_outcome  search_reason search_reason_for_stop search_type search_arrest_reason state     vehicletype   year   make   model  color     charge                article  contributed_to_accident  race gender   driver_city driver_state dl_state                          arrest_type                           geolocation violation_type
fd3ee8f9-05f0-4d57-bdb6-9498d303a94c   07/23/2019     00:12:00    MCP         3rd District, Silver Spring                                             DRIVER FAILURE TO USE SIGNAL LAMP BEFORE TURN PINEY BRANCH RD. AT DALE DR. 39.046530 -76.995292       No    No              No              No    No                 No     No                 No      No        No              Yes            Nothing         Arrest Probable Cause             13-411(c1)        Both                 Stop    MD 02 - Automobile 2005.0  ACURA      TL SILVER  21-605(a) Transportation Article                    False BLACK      M SILVER SPRING           MD       MD                    A - Marked Patrol         (39.04653, -76.9952916666667)       Citation
3ca2ee89-9e21-4a47-b583-7c426d6b3137   07/11/2015     18:00:00    MCP         3rd District, Silver Spring                    OPERATING VEHICLE ON HIGHWAY WITH UNAUTHORIZED WINDOW TINTING MATERIAL          FENTON ST/WAYNE AVE 38.993868 -77.024088       No    No              No              No    No                 No     No                 No      No        No               No                NaN       Citation            NaN             21-201(a1)         NaN                  NaN    MD 02 - Automobile 2006.0 NISSAN  MAXIMA  WHITE 22-406(i1) Transportation Article                    False BLACK      M        LANHAM           MD       MD                    A - Marked Patrol (38.9938683333333, -77.0240883333333)       Citation
a8a3c188-0cd1-4006-9436-883f04c8f061   04/10/2016     18:38:00    MCP               4th District, Wheaton        DRIVING VEHICLE ON HIGHWAY WITHOUT CURRENT REGISTRATION PLATES AND VALIDATION TABS    N/B NORBECK RD @ AVERY RD 39.091500 -77.125858       No   Yes              No              No    No                 No     No                 No      No        No               No                NaN       Citation            NaN          21-1124.2(d2)         NaN                  NaN    WA 02 - Automobile 2004.0 SUBARU IMPREZA SILVER  13-411(d) Transportation Article                    False WHITE      M     BELLINGHA           WA       WA                    A - Marked Patrol          (39.0915, -77.1258583333333)       Citation
a239bc53-3581-44a6-8cda-35ac5c5df855   03/17/2015     21:38:00    MCP Headquarters and Special Operations PERSON DRIVING MOTOR VEHICLE WHILE LICENSE SUSPENDED UNDER 17-106, 26-204, 26-206, 27-103   CENTERWAY RD/CLUB HOUSE RD 39.173320 -77.201372       No    No              No              No    No                 No     No                 No      No        No               No                NaN       Citation            NaN                 23-104         NaN                  NaN    MD 02 - Automobile 1997.0   HOND      TK  BLACK  16-303(h) Transportation Article                    False BLACK      M    GERMANTOWN           MD       MD                  B - Unmarked Patrol         (39.17332, -77.2013716666667)       Citation
fdd85f93-0e0c-4d79-9681-9e1b1fdd9bf4   08/11/2016     08:14:00    MCP            5th District, Germantown                                   EXCEEDING MAXIMUM SPEED: 56 MPH IN A POSTED 35 MPH ZONE    26400 BLK OF WOODFIELD RD 39.284442 -77.201995       No    No              No              No    No                 No     No                 No      No        No              NaN                NaN            NaN            NaN                    NaN         NaN                  NaN    MD 02 - Automobile 2004.0 TOYOTA   CAMRY SILVER   21-801.1 Transportation Article                    False WHITE      F    MOUNT AIRY           MD       MD G - Marked Moving Radar (Stationary)        (39.2844416666667, -77.201995)       Citation
```
