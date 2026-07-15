"""Claude standing in for the Gemini agent layer — offline, zero-quota.

The pipeline has exactly two LLM touchpoints, both injectable without touching
the logic layer:

  1. plan authoring (``data_analyst`` -> ``plan_author``): one JSON plan per
     task, fed to ``run_pipeline(spec_raw=...)`` which then skips both ADK
     agents entirely.
  2. Extended Feature 3 (``search_loop`` proposer): a plain ``propose(plan_json,
     context) -> dict`` callable returning an extension of the current plan —
     ``make_llm_proposer`` is just the Gemini default; ``make_replay_proposer``
     here is a pure, offline stand-in.

``PLANS`` and ``PROPOSALS`` below were authored by Claude reading the same
``make_data_summary`` digest and the same allowed-operator vocabulary the real
agents see (see ``scripts/run_claude_pipeline.py`` for how they are wired). They
stand in for *Gemini's* plans/proposals — results measure MCTS over a
Claude-authored space, which is the point: full offline throughput without
burning the Gemini free-tier quota.

Each plan deliberately holds back a strong operator or two (e.g. LGBM/XGBoost on
the model stage) so the Extended Feature 3 proposer has a genuinely NEW option to inject —
that is what the flexibility-lift figure measures.
"""

import json


def spec_raw(task: str, task_type: str | None = None) -> str:
    """The authored plan for ``task`` as the JSON string ``plan_author`` emits.

    A task with no authored plan falls back to a generic per-task-type menu
    (``_generic_plan``) so the driver runs on any staged task rather than
    raising — the same graceful degradation ``pipeline._safe_resolve`` gives
    the live agents.

    Example:
        spec_raw("open-payments")                       # -> the authored plan
        spec_raw("some-new-task", "regression")         # -> the generic menu
    """
    plan = PLANS.get(task)
    if plan is None:
        if task_type is None:
            raise KeyError(
                f"no authored plan for {task!r} and no task_type to fall back on; "
                f"authored: {sorted(PLANS)}"
            )
        plan = _generic_plan(task_type)
    return json.dumps(plan)


def _generic_plan(task_type: str) -> dict:
    """A task-agnostic plan menu: clean/encoder choices + the three base models."""
    return {
        "vectorizer": {
            "slots": {
                "high_cardinality": [
                    "skrub.GapEncoder",
                    "skrub.MinHashEncoder",
                    "skrub.StringEncoder",
                ]
            }
        },
        "stages": [
            {
                "name": "scale",
                "options": ["skip", "sklearn.preprocessing.StandardScaler"],
            }
        ],
        "model": _CLF_MODELS if task_type == "classification" else _REG_MODELS,
    }


def proposal_for(task: str, task_type: str) -> dict:
    """The authored Extended Feature 3 extension for ``task``, else the booster menu."""
    return PROPOSALS.get(task) or {"model": _boosters(task_type)}


def make_replay_proposer(extension: dict, log_dir=None):
    """A pure, offline ``propose(plan_json, context) -> dict`` stand-in.

    Returns the pre-authored *extension plan* (a partial raw plan: only the
    stages with new entries, with HP ranges) on every call — identical
    contract to ``make_llm_proposer`` minus the network call. The search
    loop's ``_merge_raw_plans`` unions it into the current plan additively, so
    repeat calls are harmless no-ops after the first merge. It is
    intentionally blind to the live search evidence (the one honesty caveat
    vs the Gemini proposer): the extension is authored up front from the data
    digest, not the mid-search state.

    Example:
        propose = make_replay_proposer(
            {"model": [{"name": "lightgbm.LGBMRegressor",
                        "params": {"n_estimators": {"int": [100, 600]}}}]})
        propose(plan_json, context)  # -> that extension dict, every call
    """
    calls = [0]

    def propose(plan_json, context=None):
        calls[0] += 1
        if log_dir is not None:
            _log(
                log_dir,
                calls[0],
                (context or {}).get("target_stage"),
                extension,
            )
        return extension

    return propose


def _log(log_dir, call_num, target_stage, extension):
    import os

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"proposer_{call_num}_response.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "agent": "claude_replay_proposer",
                "call": call_num,
                "target_stage": target_stage,
                "proposal": extension,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


# --- authored plans (Claude as data_analyst -> plan_author) ------------------
#
# Vocabulary + curated HP ranges come from spec_resolver.format_allowed_for_prompt();
# each plan follows the plan_author JSON contract (per-stage menus, not single
# choices). Model params stay inside the curated ranges (clipped anyway).

_REG_MODELS = [
    {
        "name": "sklearn.ensemble.HistGradientBoostingRegressor",
        "prior": 0.8,
        "params": {
            "learning_rate": {"float": [0.01, 0.3], "log": True},
            "max_iter": {"int": [100, 600]},
            "max_depth": {"int": [2, 16]},
            "l2_regularization": {"float": [0.0, 1.0]},
        },
    },
    {
        "name": "sklearn.ensemble.RandomForestRegressor",
        "prior": 0.5,
        "params": {
            "n_estimators": {"int": [100, 500]},
            "max_depth": {"int": [3, 30]},
            "min_samples_leaf": {"int": [1, 10]},
        },
    },
    {
        "name": "sklearn.linear_model.Ridge",
        "prior": 0.3,
        "params": {"alpha": {"float": [0.001, 1000.0], "log": True}},
    },
]

_CLF_MODELS = [
    {
        "name": "sklearn.ensemble.HistGradientBoostingClassifier",
        "prior": 0.8,
        "params": {
            "learning_rate": {"float": [0.01, 0.3], "log": True},
            "max_iter": {"int": [100, 600]},
            "max_depth": {"int": [2, 16]},
            "l2_regularization": {"float": [0.0, 1.0]},
        },
    },
    {
        "name": "sklearn.ensemble.RandomForestClassifier",
        "prior": 0.5,
        "params": {
            "n_estimators": {"int": [100, 500]},
            "max_depth": {"int": [3, 30]},
            "min_samples_leaf": {"int": [1, 10]},
        },
    },
    {
        "name": "sklearn.linear_model.LogisticRegression",
        "prior": 0.3,
        "params": {"C": {"float": [0.001, 1000.0], "log": True}},
    },
]

PLANS: dict[str, dict] = {
    # All-numeric regression: no categoricals/dates — search scaling +
    # polynomial interactions (lat/long) + a tuned model.
    "california-housing-prices": {
        "vectorizer": {
            "slots": {"high_cardinality": ["skrub.StringEncoder"]}
        },  # no high-card cats; moot
        "stages": [
            {
                "name": "scale",
                "options": [
                    "skip",
                    "sklearn.preprocessing.StandardScaler",
                    "sklearn.preprocessing.RobustScaler",
                ],
            },
            {
                "name": "feature_eng",
                "options": [
                    "skip",
                    {
                        "name": "sklearn.preprocessing.PolynomialFeatures",
                        "params": {"degree": {"int": [2, 3]}},
                    },
                ],
            },
        ],
        "model": _REG_MODELS,
    },
    # Relational: main table is (ID, fraud_flag); all signal is in the
    # `products` aux table, joined on ID<->basket_ID. Imbalanced (1.25% pos),
    # roc_auc — stratify is auto-enabled by the pipeline for classification.
    "credit-fraud": {
        "assemble": [
            {
                "name": "basket_stats",
                "table": "products",
                "main_key": "ID",
                "aux_key": "basket_ID",
                "operations": ["mean", "max", "sum", "count"],
                "cols": ["cash_price", "Nbr_of_prod_purchas"],
            },
            {
                "name": "basket_rich",
                "table": "products",
                "main_key": "ID",
                "aux_key": "basket_ID",
                "operations": ["mean", "min", "max", "sum", "std", "count"],
                "cols": ["cash_price", "Nbr_of_prod_purchas"],
            },
            {
                "name": "basket_mode",
                "table": "products",
                "main_key": "ID",
                "aux_key": "basket_ID",
                "operations": ["mode", "count"],
                "cols": ["item", "make"],
            },
        ],
        "vectorizer": {
            "slots": {
                "high_cardinality": ["skrub.GapEncoder", "skrub.MinHashEncoder"]
            }
        },
        "stages": [
            {
                "name": "scale",
                "options": ["skip", "sklearn.preprocessing.StandardScaler"],
            }
        ],
        "model": _CLF_MODELS,
    },
    # Dirty high-card categoricals (division 650, title 402) + a date string.
    "employee-salaries": {
        "vectorizer": {
            "slots": {
                "high_cardinality": ["skrub.GapEncoder", "skrub.StringEncoder"]
            }
        },
        "scoped_encodings": [
            {
                "name": "hire_date",
                "cols": ["date_first_hired"],
                "options": [
                    "skip",
                    {
                        "name": "skrub.DatetimeEncoder",
                        "params": {"resolution": {"choice": ["month", "day"]}},
                    },
                ],
                "position": "pre_encode",
                "additive": False,
            }
        ],
        "stages": [
            {
                "name": "scale",
                "options": ["skip", "sklearn.preprocessing.StandardScaler"],
            }
        ],
        "model": _REG_MODELS,
    },
    # 9-class classification; many binary Yes/No cols + high-card free text
    # ("what would you call..." 780, ZIP 1734).
    "midwest-survey": {
        "vectorizer": {
            "slots": {
                "high_cardinality": ["skrub.GapEncoder", "skrub.StringEncoder"]
            }
        },
        "scoped_encodings": [
            {
                "name": "region_freetext",
                "cols": [
                    "What_would_you_call_the_part_of_the_country_you_live_in_now"
                ],
                "options": ["skip", "skrub.MinHashEncoder"],
                "position": "pre_encode",
                "additive": False,
            }
        ],
        "model": _CLF_MODELS,
    },
    # Binary classification; high-card dirty text with heavy missingness.
    "open-payments": {
        "vectorizer": {
            "slots": {
                "high_cardinality": [
                    "skrub.GapEncoder",
                    "skrub.MinHashEncoder",
                    "skrub.StringEncoder",
                ]
            }
        },
        "model": _CLF_MODELS,
    },
    # --- staged by scripts/stage_tasks.py ------------------------------------
    # Hourly rentals: one date string + cyclical weather numerics. The date is
    # the whole game — expand it additively so the raw column also survives.
    "bike-sharing": {
        "vectorizer": {"slots": {"high_cardinality": ["skrub.StringEncoder"]}},
        "scoped_encodings": [
            {
                "name": "ride_date",
                "cols": ["date"],
                "options": [
                    "skip",
                    {
                        "name": "skrub.DatetimeEncoder",
                        "params": {"resolution": {"choice": ["day", "hour"]}},
                    },
                ],
                "position": "pre_encode",
                "additive": False,
            }
        ],
        "stages": [
            {
                "name": "scale",
                "options": ["skip", "sklearn.preprocessing.StandardScaler"],
            }
        ],
        "model": _REG_MODELS,
    },
    # Dirty high-cardinality provider names/addresses; two informative numerics.
    "medical-charge": {
        "vectorizer": {
            "slots": {
                "high_cardinality": ["skrub.GapEncoder", "skrub.MinHashEncoder"]
            }
        },
        "stages": [
            {
                "name": "scale",
                "options": ["skip", "sklearn.preprocessing.RobustScaler"],
            }
        ],
        "model": _REG_MODELS,
    },
    # ONE free-text column: the encoder stage is effectively the whole search.
    "toxicity": {
        "vectorizer": {
            "slots": {
                "high_cardinality": [
                    {
                        "name": "skrub.GapEncoder",
                        "params": {"n_components": {"int": [10, 50]}},
                    },
                    {
                        "name": "skrub.MinHashEncoder",
                        "params": {"n_components": {"int": [20, 80]}},
                    },
                    "skrub.StringEncoder",
                ]
            }
        },
        "model": _CLF_MODELS,
    },
    # 40+ mixed columns, 4 imbalanced classes; free text + dates + Yes/No flags.
    "traffic-violations": {
        "vectorizer": {
            "slots": {
                "high_cardinality": [
                    "skrub.MinHashEncoder",
                    "skrub.StringEncoder",
                ]
            }
        },
        "scoped_encodings": [
            {
                "name": "stop_date",
                "cols": ["date_of_stop"],
                "options": ["skip", "skrub.DatetimeEncoder"],
                "position": "pre_encode",
                "additive": True,
            }
        ],
        "model": _CLF_MODELS,
    },
    # Heavy-tailed sales from title/platform/genre/publisher only (regional
    # sales dropped as leakage) — the high-card 'Name' column dominates.
    "videogame-sales": {
        "vectorizer": {
            "slots": {
                "high_cardinality": [
                    "skrub.GapEncoder",
                    "skrub.MinHashEncoder",
                    "skrub.StringEncoder",
                ]
            }
        },
        "model": _REG_MODELS,
    },
    # Relational, credit-fraud shaped: main is (Country, happiness_score), so
    # every feature must arrive through an aggregate join. 1:1 joins, so 'mean'
    # of the single value column just lifts it onto the main table.
    "country-happiness": {
        "assemble": [
            {
                "name": "gdp_mean",
                "table": "gdp",
                "main_key": "Country",
                "aux_key": "Country Name",
                "operations": ["mean"],
                "cols": ["GDP per capita (current US$)"],
            },
            {
                "name": "life_mean",
                "table": "life_expectancy",
                "main_key": "Country",
                "aux_key": "Country Name",
                "operations": ["mean"],
                "cols": ["Life expectancy at birth, total (years)"],
            },
            {
                "name": "legal_mean",
                "table": "legal_rights",
                "main_key": "Country",
                "aux_key": "Country Name",
                "operations": ["mean"],
                "cols": [
                    "Strength of legal rights index (0=weak to 12=strong)"
                ],
            },
        ],
        "vectorizer": {"slots": {"high_cardinality": ["skrub.StringEncoder"]}},
        "stages": [
            {
                "name": "scale",
                "options": ["skip", "sklearn.preprocessing.StandardScaler"],
            }
        ],
        "model": _REG_MODELS,
    },
    # Relational regression with ~3% missing targets; joins origin-airport
    # geography, and the schedule columns are datetimes.
    "flight-delays": {
        "assemble": [
            {
                "name": "origin_geo",
                "table": "airports",
                "main_key": "Origin",
                "aux_key": "iata",
                "operations": ["mean"],
                "cols": ["lat", "long"],
            },
            {
                "name": "origin_place",
                "table": "airports",
                "main_key": "Origin",
                "aux_key": "iata",
                "operations": ["mode"],
                "cols": ["city", "state"],
            },
        ],
        "vectorizer": {
            "slots": {
                "high_cardinality": [
                    "skrub.StringEncoder",
                    "skrub.MinHashEncoder",
                ]
            }
        },
        "stages": [
            {
                "name": "scale",
                "options": ["skip", "sklearn.preprocessing.StandardScaler"],
            }
        ],
        "model": _REG_MODELS,
    },
    # Relational: main is (userId, movieId) — pure IDs. Without the join there
    # is nothing to learn, so 'assemble' carries the whole signal.
    "movielens": {
        "assemble": [
            {
                "name": "movie_meta",
                "table": "movies",
                "key": "movieId",
                "operations": ["mode"],
                "cols": ["genres", "title"],
            },
            {
                "name": "movie_genres",
                "table": "movies",
                "key": "movieId",
                "operations": ["mode", "count"],
                "cols": ["genres"],
            },
        ],
        "vectorizer": {
            "slots": {
                "high_cardinality": ["skrub.GapEncoder", "skrub.MinHashEncoder"]
            }
        },
        "model": _REG_MODELS,
    },
}


# --- authored Extended Feature 3 extension plans (Claude as the proposer) ---------------
#
# Each value is a PARTIAL raw plan (only the stages that gain entries) merged
# additively into the current plan by search_loop._merge_raw_plans. Every
# entry is NEW vs the plan above, so an injection demonstrably keeps an option
# the original plan lacked (the flexibility-lift figure) — and every entry
# carries HP ranges, so it enters the search TUNED, competing on equal footing
# with the plan's tuned HGB. Gradient boosters are held back here: they are
# strong, the ranges below sit inside the curated REGISTRY bounds (clipped
# anyway), and the REGISTRY pins n_jobs=1 so they are safe from the macOS-ARM
# libomp segfault when the resolver constructs them.

_LGBM_PARAMS = {
    "n_estimators": {"int": [100, 600]},
    "learning_rate": {"float": [0.01, 0.3], "log": True},
    "num_leaves": {"int": [15, 127]},
}
_XGB_PARAMS = {
    "n_estimators": {"int": [100, 600]},
    "learning_rate": {"float": [0.01, 0.3], "log": True},
    "max_depth": {"int": [3, 10]},
}


def _boosters(task_type: str) -> list[dict]:
    suffix = "Classifier" if task_type == "classification" else "Regressor"
    return [
        {
            "name": f"lightgbm.LGBM{suffix}",
            "prior": 0.7,
            "params": dict(_LGBM_PARAMS),
        },
        {
            "name": f"xgboost.XGB{suffix}",
            "prior": 0.6,
            "params": dict(_XGB_PARAMS),
        },
    ]


PROPOSALS: dict[str, dict] = {
    "california-housing-prices": {"model": _boosters("regression")},
    "credit-fraud": {"model": _boosters("classification")},
    "employee-salaries": {
        "model": _boosters("regression"),
        "vectorizer": {"slots": {"high_cardinality": ["skrub.MinHashEncoder"]}},
    },
    "midwest-survey": {"model": _boosters("classification")},
    "open-payments": {"model": _boosters("classification")},
    "bike-sharing": {"model": _boosters("regression")},
    "medical-charge": {"model": _boosters("regression")},
    "toxicity": {"model": _boosters("classification")},
    "traffic-violations": {"model": _boosters("classification")},
    "videogame-sales": {"model": _boosters("regression")},
    "country-happiness": {"model": _boosters("regression")},
    "flight-delays": {"model": _boosters("regression")},
    "movielens": {"model": _boosters("regression")},
}

ALL_TASKS = list(PLANS)
