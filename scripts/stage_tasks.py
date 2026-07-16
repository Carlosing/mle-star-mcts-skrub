"""Stage the cached `data/` datasets as tasks under `machine_learning_engineering/tasks/`.

`data/<slug>/` holds skrub's downloaded CSVs + a `metadata.json` naming the
target. A *task* is the layout `pipeline.load_task` expects:

    tasks/<name>/train.csv               <- features + target; the ONLY file any method may train on
    tasks/<name>/test.csv                <- the shared holdout, features only (no target)
    tasks/<name>/test_answer.csv         <- the holdout targets; SCORERS ONLY, never given to a method
    tasks/<name>/aux_<table>.csv         <- auxiliary tables, joined whole (relational)
    tasks/<name>/task_description.txt    <- "Predict the <target>." + "# Metric"

The split is 80/20, seeded, stratified for classification. It is the *only*
train/test boundary in the project, and it is drawn HERE, on disk, before any
method sees the data — so no method can be trusted-but-verified into honouring
it; the holdout rows are simply absent from every file a method can read.

Every arm of the benchmark (the skrub+MCTS extension, AutoGluon, MLE-STAR)
trains on `train.csv`, predicts the rows of `test.csv`, and is scored against
`test_answer.csv`. That is what makes the three numbers comparable.

Historically `test.csv` was written WITHOUT its targets and read by nobody: the
extension and AutoGluon each carved a private holdout out of `train.csv`, and
the search cross-validated over rows that later became the extension's own eval
set. See docs/BUG_LEDGER.md.

Each recipe declares the *dataset-specific* knowledge that cannot be inferred:
which columns leak the target, how big a subsample keeps CV rollouts fast, and
how auxiliary tables join. Targets are renamed to a safe identifier and moved
last, because `load_task._parse_target` matches ``Predict the <ident>`` and
otherwise falls back to the last column.

Run (offline — reads only `data/`):
    uv run python scripts/stage_tasks.py            # every missing task
    uv run python scripts/stage_tasks.py --only toxicity movielens
    uv run python scripts/stage_tasks.py --force    # restage existing ones too
"""

import argparse
import json
import os

import pandas as pd

SEED = 42
DATA_DIR = "data"
TASKS_DIR = os.path.join("machine_learning_engineering", "tasks")


# --- recipes -----------------------------------------------------------------
#
# `drop`: columns that LEAK the target (verified, not guessed — see the note on
# each). `max_rows`: seeded subsample of the MAIN table so a CV rollout stays
# fast; auxiliary tables are never subsampled except to follow the main table's
# join keys. `aux`: {task_table_name: {"csv", "main_key", "aux_key"}}.

RECIPES: dict[str, dict] = {
    "bike-sharing": {
        "source": "bike_sharing",
        "csv": "bike_sharing.csv",
        "target": "cnt",
        "rename": "rentals",
        # casual + registered == cnt EXACTLY (verified); instant is the row index
        "drop": ["casual", "registered", "instant"],
        "metric": "root_mean_squared_error",
        "note": "Hourly bike rentals. 'casual'/'registered' were dropped: they "
        "sum exactly to the target. A date column plus cyclical weather "
        "numerics — a natural fit for a DatetimeEncoder scoped group.",
    },
    "medical-charge": {
        "source": "medical_charge",
        "csv": "medical_charge.csv",
        "target": "Average_Total_Payments",
        "rename": "average_total_payments",
        # Average_Medicare_Payments is a COMPONENT of the target (r=0.989, it is
        # ~85% of it by construction). Average_Covered_Charges (r=0.77) stays.
        "drop": ["Average_Medicare_Payments"],
        "max_rows": 20_000,
        "metric": "root_mean_squared_error",
        "note": "Hospital inpatient payments. 'Average_Medicare_Payments' was "
        "dropped: it is a component of the target (r=0.989). Dirty "
        "high-cardinality provider names/addresses drive the difficulty.",
    },
    "toxicity": {
        "source": "toxicity_v1",
        "csv": "toxicity_v1.csv",
        "target": "is_toxic",
        "rename": "is_toxic",
        "metric": "accuracy",
        "note": "1000 rows, ONE free-text feature column. A pure "
        "text-encoder benchmark: the entire action space that matters is "
        "the encoder stage (GapEncoder vs MinHash vs StringEncoder).",
    },
    "traffic-violations": {
        "source": "traffic_violations",
        "csv": "traffic_violations.csv",
        "target": "violation_type",
        "rename": "violation_type",
        "max_rows": 15_000,
        "metric": "accuracy",
        "note": "Multi-class (4 classes, heavily imbalanced tail). 40+ mixed "
        "columns: dirty free text, dates, times, lat/long, many binary "
        "Yes/No flags. The widest feature space of any staged task.",
    },
    "videogame-sales": {
        "source": "videogame_sales",
        "csv": "videogame_sales.csv",
        "target": "Global_Sales",
        "rename": "global_sales",
        # the four regional sales columns sum to Global_Sales (verified);
        # Rank is a monotone function of it
        "drop": ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Rank"],
        "metric": "root_mean_squared_error",
        "note": "Regional sales columns and 'Rank' were dropped: they sum to / "
        "rank the target. What remains (title, platform, genre, "
        "publisher, year) makes this a genuinely hard, heavy-tailed "
        "regression.",
    },
    # --- relational -----------------------------------------------------------
    "country-happiness": {
        "source": "country_happiness",
        "csv": "happiness_report.csv",
        "target": "Happiness score",
        "rename": "happiness_score",
        # RANK / Whisker-high / Whisker-low / Dystopia+residual are direct
        # functions of the score, and every "Explained by:" column is a term of
        # its additive decomposition. Dropping them leaves (Country, score) —
        # ALL signal must come from the aggregate joins, as in credit-fraud.
        "drop": [
            "RANK",
            "Whisker-high",
            "Whisker-low",
            "Dystopia (1.83) + residual",
            "Explained by: GDP per capita",
            "Explained by: Social support",
            "Explained by: Healthy life expectancy",
            "Explained by: Freedom to make life choices",
            "Explained by: Generosity",
            "Explained by: Perceptions of corruption",
        ],
        "metric": "root_mean_squared_error",
        "aux": {
            "gdp": {
                "csv": "GDP_per_capita.csv",
                "main_key": "Country",
                "aux_key": "Country Name",
            },
            "life_expectancy": {
                "csv": "life_expectancy.csv",
                "main_key": "Country",
                "aux_key": "Country Name",
            },
            "legal_rights": {
                "csv": "legal_rights_index.csv",
                "main_key": "Country",
                "aux_key": "Country Name",
            },
        },
        "note": "Relational, credit-fraud shaped: the main table is reduced to "
        "(Country, happiness_score) — every leaky decomposition column "
        "was dropped — so ALL signal must arrive through aggregate "
        "joins on three World-Bank aux tables. Only 146 rows, and the "
        "join is by country NAME across differently-named key columns "
        "(Country <-> 'Country Name'), so misses are silent.",
    },
    "flight-delays": {
        "source": "flight_delays",
        "csv": "flights.csv",
        "target": "ArrDelay",
        "rename": "arr_delay",
        "max_rows": 15_000,
        "metric": "root_mean_squared_error",
        "aux": {
            "airports": {
                "csv": "airports.csv",
                "main_key": "Origin",
                "aux_key": "iata",
            }
        },
        "note": "Relational regression with ~3% MISSING TARGETS (cancelled "
        "flights) — load_task drops those rows. Joins origin airport "
        "metadata (lat/long/city). Datetime columns throughout.",
    },
    "movielens": {
        "source": "movielens",
        "csv": "ratings.csv",
        "target": "rating",
        "rename": "rating",
        "drop": ["timestamp"],
        "max_rows": 15_000,
        "metric": "root_mean_squared_error",
        "aux": {
            "movies": {
                "csv": "movies.csv",
                "main_key": "movieId",
                "aux_key": "movieId",
            }
        },
        "note": "Relational: the main table is (userId, movieId, rating) — pure "
        "IDs, so the model must aggregate-join movie titles/genres to "
        "learn anything. Target is a 0.5-5.0 half-star scale (10 "
        "distinct values), which sits right on the regression / "
        "classification inference boundary.",
    },
}


def _is_classification(series: pd.Series) -> bool:
    """Mirror data_summary.infer_task_type so the description matches reality."""
    return not (
        pd.api.types.is_numeric_dtype(series)
        and series.nunique(dropna=True) > 20
    )


def _split(df: pd.DataFrame, target: str, seed: int = SEED):
    """Seeded 80/20 split; stratified when the target is categorical.

    Example:
        train, test = _split(df, "status")   # test keeps the class ratio
    """
    if _is_classification(df[target]):
        test = df.groupby(df[target], group_keys=False).sample(
            frac=0.2, random_state=seed
        )
    else:
        test = df.sample(frac=0.2, random_state=seed)
    return df.drop(index=test.index), test


def _describe(name: str, recipe: dict, train: pd.DataFrame, target: str) -> str:
    task_type = (
        "classification" if _is_classification(train[target]) else "regression"
    )
    aux = recipe.get("aux") or {}
    lines = [
        "# Task",
        "",
        f"Predict the {target}.",
    ]
    if aux:
        joins = "; ".join(
            f"aux_{n}.csv on main.{c['main_key']} = {n}.{c['aux_key']}"
            for n, c in aux.items()
        )
        lines += [
            "",
            f"This is a relational task: aggregate the auxiliary table(s) per "
            f"main row before modelling ({joins}).",
        ]
    lines += [
        "",
        "# Metric",
        "",
        recipe["metric"],
        "",
        "# Dataset",
        "",
        f"{task_type.capitalize()} task ({len(train)} training rows, "
        f"{train.shape[1] - 1} feature columns). train.csv has the features "
        f"and the '{target}' column; test.csv has the features only.",
        "",
        recipe["note"],
    ]
    return "\n".join(lines) + "\n"


def stage(name: str, recipe: dict, force: bool = False) -> str | None:
    """Write one task directory from its recipe. Returns the dir, or None if skipped."""
    out_dir = os.path.join(TASKS_DIR, name)
    if os.path.exists(out_dir) and not force:
        return None

    src = os.path.join(DATA_DIR, recipe["source"])
    df = pd.read_csv(os.path.join(src, recipe["csv"]))
    df = df.drop(columns=recipe.get("drop", []), errors="ignore")

    target = recipe["target"]
    if recipe.get("rename") and recipe["rename"] != target:
        df = df.rename(columns={target: recipe["rename"]})
        target = recipe["rename"]

    max_rows = recipe.get("max_rows")
    if max_rows and len(df) > max_rows:
        if _is_classification(df[target]):
            # keep every class present, proportionally
            df = df.groupby(df[target], group_keys=False).sample(
                frac=max_rows / len(df), random_state=SEED
            )
        else:
            df = df.sample(n=max_rows, random_state=SEED)

    # the target must be LAST: load_task falls back to the last column when the
    # "Predict the <ident>" parse misses
    df = df[[c for c in df.columns if c != target] + [target]]

    train, test = _split(df, target)
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.drop(columns=[target]).to_csv(
        os.path.join(out_dir, "test.csv"), index=False
    )
    # The held-out targets — the ground truth that makes test.csv scoreable.
    # Written in the SAME row order as test.csv (the submission-format contract:
    # one prediction per test row, in order); `row_id` makes that order explicit
    # and lets the scorers assert alignment instead of trusting it.
    #
    # The "answer" in the filename is load-bearing: MLE-STAR's create_workspace
    # copies the task dir into the agent's ./input but skips any file matching
    # "answer", so this stays invisible to the agent that must predict test.csv.
    test_answer = pd.DataFrame(
        {"row_id": range(len(test)), target: test[target].to_numpy()}
    )
    test_answer.to_csv(os.path.join(out_dir, "test_answer.csv"), index=False)

    for aux_name, cfg in (recipe.get("aux") or {}).items():
        aux = pd.read_csv(os.path.join(src, cfg["csv"]))
        # Follow the main table's keys so an aux table can't dwarf it — but the
        # keys of the WHOLE table, train and holdout alike. Filtering to `train`
        # left every holdout row joining to nothing (0/29 covered on
        # country-happiness), so each aux-derived feature came out NaN at predict
        # time and the relational tasks were unscoreable on the shared holdout.
        # This is not leakage: aux tables carry features, never the target, and a
        # holdout row you cannot join is a holdout row you cannot predict.
        aux = aux[aux[cfg["aux_key"]].isin(df[cfg["main_key"]])]
        aux.to_csv(os.path.join(out_dir, f"aux_{aux_name}.csv"), index=False)

    with open(
        os.path.join(out_dir, "task_description.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(_describe(name, recipe, train, target))
    return out_dir


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="stage just these task names")
    parser.add_argument(
        "--force", action="store_true", help="restage tasks that already exist"
    )
    args = parser.parse_args()

    names = args.only or list(RECIPES)
    for name in names:
        if name not in RECIPES:
            raise SystemExit(f"unknown task {name!r}; known: {sorted(RECIPES)}")
        out = stage(name, RECIPES[name], force=args.force)
        if out is None:
            print(f"  {name:<20} exists, skipped (--force to restage)")
            continue
        files = sorted(os.listdir(out))
        rows = len(pd.read_csv(os.path.join(out, "train.csv")))
        print(f"  {name:<20} {rows:>6} train rows  -> {out}  {files}")


if __name__ == "__main__":
    _main()
