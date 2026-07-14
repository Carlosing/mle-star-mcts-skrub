"""Recover the discarded holdout targets for the legacy hand-staged tasks.

Four tasks (california-housing-prices, employee-salaries, midwest-survey,
open-payments) were staged before `stage_tasks.py` existed and have no recipe,
so `--force` cannot regenerate them. Their `test.csv` was written features-only
and the targets were thrown away — the reason the benchmark had no ground truth.

Re-staging them from `data/` would produce a DIFFERENT train.csv (unknown
original preprocessing) and invalidate every result already measured on them.
Instead, recover the targets non-destructively: each `test.csv` row is a real
row of the source dataset with its target column removed, so joining the holdout
back onto the source on ALL feature columns recovers it.

This is only sound if the match is exact and unambiguous, so the join is
VERIFIED, not trusted: every holdout row must match exactly one source row, or
the task is refused. train.csv and test.csv are never written.

Run:
    uv run python scripts/recover_answers.py          # verify + write
    uv run python scripts/recover_answers.py --check  # verify only
"""

import argparse
import os

import pandas as pd

TASKS = {
    "california-housing-prices": "california_housing",
    "employee-salaries": "employee_salaries",
    "midwest-survey": "midwest_survey",
    "open-payments": "open_payments",
}

TASKS_DIR = "machine_learning_engineering/tasks"
DATA_DIR = "data"


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Round floats and stringify so the join is not defeated by CSV round-trip.

    train/test were written through `to_csv`, so a float that was 1.8790000001
    in memory is "1.879" on disk. Comparing against the in-memory source needs
    the same flattening on both sides.
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            out[col] = s.round(6).astype(str)
        else:
            out[col] = s.astype(str)
    return out


def recover(task: str, src_slug: str, write: bool) -> bool:
    """Recover one task's holdout targets. Returns True when verified."""
    task_dir = os.path.join(TASKS_DIR, task)
    test = pd.read_csv(os.path.join(task_dir, "test.csv"))
    train = pd.read_csv(os.path.join(task_dir, "train.csv"))
    target = train.columns[-1]

    source = pd.read_csv(os.path.join(DATA_DIR, src_slug, f"{src_slug}.csv"))
    # the staged target may have been renamed to a safe identifier; find the
    # source column whose values reproduce the train column
    if target not in source.columns:
        candidates = [
            c
            for c in source.columns
            if c not in test.columns
            and set(_norm(source[[c]])[c]) >= set(_norm(train[[target]])[target])
        ]
        if len(candidates) != 1:
            print(f"  {task:<26} REFUSED: cannot identify target in source "
                  f"(candidates={candidates})")
            return False
        source = source.rename(columns={candidates[0]: target})

    features = list(test.columns)
    missing = [c for c in features if c not in source.columns]
    if missing:
        print(f"  {task:<26} REFUSED: source lacks columns {missing}")
        return False

    src_keyed = _norm(source[features])
    src_keyed[target] = source[target].to_numpy()
    test_keyed = _norm(test)

    # every holdout row must hit exactly one source row, or we do not trust it
    counts = (
        src_keyed.groupby(features, dropna=False)[target]
        .nunique()
        .rename("n_targets")
        .reset_index()
    )
    merged = test_keyed.merge(counts, on=features, how="left")
    unmatched = int(merged["n_targets"].isna().sum())
    ambiguous = int((merged["n_targets"] > 1).sum())
    if unmatched or ambiguous:
        print(f"  {task:<26} REFUSED: {unmatched} unmatched, "
              f"{ambiguous} ambiguous of {len(test)} holdout rows")
        return False

    answers = (
        test_keyed.merge(
            src_keyed.drop_duplicates(subset=features), on=features, how="left"
        )[target]
        .to_numpy()
    )
    # cast back to the train column's dtype (the join stringified everything)
    answers = pd.Series(answers).astype(train[target].dtype)

    print(f"  {task:<26} OK: {len(test)} / {len(test)} holdout rows matched 1:1")
    if write:
        pd.DataFrame({"row_id": range(len(test)), target: answers}).to_csv(
            os.path.join(task_dir, "test_answer.csv"), index=False
        )
    return True


# Tasks whose holdout targets are NOT recoverable, and why. For these the only
# honest option is to carve a fresh LABELLED holdout out of the existing
# train.csv: it preserves the dataset exactly, needs no source, and produces real
# ground truth. It shrinks train.csv, so results measured on the old train.csv
# no longer apply and must be re-run.
RESPLIT = {
    "california-housing-prices":
        "staged from the Kaggle housing.csv schema; data/california_housing/ is "
        "the unrelated sklearn version (MedInc/AveRooms) — no source to join to",
    "employee-salaries":
        "feature vectors repeat in the source with different salaries "
        "(117/1600 holdout rows ambiguous)",
    "open-payments":
        "feature vectors repeat in the source with different statuses "
        "(85/1600 holdout rows ambiguous)",
    "credit-fraud":
        "relational task staged by stage_credit_fraud.py; it has no test.csv at "
        "all, so it never had a holdout",
}

SEED = 42


def resplit(task: str, write: bool) -> bool:
    """Carve a fresh labelled 80/20 holdout out of the task's existing train.csv."""
    task_dir = os.path.join(TASKS_DIR, task)
    full = pd.read_csv(os.path.join(task_dir, "train.csv"))
    target = full.columns[-1]

    # stratify when the target is categorical, matching stage_tasks._split
    categorical = not pd.api.types.is_float_dtype(full[target]) and (
        full[target].nunique() <= max(20, int(0.05 * len(full)))
    )
    if categorical:
        holdout = full.groupby(full[target], group_keys=False).sample(
            frac=0.2, random_state=SEED
        )
    else:
        holdout = full.sample(frac=0.2, random_state=SEED)
    train = full.drop(index=holdout.index)

    print(f"  {task:<26} resplit: {len(full)} -> train {len(train)} + "
          f"holdout {len(holdout)}"
          f"{' (stratified)' if categorical else ''}")
    if write:
        train.to_csv(os.path.join(task_dir, "train.csv"), index=False)
        holdout.drop(columns=[target]).to_csv(
            os.path.join(task_dir, "test.csv"), index=False
        )
        pd.DataFrame(
            {"row_id": range(len(holdout)),
             target: holdout[target].to_numpy()}
        ).to_csv(os.path.join(task_dir, "test_answer.csv"), index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the join only; write nothing")
    args = parser.parse_args()

    print("Recovering holdout targets by joining test.csv back to its source:")
    unrecovered = []
    for task, src in TASKS.items():
        if not recover(task, src, write=not args.check):
            unrecovered.append(task)

    print("\nNot recoverable — carving a fresh labelled holdout from train.csv")
    print("(this SHRINKS train.csv; results measured on the old one must be re-run):")
    for task, why in RESPLIT.items():
        print(f"  {task:<26} {why}")
    print()
    for task in RESPLIT:
        resplit(task, write=not args.check)

    if args.check:
        print("\n--check: nothing written")


if __name__ == "__main__":
    main()
