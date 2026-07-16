"""Stage skrub's credit-fraud dataset as a relational task under tasks/.

Downloads `fetch_credit_fraud` (baskets + products), takes a seeded subsample
of baskets (CV rollouts must stay fast) with their matching products, and
writes the multi-table task layout `load_task` expects:

    tasks/credit-fraud/train.csv          <- baskets (main table, target)
    tasks/credit-fraud/aux_products.csv   <- products (auxiliary table)
    tasks/credit-fraud/task_description.txt

Run once, online:  uv run python scripts/stage_credit_fraud.py

Example:
    $ uv run python scripts/stage_credit_fraud.py
    Staged 15000 baskets + 26389 products -> machine_learning_engineering/tasks/credit-fraud
"""

import os

from skrub.datasets import fetch_credit_fraud

N_BASKETS = 15_000
SEED = 42
OUT_DIR = os.path.join("machine_learning_engineering", "tasks", "credit-fraud")

DESCRIPTION = """# Task

Predict the fraud_flag of each basket (binary classification). The main table
lists baskets; the auxiliary table lists the products each basket contains
(join on baskets.ID = products.basket_ID) and must be aggregated per basket.

# Metric

roc_auc
"""


def main() -> None:
    bunch = fetch_credit_fraud()
    baskets, products = bunch.baskets, bunch.products

    target = (
        "fraud_flag" if "fraud_flag" in baskets.columns else baskets.columns[-1]
    )
    # Keep the class balance representative: plain seeded row sample.
    small = baskets.sample(n=min(N_BASKETS, len(baskets)), random_state=SEED)
    small_products = products[products["basket_ID"].isin(small["ID"])]

    os.makedirs(OUT_DIR, exist_ok=True)
    small.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    small_products.to_csv(
        os.path.join(OUT_DIR, "aux_products.csv"), index=False
    )
    with open(
        os.path.join(OUT_DIR, "task_description.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(DESCRIPTION)

    print(
        f"Staged {len(small)} baskets + {len(small_products)} products -> {OUT_DIR} "
        f"(target={target!r}, fraud rate={small[target].mean():.3%})"
    )


if __name__ == "__main__":
    main()
