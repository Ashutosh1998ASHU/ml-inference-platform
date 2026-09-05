from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUM_CUSTOMERS = 1000


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    tenure = rng.integers(1, 73, size=NUM_CUSTOMERS)

    monthly_charges = np.round(
        rng.uniform(30, 120, size=NUM_CUSTOMERS),
        2,
    )

    contract_type = rng.choice(
        ["monthly", "annual"],
        size=NUM_CUSTOMERS,
        p=[0.65, 0.35],
    )

    support_tickets = rng.poisson(
        lam=3,
        size=NUM_CUSTOMERS,
    )

    usage_score = np.round(
        rng.uniform(20, 100, size=NUM_CUSTOMERS),
        2,
    )

    churn_score = (
        1.5
        - 0.035 * tenure
        + 0.018 * monthly_charges
        + 0.18 * support_tickets
        - 0.025 * usage_score
        + np.where(contract_type == "monthly", 0.9, -0.4)
        + rng.normal(0, 0.5, size=NUM_CUSTOMERS)
    )

    churn_probability = 1 / (1 + np.exp(-churn_score))

    churn = rng.binomial(
        1,
        churn_probability,
    )

    return pd.DataFrame(
        {
            "customer_id": [
                f"C{i:04d}" for i in range(1, NUM_CUSTOMERS + 1)
            ],
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "support_tickets": support_tickets,
            "usage_score": usage_score,
            "churn": churn,
        }
    )


def main() -> None:
    output_path = Path("ml/data/churn.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = generate_dataset()

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Dataset generated: {output_path}")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print("\nChurn distribution:")
    print(dataframe["churn"].value_counts())
    print("\nSample:")
    print(dataframe.head())


if __name__ == "__main__":
    main()