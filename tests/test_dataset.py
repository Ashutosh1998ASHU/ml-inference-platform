from pathlib import Path

import pandas as pd


DATASET_PATH = Path("ml/data/churn.csv")

EXPECTED_COLUMNS = {
    "customer_id",
    "tenure",
    "monthly_charges",
    "contract_type",
    "support_tickets",
    "usage_score",
    "churn",
}


def test_dataset_exists():
    assert DATASET_PATH.exists()


def test_dataset_schema():
    dataframe = pd.read_csv(DATASET_PATH)

    assert set(dataframe.columns) == EXPECTED_COLUMNS


def test_dataset_has_both_target_classes():
    dataframe = pd.read_csv(DATASET_PATH)

    assert set(dataframe["churn"].unique()) == {0, 1}


def test_dataset_has_expected_size():
    dataframe = pd.read_csv(DATASET_PATH)

    assert len(dataframe) == 1000