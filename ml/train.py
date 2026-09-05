from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import pandas as pd
import joblib

DATASET_PATH = Path("ml/data/churn.csv")
ARTIFACT_DIR = Path("ml/artifacts")
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"

FEATURES = [
    "tenure",
    "monthly_charges",
    "contract_type",
    "support_tickets",
    "usage_score",
]

TARGET = "churn"

NUMERICAL_FEATURES = [
    "tenure",
    "monthly_charges",
    "support_tickets",
    "usage_score",
]

CATEGORICAL_FEATURES = [
    "contract_type",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    required_columns = set(FEATURES + [TARGET])

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    return dataframe

def split_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X = dataframe[FEATURES]
    y = dataframe[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

def build_model_pipeline() -> Pipeline:
    preprocessor  = build_preprocessor()

    classifier = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

def train_model(
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:

    model_pipeline.fit(X_train, y_train)

    return model_pipeline

def evaluate_model(
    model_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:

    predictions = model_pipeline.predict(X_test)
    probabilities = model_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print("\nEvaluation Results")
    print("------------------")

    for name, value in metrics.items():
        print(f"{name:10s}: {value:.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(y_test, predictions))

    return metrics

def save_model(model_pipeline: Pipeline) -> None:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_pipeline,
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")

def main() -> None:
    print("Loading dataset...")

    dataframe = load_dataset()

    print(f"Dataset shape: {dataframe.shape}")

    X_train, X_test, y_train, y_test = split_dataset(
        dataframe
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    model_pipeline = build_model_pipeline()

    print("\nTraining model...")

    model_pipeline = train_model(
        model_pipeline,
        X_train,
        y_train,
    )

    evaluate_model(
        model_pipeline,
        X_test,
        y_test,
    )

    save_model(model_pipeline)


if __name__ == "__main__":
    main()