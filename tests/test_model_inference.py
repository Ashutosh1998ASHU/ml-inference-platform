from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("ml/artifacts/churn_model.joblib")


def test_saved_model_can_predict():
    assert MODEL_PATH.exists()

    model = joblib.load(MODEL_PATH)

    customer = pd.DataFrame(
        [
            {
                "tenure": 6,
                "monthly_charges": 95.0,
                "contract_type": "monthly",
                "support_tickets": 5,
                "usage_score": 35.0,
            }
        ]
    )

    prediction = model.predict(customer)
    probability = model.predict_proba(customer)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]

    assert probability.shape == (1, 2)
    assert 0.0 <= probability[0][1] <= 1.0