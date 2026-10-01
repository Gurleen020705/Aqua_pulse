import joblib
import pandas as pd
from pathlib import Path


MODEL_FILE = Path(__file__).resolve().parent / "GEMStat_RandomForest_v2.pkl"


# Load model once
model = joblib.load(MODEL_FILE)


FEATURES = [
    "DO",
    "Temperature",
    "pH",
    "Nitrate",
    "Phosphorus"
]


def predict_water_quality(
    do,
    temperature,
    ph,
    nitrate,
    phosphorus
):

    # Create input
    input_data = pd.DataFrame(
        [[
            do,
            temperature,
            ph,
            nitrate,
            phosphorus
        ]],
        columns=FEATURES
    )

    # Prediction
    prediction = model.predict(
        input_data
    )[0]

    # Probabilities
    probabilities = model.predict_proba(
        input_data
    )[0]

    # Find probability corresponding
    # to At-Risk class (Label = 0)
    at_risk_index = list(
        model.classes_
    ).index(0)

    healthy_index = list(
        model.classes_
    ).index(1)

    at_risk_probability = (
        probabilities[at_risk_index]
    )

    healthy_probability = (
        probabilities[healthy_index]
    )

    if prediction == 0:
        result = "At-Risk"
    else:
        result = "Healthy"

    return {
        "prediction": result,
        "at_risk_probability": round(
            float(at_risk_probability),
            4
        ),
        "healthy_probability": round(
            float(healthy_probability),
            4
        )
    }
