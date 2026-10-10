import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS


# ============================================================
# 1. CREATE FLASK APPLICATION AND CONFIGURE CORS
# ============================================================

app = Flask(__name__)

DASHBOARD_ORIGIN = (
    "https://carepredict-ai-v2-dashboard.onrender.com"
)

CORS(
    app,
    resources={
        r"/*": {
            "origins": [DASHBOARD_ORIGIN]
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=False,
    max_age=600
)


# ============================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "readmission_model_improved.pkl"
)

model = None
model_bundle = {}

FEATURES = []
NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = []

MODEL_NAME = "RandomForest"
MODEL_THRESHOLD = 0.40
MODEL_LOAD_ERROR = None


try:
    model_bundle = joblib.load(MODEL_PATH)

    if not isinstance(model_bundle, dict):
        raise ValueError(
            "The saved model must contain a dictionary bundle."
        )

    model = model_bundle["model"]

    FEATURES = list(model_bundle["features"])

    NUMERIC_FEATURES = list(
        model_bundle.get("numeric_features", [])
    )

    CATEGORICAL_FEATURES = list(
        model_bundle.get("categorical_features", [])
    )

    MODEL_NAME = model_bundle.get(
        "best_model_name",
        "RandomForest"
    )

    MODEL_THRESHOLD = float(
        model_bundle.get("threshold", 0.40)
    )

    print("Model loaded successfully.")
    print("Model name:", MODEL_NAME)
    print("Model path:", MODEL_PATH)
    print("Number of required features:", len(FEATURES))

except Exception as error:
    MODEL_LOAD_ERROR = str(error)

    print("MODEL LOADING FAILED:")
    print(MODEL_LOAD_ERROR)


# ============================================================
# 3. VALID CATEGORICAL VALUES
# ============================================================

VALID_CATEGORIES = {
    "age": [
        "[0-10)",
        "[10-20)",
        "[20-30)",
        "[30-40)",
        "[40-50)",
        "[50-60)",
        "[60-70)",
        "[70-80)",
        "[80-90)",
        "[90-100)"
    ],

    "A1Cresult": [
        "None",
        "Norm",
        ">7",
        ">8"
    ],

    "diabetesMed": [
        "Yes",
        "No"
    ],

    "max_glu_serum": [
        "None",
        "Norm",
        ">200",
        ">300"
    ]
}


# ============================================================
# 4. CLASSIFY DISPLAY RISK
# ============================================================

def classify_risk(risk_percentage):
    """
    Demonstration display categories.

    LOW:      0% to 30%
    MODERATE: Above 30% to 55%
    HIGH:     Above 55% to 100%

    These categories are not clinically validated.
    """

    if risk_percentage <= 30:
        return "LOW"

    elif risk_percentage <= 55:
        return "MODERATE"

    return "HIGH"


# ============================================================
# 5. GET READMISSION PROBABILITY
# ============================================================

def get_probability(input_data):
    """
    Get the model probability for the positive class.

    Confirm that the positive class represents readmission
    within 30 days in the saved model.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba."
        )

    probabilities = model.predict_proba(input_data)[0]

    if hasattr(model, "classes_"):
        classes = list(model.classes_)

        positive_labels = [
            1,
            "1",
            True,
            "true",
            "True",
            "yes",
            "Yes",
            "readmitted",
            "<30"
        ]

        for label in positive_labels:
            if label in classes:
                index = classes.index(label)
                return float(probabilities[index])

    if len(probabilities) == 2:
        return float(probabilities[1])

    raise ValueError(
        "Unable to identify the positive readmission class."
    )


# ============================================================
# 6. HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "CarePredict AI",
        "message": "Backend is ready.",
        "status": "online",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "clinical_status": "Educational prototype only"
    }), 200


# ============================================================
# 7. HEALTH CHECK ROUTE
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    response = {
        "status": "online" if model is not None else "error",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "features": FEATURES,
        "threshold": MODEL_THRESHOLD
    }

    if MODEL_LOAD_ERROR:
        response["error"] = MODEL_LOAD_ERROR

    return jsonify(response), (
        200 if model is not None else 500
    )


# ============================================================
# 8. PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Check model availability.
        if model is None:
            return jsonify({
                "status": "error",
                "message": "The trained model could not be loaded.",
                "details": MODEL_LOAD_ERROR
            }), 500

        # Read JSON request.
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Please send patient details as JSON."
            }), 400

        # Accept either:
        # {"features": {...}}
        # or a direct feature dictionary.

        if isinstance(data.get("features"), dict):
            patient_data = data["features"]
        else:
            patient_data = data

        # Check for missing features.
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in patient_data
        ]

        if missing_features:
            return jsonify({
                "status": "error",
                "message": "Some required patient details are missing.",
                "missing_features": missing_features,
                "required_features": FEATURES
            }), 400

        cleaned_data = {}

        # Validate numeric features.
        for feature in NUMERIC_FEATURES:

            value = patient_data[feature]

            try:
                numeric_value = float(value)

            except (TypeError, ValueError):
                return jsonify({
                    "status": "error",
                    "message": (
                        "Invalid numeric value for " + feature
                    )
                }), 400

            if not np.isfinite(numeric_value):
                return jsonify({
                    "status": "error",
                    "message": (
                        "Numeric values must be finite for "
                        + feature
                    )
                }), 400

            cleaned_data[feature] = numeric_value

        # Validate categorical features.
        for feature in CATEGORICAL_FEATURES:

            value = patient_data[feature]

            if value is None:
                value = "None"
            else:
                value = str(value).strip()

            allowed_values = VALID_CATEGORIES.get(feature)

            if (
                allowed_values is not None
                and value not in allowed_values
            ):
                return jsonify({
                    "status": "error",
                    "message": (
                        "Invalid category for " + feature
                    ),
                    "allowed_values": allowed_values
                }), 400

            cleaned_data[feature] = value

        # Preserve the training feature order.
        input_data = pd.DataFrame(
            [
                {
                    feature: cleaned_data[feature]
                    for feature in FEATURES
                }
            ],
            columns=FEATURES
        )

        # Generate prediction probability.
        probability = get_probability(input_data)

        probability = max(
            0.0,
            min(1.0, probability)
        )

        risk_percentage = round(
            probability * 100,
            2
        )

        risk_level = classify_risk(risk_percentage)

        predicted_readmission = (
            probability >= MODEL_THRESHOLD
        )

        return jsonify({
            "status": "success",
            "model_name": MODEL_NAME,
            "risk_percentage": risk_percentage,
            "risk_level": risk_level,
            "predicted_readmission": bool(
                predicted_readmission
            ),
            "decision_threshold": MODEL_THRESHOLD,
            "required_features": FEATURES,
            "clinical_status": "Educational prototype only",
            "message": (
                "This result is an educational model estimate, "
                "not a medical diagnosis."
            )
        }), 200

    except Exception as error:

        app.logger.exception("Prediction failed.")

        return jsonify({
            "status": "error",
            "message": "The prediction could not be completed.",
            "details": str(error)
        }), 500


# ============================================================
# 9. START THE BACKEND
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5001))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )