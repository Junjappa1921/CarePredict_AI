
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import math

# ==========================================
# CarePredict AI - Flask Backend
# ==========================================

app = Flask(__name__)
CORS(app)

# ==========================================
# Load trained model bundle
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "readmission_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model_bundle = joblib.load(MODEL_PATH)

# The training script saves a dictionary containing
# the model, features, and selected threshold.
if not isinstance(model_bundle, dict) or "model" not in model_bundle:
    raise ValueError(
        "The model file does not contain the expected model bundle. "
        "Run python model\\train_model.py again."
    )

model = model_bundle["model"]

REQUIRED_FEATURES = model_bundle["features"]

NUMERIC_FEATURES = model_bundle["numeric_features"]

CATEGORICAL_FEATURES = model_bundle["categorical_features"]

DECISION_THRESHOLD = float(model_bundle["threshold"])

print("CarePredict AI Model Loaded Successfully")
print("Selected model:", model_bundle.get("best_model_name"))
print("Decision threshold:", DECISION_THRESHOLD)
print("Features:", REQUIRED_FEATURES)


# ==========================================
# Input validation
# ==========================================

def validate_numeric(data, feature):
    try:
        value = float(data[feature])

        if not math.isfinite(value):
            raise ValueError

        if value < 0:
            raise ValueError

        return value

    except (TypeError, ValueError, KeyError):
        raise ValueError(
            f"{feature} must be a valid non-negative number."
        )


# ==========================================
# Home route
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "CarePredict AI Backend is Running",
        "project": "Hospital Readmission Risk Prediction",
        "clinical_status": "Educational prototype only"
    })


# ==========================================
# Health check
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "model_name": model_bundle.get("best_model_name"),
        "features": REQUIRED_FEATURES,
        "threshold": DECISION_THRESHOLD
    })


# ==========================================
# Prediction route
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "error": "Send patient information as a JSON object."
            }), 400

        # Check that all model features are provided.
        missing_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "status": "error",
                "error": "Some required model features are missing.",
                "missing_features": missing_features
            }), 400

        patient_data = {}

        # Validate numerical features.
        for feature in NUMERIC_FEATURES:
            patient_data[feature] = validate_numeric(
                data,
                feature
            )

        # Validate categorical features.
        allowed_categories = {
            "age": [
                "[0-10)", "[10-20)", "[20-30)",
                "[30-40)", "[40-50)", "[50-60)",
                "[60-70)", "[70-80)", "[80-90)",
                "[90-100)"
            ],
            "A1Cresult": [
                "None", "Norm", ">7", ">8"
            ],
            "diabetesMed": [
                "Yes", "No"
            ],
            "max_glu_serum": [
                "None", "Norm", ">200", ">300"
            ]
        }

        for feature in CATEGORICAL_FEATURES:
            value = str(data[feature]).strip()

            if value not in allowed_categories.get(feature, []):
                return jsonify({
                    "status": "error",
                    "error": (
                        f"Invalid value for {feature}: {value}. "
                        f"Allowed values: "
                        f"{allowed_categories.get(feature, [])}"
                    )
                }), 400

            patient_data[feature] = value

        # Create the DataFrame in the model's expected order.
        patient = pd.DataFrame(
            [patient_data],
            columns=REQUIRED_FEATURES
        )

        # ==========================================
        # Generate prediction
        # ==========================================

        probabilities = model.predict_proba(patient)[0]

        classes = list(model.classes_)

        if 1 not in classes:
            return jsonify({
                "status": "error",
                "error": "Positive readmission class is missing."
            }), 500

        positive_index = classes.index(1)

        risk_probability = float(
            probabilities[positive_index]
        )

        risk_percentage = round(
            risk_probability * 100,
            2
        )

        predicted_readmission = (
            risk_probability >= DECISION_THRESHOLD
        )

        # These labels are demonstration indicators only.
        risk_level = (
            "ELEVATED"
            if predicted_readmission
            else "LOWER"
        )

        print("\nCarePredict AI Prediction")
        print("----------------------------")
        print("Model:", model_bundle.get("best_model_name"))
        print("Model score:", risk_percentage, "%")
        print("Threshold:", DECISION_THRESHOLD)
        print("Demo indicator:", risk_level)
        print("----------------------------")

        return jsonify({
            "status": "success",
            "risk_percentage": risk_percentage,
            "decision_threshold": DECISION_THRESHOLD,
            "predicted_readmission": bool(
                predicted_readmission
            ),
            "risk_level": risk_level,
            "glucose_feature_used": "max_glu_serum",
            "glucose_note": (
                "The model uses a categorical serum-glucose "
                "dataset feature, not the patient's exact "
                "glucometer reading in mg/dL."
            ),
            "message": (
                "Educational model output only. "
                "Not validated for clinical decisions."
            )
        })

    except ValueError as error:
        return jsonify({
            "status": "error",
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "status": "error",
            "error": (
                "The prediction could not be completed. "
                "Check the backend terminal for details."
            )
        }), 500


# ==========================================
# Start server
# ==========================================

if __name__ == "__main__":

    print("\n===================================")
    print("       CarePredict AI Backend")
    print("===================================")
    print("Model path:", MODEL_PATH)
    print("Decision threshold:", DECISION_THRESHOLD)
    print("Local URL: http://127.0.0.1:5000")
    print("===================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

