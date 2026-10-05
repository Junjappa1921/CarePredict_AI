from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "readmission_model.pkl")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard")

# Create Flask application
app = Flask(__name__)
CORS(app)

# Load trained AI model
model = joblib.load(MODEL_PATH)


# Serve dashboard
@app.route("/")
def home():
    return send_from_directory(DASHBOARD_PATH, "index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    patient_data = [[
        data["Age"],
        data["Previous_Admissions"],
        data["Previous_Stay_Days"],
        data["Heart_Rate"],
        data["SpO2"],
        data["Temperature"],
        data["Systolic_BP"],
        data["Diabetes"],
        data["Hypertension"]
    ]]

    patient_data = pd.DataFrame(
        patient_data,
        columns=[
            "Age",
            "Previous_Admissions",
            "Previous_Stay_Days",
            "Heart_Rate",
            "SpO2",
            "Temperature",
            "Systolic_BP",
            "Diabetes",
            "Hypertension"
        ]
    )

    # AI prediction
    risk_probability = model.predict_proba(patient_data)[0][1]

    risk_percentage = round(
        risk_probability * 100,
        2
    )

    # Risk level
    if risk_percentage < 40:
        risk_level = "LOW"
    elif risk_percentage < 70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return jsonify({
        "risk_percentage": risk_percentage,
        "risk_level": risk_level
    })


# Start server
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )