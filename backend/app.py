from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd


# Create Flask application
app = Flask(__name__)

# Allow dashboard to communicate with backend
CORS(app)


# Load trained AI model
model = joblib.load("../readmission_model.pkl")


# Home route
@app.route("/")
def home():

    return "CarePredict AI Backend is Running!"


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # Receive patient data
    data = request.get_json()


    # Get patient information
    age = data["Age"]

    previous_admissions = data["Previous_Admissions"]

    previous_stay_days = data["Previous_Stay_Days"]

    heart_rate = data["Heart_Rate"]

    spo2 = data["SpO2"]

    temperature = data["Temperature"]

    systolic_bp = data["Systolic_BP"]

    diabetes = data["Diabetes"]

    hypertension = data["Hypertension"]


    # Prepare patient data
    patient_data = [[

        age,

        previous_admissions,

        previous_stay_days,

        heart_rate,

        spo2,

        temperature,

        systolic_bp,

        diabetes,

        hypertension

    ]]


    # Give feature names to the data
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


    # Calculate readmission risk
    risk_probability = model.predict_proba(
        patient_data
    )[0][1]


    # Convert probability to percentage
    risk_percentage = round(
        risk_probability * 100,
        2
    )


    # Determine risk level

    if risk_percentage < 40:

        risk_level = "LOW"

    elif risk_percentage < 70:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"


    # Send result to dashboard
    return jsonify({

        "risk_percentage":
            risk_percentage,

        "risk_level":
            risk_level

    })


# Start Flask server
if __name__ == "__main__":

    app.run(
        debug=True
    )