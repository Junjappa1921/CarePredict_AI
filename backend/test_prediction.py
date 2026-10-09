import joblib
import pandas as pd


# Load the CarePredict AI model
model = joblib.load("readmission_model.pkl")


# Test patient
patient = pd.DataFrame([{
    "age": "[50-60)",
    "time_in_hospital": 5,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 12,
    "number_outpatient": 1,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 6,
    "A1Cresult": "Norm",
    "diabetesMed": "Yes"
}])


# Get prediction probability
risk_probability = model.predict_proba(patient)[0][1]

risk_percentage = round(risk_probability * 100, 2)


# Determine risk level
if risk_percentage < 40:
    risk_level = "LOW"

elif risk_percentage < 70:
    risk_level = "MEDIUM"

else:
    risk_level = "HIGH"


# Display result
print()
print("CarePredict AI")
print("----------------------------")
print("Patient Prediction")
print("Risk Percentage:", risk_percentage, "%")
print("Risk Level:", risk_level)
print("----------------------------")