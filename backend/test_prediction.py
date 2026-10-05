import requests

patient = {
    "Age": 45,
    "Previous_Admissions": 0,
    "Previous_Stay_Days": 3,
    "Heart_Rate": 76,
    "SpO2": 98,
    "Temperature": 36.6,
    "Systolic_BP": 120,
    "Diabetes": 0,
    "Hypertension": 0
}

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=patient
)

print(response.json())