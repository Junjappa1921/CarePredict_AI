import pandas as pd
import numpy as np

# Fixed seed so the same dataset is generated every time
np.random.seed(42)

patients = []

for i in range(300):

    # Generate patient information
    age = np.random.randint(20, 86)

    previous_admissions = np.random.randint(0, 6)

    previous_stay = np.random.randint(2, 13)

    heart_rate = np.random.randint(65, 121)

    spo2 = np.random.randint(87, 100)

    temperature = round(
        np.random.uniform(36.3, 39.0), 1
    )

    systolic_bp = np.random.randint(110, 166)

    diabetes = np.random.randint(0, 2)

    hypertension = np.random.randint(0, 2)

    # Calculate a risk score
    risk_score = 0

    # Age contribution
    if age >= 65:
        risk_score += 2
    elif age >= 50:
        risk_score += 1

    # Previous admissions
    risk_score += previous_admissions * 0.5

    # Previous hospital stay
    if previous_stay >= 8:
        risk_score += 2
    elif previous_stay >= 5:
        risk_score += 1

    # Heart rate
    if heart_rate >= 105:
        risk_score += 2
    elif heart_rate >= 90:
        risk_score += 1

    # SpO2
    if spo2 <= 90:
        risk_score += 3
    elif spo2 <= 94:
        risk_score += 2
    elif spo2 <= 96:
        risk_score += 1

    # Temperature
    if temperature >= 38.2:
        risk_score += 2
    elif temperature >= 37.5:
        risk_score += 1

    # Blood pressure
    if systolic_bp >= 150:
        risk_score += 2
    elif systolic_bp >= 135:
        risk_score += 1

    # Medical conditions
    risk_score += diabetes * 1
    risk_score += hypertension * 1

    # Add random variation
    risk_score += np.random.normal(0, 1.5)

    # Convert score into probability
    probability = 1 / (
        1 + np.exp(-(risk_score - 6) / 2)
    )

    # Generate readmission outcome
    readmitted = np.random.binomial(
        1,
        probability
    )

    patients.append([
        age,
        previous_admissions,
        previous_stay,
        heart_rate,
        spo2,
        temperature,
        systolic_bp,
        diabetes,
        hypertension,
        readmitted
    ])


# Create DataFrame
columns = [
    "Age",
    "Previous_Admissions",
    "Previous_Stay_Days",
    "Heart_Rate",
    "SpO2",
    "Temperature",
    "Systolic_BP",
    "Diabetes",
    "Hypertension",
    "Readmitted"
]

data = pd.DataFrame(
    patients,
    columns=columns
)

# Save dataset
data.to_csv(
    "dataset/patients.csv",
    index=False
)

print("New synthetic dataset created!")
print("Total patients:", len(data))

print("\nReadmission distribution:")
print(data["Readmitted"].value_counts())

print("\nDataset saved to:")
print("dataset/patients.csv")