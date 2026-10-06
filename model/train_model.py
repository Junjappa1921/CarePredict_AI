import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("CarePredict AI Model")
print("--------------------")

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "patients.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "readmission_model.pkl"
)

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Input features
features = [
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

X = data[features]
y = data["Readmitted"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create AI pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            C=0.3,
            max_iter=2000
        )
    )
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Model training completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save model
joblib.dump(
    model,
    MODEL_PATH
)

print("Model saved as readmission_model.pkl")