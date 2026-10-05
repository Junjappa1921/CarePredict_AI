import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load patient dataset
data = pd.read_csv("dataset/patients.csv")

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

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create AI model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("CarePredict AI Model")
print("--------------------")
print("Model training completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save trained model
joblib.dump(model, "readmission_model.pkl")

print("Model saved as readmission_model.pkl")