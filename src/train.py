import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load Dataset

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target, name="target")


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train Logistic Regression

model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)


# Evaluate Model

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy : {accuracy:.4f}")


# Save Model

joblib.dump(model, "models/model.pkl")

print("Model Saved Successfully")


# Create Reference Dataset

reference_data = X.copy()

reference_data["target"] = y

reference_data.to_csv(
    "data/reference.csv",
    index=False
)

print("Reference Dataset Saved")