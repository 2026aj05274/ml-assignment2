"""Quick sanity check that saved models score bundled test_data.csv."""

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load test dataset
test_data = pd.read_csv("test_data.csv")

# Separate input features and actual target
X_test = test_data.drop(columns=["is_spam"])
y_test = test_data["is_spam"]

# Saved model filenames
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

print("Verifying all five saved models...\n")

# Load and test each model
for model_name, filename in model_files.items():

    model_path = os.path.join("model", filename)

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{model_name}:")
    print(f"  Predictions generated: {len(predictions)}")
    print(f"  Test accuracy: {accuracy:.4f}")
    print("  Status: SUCCESS\n")

print("All five models verified successfully!")
