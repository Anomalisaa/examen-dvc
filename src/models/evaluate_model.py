#!/usr/bin/env python3

import os
import json
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def main():
    processed_dir = os.path.join("data", "processed")
    metrics_dir = os.path.join("metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Load the test data
    X_test_scaled_path = os.path.join(processed_dir, "X_test_scaled.csv")
    y_test_path = os.path.join(processed_dir, "y_test.csv")
    X_test = pd.read_csv(X_test_scaled_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    # TTrain the model
    models_dir = os.path.join("models")
    model_path = os.path.join(models_dir, "trained_model.joblib")
    model = joblib.load(model_path)

    # Predict the test data
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    scores = {"MSE": mse, "R2": r2}
    print("Evaluation metrics:", scores)

    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions, columns=["predictions"])
    predictions_path = os.path.join("data", "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    # Save evaluation scores to JSON
    scores_path = os.path.join(metrics_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=4)

    print("Predictions saved to", predictions_path)
    print("Evaluation scores saved to", scores_path)

if __name__ == "__main__":
    main()
