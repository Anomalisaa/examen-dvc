#!/usr/bin/env python3

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def main():
    processed_dir = os.path.join("data", "processed")
    X_train_scaled_path = os.path.join(processed_dir, "X_train_scaled.csv")
    y_train_path = os.path.join(processed_dir, "y_train.csv")

    # Load the training data
    X_train = pd.read_csv(X_train_scaled_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # Convert DataFrame to Series

    # Define models and their parameter grids
    models = {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        }
    }

    best_models = {}

    # Iterate over each model and perform grid search
    for model_name, model_info in models.items():
        print(f"Performing GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(
            estimator=model_info["model"],
            param_grid=model_info["param_grid"],
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best parameters for {model_name}: {best_params}")
        best_models[model_name] = best_params

    # Save the best parameters for all models in one pickle file
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)
    best_params_path = os.path.join(models_dir, "best_params.pkl")
    with open(best_params_path, "wb") as f:
        pickle.dump(best_models, f)

    print("Best parameters for all models saved to", best_params_path)

if __name__ == "__main__":
    main()
