#!/usr/bin/env python3

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def main():
    # Define directories and file paths
    processed_dir = os.path.join("data", "processed")
    X_train_scaled_path = os.path.join(processed_dir, "X_train_scaled.csv")
    y_train_path = os.path.join(processed_dir, "y_train.csv")
    
    # Load the training data
    X_train = pd.read_csv(X_train_scaled_path)
    y_train = pd.read_csv(y_train_path).squeeze()  # Convert DataFrame to Series
    
    # Load the best parameters from the pickle file (which contains parameters for multiple models)
    models_dir = os.path.join("models")
    best_params_path = os.path.join(models_dir, "best_params.pkl")
    with open(best_params_path, "rb") as f:
        best_params_dict = pickle.load(f)
    
    # Select the best parameters for RandomForest (you can change this to another model if needed)
    best_rf_params = best_params_dict.get("RandomForest")
    if best_rf_params is None:
        raise ValueError("No best parameters found for RandomForest in the pickle file.")
    
    # Initialize and train the model using RandomForestRegressor with the best parameters
    model = RandomForestRegressor(random_state=42, **best_rf_params)
    model.fit(X_train, y_train)
    
    # Save the trained model as a .joblib file
    model_path = os.path.join(models_dir, "trained_model.joblib")
    joblib.dump(model, model_path)
    
    print("Model trained and saved to", model_path)

if __name__ == "__main__":
    main()
