#!/usr/bin/env python3

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# This script normalizes the training and testing features using StandardScaler.
# It outputs the normalized data to the data/processed directory.
def main():
    processed_dir = os.path.join("data", "processed")
    X_train_path = os.path.join(processed_dir, "X_train.csv")
    X_test_path = os.path.join(processed_dir, "X_test.csv")

    # Load the training and testing data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # Initialize StandardScaler and fit to the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the scaled data back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Save the scaled data to CSV files
    X_train_scaled.to_csv(os.path.join(processed_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, "X_test_scaled.csv"), index=False)

    print("Data normalization complete. Scaled files saved in", processed_dir)

if __name__ == "__main__":
    main()
