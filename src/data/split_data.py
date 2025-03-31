#!/usr/bin/env python3

import os 
import sys 
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Path to data: data/raw/raw.csv
    raw_data_path = os.path.join("data", "raw", "raw_new.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # load the raw data
    df = pd.read_csv(raw_data_path)
    
    # Last column is the target variable = 'silica_concentrate'
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    # Splitting the data into training and testing sets
    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the split data to CSV files
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    print("Data splitting complete. Files saved in", processed_dir)

# Check if the script is being run directly
if __name__ == "__main__":
    main()
