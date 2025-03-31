#!/usr/bin/env python3

import os
import pandas as pd

def main():
    # Path to data: data/raw/raw.csv
    input_path = os.path.join("data", "raw", "raw.csv")
    processed_dir = os.path.join("data", "processed")
    # output path for the new dataset
    output_path = os.path.join("data", "raw", "raw_new.csv")
    
    # load the raw data
    df = pd.read_csv(input_path)
    print("Original column", df.columns.tolist())
    
    # Drop the 'date' column
    if "date" in df.columns:
        df = df.drop(columns=["date"])
        print("Column 'date' was dropped.")
    else:
        print("Column 'date' not found, no column dropped.")
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"New dataset saved to {output_path}")

# Save the modified DataFrame to a new CSV file
if __name__ == "__main__":
    main()
