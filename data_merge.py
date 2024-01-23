"""
Create concatnated datasets with all data files
"""

import os
import pandas as pd

def concatenate_csv_files(folder_path, output_file):
    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Check if there are any CSV files in the folder
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Initialize an empty DataFrame to store the concatenated data
    concatenated_data = pd.DataFrame()

    # Iterate through each CSV file and concatenate its data
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)

    # Write the concatenated data to a new CSV file
    concatenated_data.to_csv(output_file, index=False)
    print(f"Concatenated data saved to {output_file}")

# Example usage:
folder_path = r"C:\Users\AIMAN\data science learn\Projects\Recommendation system\Career-Recommendation\data"
output_file = r"C:\Users\AIMAN\data science learn\Projects\Recommendation system\Career-Recommendation\data\concatenated_data.csv"

concatenate_csv_files(folder_path, output_file)
