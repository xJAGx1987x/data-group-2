# This script processes a large CSV file containing crime data from Chicago.
# It cleans the data, classifies crimes based on IUCR codes, and saves the cleaned data to a new CSV file.
# The script uses threading to speed up the processing of large datasets.
# It also includes a mapping file for IUCR codes to determine the type and subtype of crimes.
# The cleaned dataset includes columns for IUCR, crime type, subtype, violence classification, district, location description, date, time, hour, AM/PM, month, and season.
# The script is designed to be efficient and can handle large datasets by splitting the workload across multiple threads.

import pandas as pd
import numpy as np 
import os
import json
import threading


# === Configuration ===
input_file = r"D:\Data\Data Mining 2025\Chicago\Crimes_-_2001_to_Present_20250220.csv"
iucr_mapping_file = "iucr_codes.json"  # Ensure this is in the same directory as your script
output_folder = "Chicago_cleaned_output"
output_file = os.path.join(output_folder, "Chicago_cleaned.csv")
num_threads = 4  # You can adjust this depending on your CPU cores

# === Ensure Output Directory Exists ===
os.makedirs(output_folder, exist_ok=True)

# === Load IUCR Mapping ===
with open(iucr_mapping_file, "r") as f:
    iucr_dict = json.load(f)

# === Helper Functions ===
def is_violent(iucr):
    entry = iucr_dict.get(str(iucr).strip())
    if entry:
        return 1 if entry.get("Index") == "I" else 0
    return None

def get_primary(iucr):
    return iucr_dict.get(str(iucr).strip(), {}).get("Primary")

def get_secondary(iucr):
    return iucr_dict.get(str(iucr).strip(), {}).get("Secondary")

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return None

def process_chunk(chunk, results, index):
    # Ensure IUCR is a zero-padded 4-character string
    chunk['IUCR'] = chunk['IUCR'].astype(str).str.zfill(4)

    # Extract AM/PM from original Date string
    chunk['AM_PM'] = chunk['Date'].str.extract(r'(AM|PM)')

    # Convert Date to Datetime and split components
    chunk['DateTime'] = pd.to_datetime(chunk['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    chunk['Date'] = chunk['DateTime'].dt.date
    chunk['Time'] = chunk['DateTime'].dt.time
    chunk['Hour'] = chunk['DateTime'].dt.hour
    chunk['Month'] = chunk['DateTime'].dt.month

    # Create Season column
    chunk['Season'] = chunk['Month'].apply(month_to_season)

    # Extract Latitude and Longitude
    chunk[['Latitude', 'Longitude']] = chunk['Location'].str.extract(r'\((.*), (.*)\)').astype(float)

    # Classify Crimes
    chunk['is_violent'] = chunk['IUCR'].apply(is_violent)
    chunk['Crime_Type'] = chunk['IUCR'].apply(get_primary)
    chunk['Crime_Subtype'] = chunk['IUCR'].apply(get_secondary)

    # Drop rows with missing critical values
    chunk = chunk.dropna(subset=['IUCR', 'District', 'DateTime', 'Latitude', 'Longitude'])

    # Drop unnecessary columns
    chunk = chunk.drop(columns=['Location', 'DateTime'])

    # Final column order
    chunk = chunk[['IUCR', 'Crime_Type', 'Crime_Subtype', 'is_violent',
                   'District', 'Latitude', 'Longitude', 'Location Description',
                   'Date', 'Time', 'Hour', 'AM_PM', 'Month', 'Season']]

    results[index] = chunk

# === Load Only Needed Columns ===
columns_to_keep = ["IUCR", "Date", "District", "Location Description", "Location"]
df = pd.read_csv(input_file, usecols=lambda col: col in columns_to_keep)

# === Split into Chunks ===
chunks = np.array_split(df, num_threads)
threads = []
results = [None] * num_threads

# === Start Threads ===
for i in range(num_threads):
    t = threading.Thread(target=process_chunk, args=(chunks[i], results, i))
    threads.append(t)
    t.start()

# === Wait for Threads to Finish ===
for t in threads:
    t.join()

# === Combine Processed Chunks ===
final_df = pd.concat(results)

# === Save Cleaned CSV ===
final_df.to_csv(output_file, index=False)
print(f"✅ Final cleaned dataset with violence classification saved to: {output_file} ({len(final_df)} rows, {len(final_df.columns)} columns)")
