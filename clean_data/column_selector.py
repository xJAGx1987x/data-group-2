import pandas as pd
import numpy as np
import os
import json

# === Configuration ===
input_file = r"D:\Data\Data Mining 2025\Chicago\Crimes_-_2001_to_Present_20250220.csv"
iucr_mapping_file = "iucr_codes.json"
output_folder = "Chicago_cleaned_output"
output_file = os.path.join(output_folder, "Chicago_cleaned.csv")

os.makedirs(output_folder, exist_ok=True)

# === Load IUCR Mapping ===
with open(iucr_mapping_file, "r") as f:
    iucr_dict = json.load(f)

# === Helper Functions ===
def is_violent(iucr):
    entry = iucr_dict.get(str(iucr).strip())
    return 1 if entry and entry.get("Index") == "I" else 0

def get_primary(iucr):
    return iucr_dict.get(str(iucr).strip(), {}).get("Primary")

def get_secondary(iucr):
    return iucr_dict.get(str(iucr).strip(), {}).get("Secondary")

def month_to_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'
    return None

def process_chunk(chunk):
    chunk['IUCR'] = chunk['IUCR'].astype(str).str.zfill(4)
    chunk['AM_PM'] = chunk['Date'].str.extract(r'(AM|PM)')
    chunk['DateTime'] = pd.to_datetime(chunk['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    chunk['Date'] = chunk['DateTime'].dt.date
    chunk['Time'] = chunk['DateTime'].dt.time
    chunk['Hour'] = chunk['DateTime'].dt.hour
    chunk['Month'] = chunk['DateTime'].dt.month
    chunk['Season'] = chunk['Month'].apply(month_to_season)
    chunk[['Latitude', 'Longitude']] = chunk['Location'].str.extract(r'\((.*), (.*)\)').astype(float)
    chunk['is_violent'] = chunk['IUCR'].apply(is_violent)
    chunk['Crime_Type'] = chunk['IUCR'].apply(get_primary)
    chunk['Crime_Subtype'] = chunk['IUCR'].apply(get_secondary)

    # Drop rows missing critical info
    chunk = chunk.dropna(subset=['IUCR', 'District', 'Beat', 'Ward', 'DateTime', 'Latitude', 'Longitude', 'is_violent'])

    # Final column ordering
    final_cols = ['IUCR', 'Crime_Type', 'Crime_Subtype', 'is_violent',
                  'District', 'Beat', 'Ward', 'Latitude', 'Longitude',
                  'Location Description', 'Date', 'Time', 'Hour', 'AM_PM', 'Month', 'Season']

    return chunk[final_cols]

# === Load Dataset ===
columns_to_keep = ["IUCR", "Date", "District", "Beat", "Ward", "Location Description", "Location"]
df = pd.read_csv(input_file, usecols=lambda col: col in columns_to_keep)

# === Process in Single Thread ===
print("🚀 Starting simplified processing using built-in location columns...")
cleaned_df = process_chunk(df)

# === Save Final Cleaned CSV ===
cleaned_df.to_csv(output_file, index=False)
print(f"✅ Final cleaned dataset saved to: {output_file} ({len(cleaned_df)} rows)")
