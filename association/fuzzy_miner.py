import pandas as pd
import numpy as np
import json
import os
import math
from collections import defaultdict

# === Config ===
data_file = r"C:/Users/trend/PycharmProjects/data-group-2/Chicago_cleaned_output/Chicago_cleaned.csv"
chunk_size = 100_000
results_folder = "association/output"
os.makedirs(results_folder, exist_ok=True)

# === Load IUCR and Build Crime Mapping ===
with open("C:/Users/trend/PycharmProjects/data-group-2/iucr_codes.json", "r") as f:
    iucr_data = json.load(f)
primary_crimes = sorted(set(entry["Primary"] for entry in iucr_data.values()))
crime_map = {crime: idx + 1 for idx, crime in enumerate(primary_crimes)}

# === Season Mapping ===
season_map = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}

# === Fuzzy Membership Functions ===
def fuzzy_membership_season(val):
    return (val - 1) / 3.0 if pd.notnull(val) else 0

def fuzzy_membership_crime(val):
    return (val - 1) / (len(crime_map) - 1) if pd.notnull(val) else 0

# === Init Global Counters ===
rule_counts = defaultdict(int)
antecedent_counts = defaultdict(int)
consequent_counts = defaultdict(int)
total_records = 0

# === Count Total Lines ===
with open(data_file, 'r') as f:
    total_lines = sum(1 for _ in f) - 1
num_chunks = math.ceil(total_lines / chunk_size)

# === Process Chunks ===
for idx, chunk in enumerate(pd.read_csv(data_file, chunksize=chunk_size)):
    print(f"\n🚀 Processing chunk {idx + 1} of {num_chunks}...")
    total_records += len(chunk)

    # Map categories to numbers
    chunk["Season_val"] = chunk["Season"].map(season_map)
    chunk["Crime_val"] = chunk["Crime_Type"].map(crime_map)

    # Fuzzy Membership
    chunk["Season_mem"] = chunk["Season_val"].apply(fuzzy_membership_season)
    chunk["Crime_mem"] = chunk["Crime_val"].apply(fuzzy_membership_crime)

    # Filter on threshold
    filtered = chunk[(chunk["Season_mem"] > 0.6) & (chunk["Crime_mem"] > 0.6)]

    if filtered.empty:
        continue

    # Group & Count
    grouped = filtered.groupby(["Season_val", "Crime_val"]).size().reset_index(name="count")
    season_totals = filtered["Season_val"].value_counts().to_dict()
    crime_totals = filtered["Crime_val"].value_counts().to_dict()

    # Compute rule stats and save per-chunk
    results = []
    for _, row in grouped.iterrows():
        sv, cv, cnt = row["Season_val"], row["Crime_val"], row["count"]
        support = cnt / len(chunk)
        confidence = cnt / season_totals.get(sv, cnt)
        lift = confidence / (crime_totals.get(cv, 1) / len(chunk))
        results.append({
            "Season": sv,
            "Crime_Type": cv,
            "Support": round(support, 4),
            "Confidence": round(confidence, 4),
            "Lift": round(lift, 4)
        })

        # Update global counts
        rule_counts[(sv, cv)] += cnt
        antecedent_counts[sv] += season_totals.get(sv, 0)
        consequent_counts[cv] += crime_totals.get(cv, 0)

    # Save chunk results
    pd.DataFrame(results).to_csv(f"{results_folder}/chunk_{idx+1}_rules.csv", index=False)

# === Aggregate Global Rules ===
final_results = []
for (sv, cv), count in rule_counts.items():
    supp = count / total_records
    conf = count / antecedent_counts.get(sv, count)
    lift = conf / (consequent_counts.get(cv, 1) / total_records)
    s_label = [k for k,v in season_map.items() if v == sv][0]
    c_label = [k for k,v in crime_map.items() if v == cv][0]
    final_results.append({
        "Season": s_label,
        "Crime_Type": c_label,
        "Support": round(supp, 4),
        "Confidence": round(conf, 4),
        "Lift": round(lift, 4)
    })

# Sort and Display
final_df = pd.DataFrame(final_results).sort_values(by="Lift", ascending=False)
final_df.to_csv(f"{results_folder}/global_fuzzy_rules.csv", index=False)
print("\n✅ Global fuzzy rules saved to:", f"{results_folder}/global_fuzzy_rules.csv")
print("\n🔥 Top 10 fuzzy rules:")
print(final_df.head(10).to_string(index=False))
