import pandas as pd
import folium
import os
import webbrowser
from folium.plugins import MarkerCluster

# === Load Cleaned Data ===
data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_cleaned.csv"
df = pd.read_csv(data_file)

# === Drop Rows Without Coordinates or Classification ===
df = df.dropna(subset=['Latitude', 'Longitude', 'is_violent'])

# === Optional Downsample to ~100k for performance ===
sample_size = 100_000
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)

print(f"✅ Using {len(df)} crime records.")

# === Create Folium Map ===
m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

# === Plot Points: Red = Violent, Blue = Non-Violent with Tooltip ===
for _, row in df.iterrows():
    color = 'red' if row['is_violent'] == 1 else 'blue'
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=2,
        color=color,
        fill=True,
        fill_opacity=0.4,
        tooltip=f"{row['Crime_Type']} (District {row['District']})"
    ).add_to(marker_cluster)

# === Output Directory and File ===
output_folder = r"classification/output"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "chicago_violent_vs_nonviolent.html")
m.save(output_file)

print(f"✅ Crime classification map saved to: {output_file}")

# === Open in Browser ===
webbrowser.open('file://' + os.path.realpath(output_file))
