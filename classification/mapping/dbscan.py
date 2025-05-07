import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import folium
import os
import webbrowser

# === Load Data ===
data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_cleaned.csv"
df = pd.read_csv(data_file)

# === Drop Missing Coordinates ===
df = df.dropna(subset=['Latitude', 'Longitude'])

# === OPTIONAL: Downsample to avoid MemoryError ===
sample_size = 100_000  # Choose 100,000 points
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)

print(f"✅ Using {len(df)} points for clustering.")

# === Prepare Coordinates for Haversine ===
coords = df[['Latitude', 'Longitude']].to_numpy()
coords_rad = np.radians(coords)

# === Setup DBSCAN ===
kms_per_radian = 6371.0088
epsilon = 1 / kms_per_radian  # 1 km radius

db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine')
db.fit(coords_rad)

df['cluster'] = db.labels_

# === Plot Static Map ===
plt.figure(figsize=(10, 8))
plt.scatter(df['Longitude'], df['Latitude'], c=df['cluster'], cmap='tab10', s=10)
plt.title('Crime Hotspots in Chicago (DBSCAN Clustering)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster ID')
plt.show()

# === Interactive Folium Map ===
m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=2,
        color='red' if row['cluster'] != -1 else 'gray',
        fill=True,
        fill_opacity=0.5
    ).add_to(m)

# === Safely Create Output Folder ===
output_folder = r"classification/output"
os.makedirs(output_folder, exist_ok=True)

# === Save Map Correctly ===
output_file = os.path.join(output_folder, "chicago_crime_clusters_downsampled.html")
m.save(output_file)

print(f"✅ Interactive cluster map saved: {output_file}")

# === Auto-Open Map in Web Browser (optional but nice) ===
webbrowser.open('file://' + os.path.realpath(output_file))
