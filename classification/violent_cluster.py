import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import DBSCAN
import os
import webbrowser

# === Load Cleaned Dataset ===
data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_cleaned.csv"
df = pd.read_csv(data_file)

# === Filter to Violent Crimes Only with Coordinates ===
df = df[(df['is_violent'] == 1) & df['Latitude'].notna() & df['Longitude'].notna()]

# === Optional Downsample ===
sample_size = 100_000
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)

print(f"✅ Using {len(df)} violent crime records.")

# === Prepare Coordinates in Radians ===
coords = df[['Latitude', 'Longitude']].to_numpy()
coords_rad = np.radians(coords)

# === Run DBSCAN with Haversine Distance ===
kms_per_radian = 6371.0088
epsilon = 1.0 / kms_per_radian  # 1 km radius
db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine')
db.fit(coords_rad)
df['cluster'] = db.labels_

# === Create Folium Map ===
m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

# === Add Markers Colored by Cluster ===
cluster_colors = {}

for _, row in df.iterrows():
    cluster_id = row['cluster']
    if cluster_id == -1:
        color = 'gray'  # noise
    else:
        if cluster_id not in cluster_colors:
            cluster_colors[cluster_id] = f"#{np.random.randint(0, 0xFFFFFF):06x}"
        color = cluster_colors[cluster_id]

    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=2,
        color=color,
        fill=True,
        fill_opacity=0.5,
        tooltip=f"Cluster: {cluster_id} | District: {row['District']}"
    ).add_to(marker_cluster)

# === Save and Open Map ===
output_folder = r"classification/output"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "chicago_violent_crime_clusters.html")
m.save(output_file)

print(f"✅ Clustered violent crime map saved to: {output_file}")
webbrowser.open('file://' + os.path.realpath(output_file))

# === Season Analysis by Cluster ===
if 'Season' in df.columns:
    ct = pd.crosstab(df['cluster'], df['Season'])
    ct_pct = ct.div(ct.sum(axis=1), axis=0)

    ct.to_csv(os.path.join(output_folder, "violent_crime_by_cluster_and_season.csv"))
    ct_pct.to_csv(os.path.join(output_folder, "violent_crime_percent_by_cluster_and_season.csv"))

    print("✅ Saved seasonal breakdowns of violent crime by cluster.")
    print(ct.head())
    print("\nTop % table:")
    print(ct_pct.head())
else:
    print("⚠️ 'Season' column not found in DataFrame.")
