import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import folium

#loading the data
df = pd.read_csv('chicago_crime_data.csv')

#drop any missing coordinates
df=df.dropna(subset=['Latitude','Longitude'])

#convert to radians for Haversine
coords = df[['Latitude', 'Longitude']].to_numpy()
coords_rad = np.radians(coords)

#Earth radius in kilometers
kms_per_radian = 6371.0088

#Choose an appropriate epsilon (radius for a cluster), like 1 km
epsilon = 1 / kms_per_radian

db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(coords_rad)

df['cluster'] = db.labels_

plt.figure(figsize=(10, 8))
plt.scatter(df['Longitude'], df['Latitude'], c=df['cluster'], cmap='tab10', s=10)
plt.title('Crime Hotspots in Chicago (DBSCAN)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster ID')
plt.show()

#Centered on Chicago
m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=2,
        color='red' if row['cluster'] != -1 else 'gray',
        fill=True
    ).add_to(m)

m.save("chicago_crime_cluster.html")