# random_forest_sampled.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time

# === Load Fully Encoded Data ===
data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_encoded.csv"
df = pd.read_csv(data_file)

# === OPTIONAL: Subsample the Data ===
sample_size = 500_000  # Only use 500,000 points
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)

print(f"✅ Using {len(df)} points for Random Forest.")

# === Features and Target ===
features = ['District', 'Latitude', 'Longitude', 'Hour', 'Month',
            'Location Description', 'AM_PM', 'Season', 'Crime_Type', 'Crime_Subtype']
target = 'is_violent'

X = df[features]
y = df[target]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Random Forest Classifier ===
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    max_samples=0.2,   # 20% of training set per tree
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

end_time = time.time()

# === Evaluation ===
y_pred = rf_model.predict(X_test)

print("\n🔍 Random Forest Classifier (Sampled) Results:")
print(classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\n⏱️ Training completed in {(end_time - start_time):.2f} seconds.")