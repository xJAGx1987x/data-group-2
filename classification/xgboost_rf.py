# random_forest_xgboost.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import time

# === Load Fully Encoded Data ===
data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_encoded.csv"
df = pd.read_csv(data_file)

# === Features and Target ===
features = ['District', 'Latitude', 'Longitude', 'Hour', 'Month',
            'Location Description', 'AM_PM', 'Season', 'Crime_Type', 'Crime_Subtype']
target = 'is_violent'

X = df[features]
y = df[target]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train XGBoost Classifier ===
start_time = time.time()

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    subsample=0.5,        # Use 50% of data at each tree
    tree_method='hist',   # Fast, memory-optimized training
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

end_time = time.time()

# === Evaluation ===
y_pred = xgb_model.predict(X_test)

print("\n🔍 XGBoost Classifier Results:")
print(classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\n⏱️ Training completed in {(end_time - start_time):.2f} seconds.")