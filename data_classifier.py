import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Load Cleaned Data ===
df = pd.read_csv("Chicago_cleaned_output/Chicago_cleaned.csv")

# === Drop rows where 'is_violent' is missing (just in case) ===
df = df.dropna(subset=['is_violent'])

# === Select Features and Target ===
features = ['District', 'Latitude', 'Longitude', 'Location Description', 'Hour', 'Month']
target = 'is_violent'

X = df[features]
y = df[target]

# === One-Hot Encode Categorical Columns ===
X_encoded = pd.get_dummies(X, columns=['Location Description'], drop_first=True)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# === Scale Numerical Features (optional but helpful for some classifiers) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# === Predictions ===
y_pred = clf.predict(X_test_scaled)

# === Evaluation ===
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
