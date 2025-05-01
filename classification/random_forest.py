import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === Load Cleaned Dataset ===
data_file = r"Chicago_cleaned_output\Chicago_encoded.csv"
df = dd.read_csv(data_file, dtype={'IUCR': 'object'})

# === Define target and features
target_column = 'is_violent'
drop_cols = ['IUCR', 'Crime_Type', 'Crime_Subtype', 'Date', 'Time', 'Latitude', 'Longitude', target_column]
X = df.drop(columns=drop_cols)
y = df[target_column]

# === Convert to pandas
X = X.compute()
y = y.compute()

# === Clean and drop missing
full_df = pd.concat([X, y], axis=1).dropna()
X = full_df.drop(columns=[target_column])
y = full_df[target_column]

# === Label Encode Categorical Columns
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna("Unknown"))
        label_encoders[col] = le

# === Subsample 100,000 rows
sample_df = pd.concat([X, y], axis=1).sample(n=250_000, random_state=42)
X_sample = sample_df.drop(columns=[target_column])
y_sample = sample_df[target_column]

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# === Train Random Forest with Class Balancing
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


# === Evaluation
print("\n=== Random Forest Classifier Results ===")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds))

# === Feature Importance
importances = rf_model.feature_importances_
features = X_sample.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n🌟 Top 10 Important Features:")
print(importance_df.head(10).to_string(index=False))
