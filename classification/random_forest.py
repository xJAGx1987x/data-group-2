import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# === Configuration ===
data_file = r"Chicago_cleaned_output\Chicago_balanced_encoded.csv"
target_column = 'is_violent'
sample_per_class = 50_000

# === Load Data ===
print("📂 Loading dataset...")
df = pd.read_csv(data_file)

# === Drop Non-Feature or Derived Columns ===
drop_cols = ['IUCR', 'Date', 'Time', 'Latitude', 'Longitude', 'Crime_Type', 'Crime_Subtype']
df = df.drop(columns=drop_cols, errors='ignore')  # ignore in case some columns aren't present

# === One-hot Encode Categorical Columns ===
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [col for col in cat_cols if col != target_column]
df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

# === Stratified Sampling ===
print("🔄 Stratified sampling...")
df_sampled = df.groupby(target_column, group_keys=False).apply(
    lambda x: x.sample(n=sample_per_class, random_state=42)
).reset_index(drop=True)

# === Train/Test Split ===
X = df_sampled.drop(columns=[target_column])
y = df_sampled[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Random Forest ===
print("🌲 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC

# === Evaluation ===
print("\n=== Random Forest Classifier Results ===")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Violent", "Violent"], yticklabels=["Non-Violent", "Violent"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === ROC Curve ===
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.title("Receiver Operating Characteristic (ROC)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# === Feature Importance ===
importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n🌟 Top 10 Important Features:")
print(importance_df.head(10).to_string(index=False))
