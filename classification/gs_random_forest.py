import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
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
df = df.drop(columns=drop_cols, errors='ignore')

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

# === Hyperparameter Tuning ===
print("🔍 Performing Grid Search on Random Forest...")
param_grid = {
    'n_estimators': [25, 50, 100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='f1',
    verbose=2
)
grid.fit(X_train, y_train)

# === Use Best Estimator ===
best_rf = grid.best_estimator_
print("\n✅ Best Parameters:", grid.best_params_)

# === Evaluate Best Estimator on Test Set ===
rf_preds = best_rf.predict(X_test)
rf_probs = best_rf.predict_proba(X_test)[:, 1]

print("\n=== Tuned Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Non-Violent", "Violent"],
            yticklabels=["Non-Violent", "Violent"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, rf_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Receiver Operating Characteristic (ROC)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# === Feature Importance from best_rf ===
importances = best_rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n🌟 Top 10 Important Features:")
print(importance_df.head(10).to_string(index=False))

# === Retrain Best Estimator on All Data ===
print("\n🔁 Retraining best model on full dataset...")
best_rf_final = RandomForestClassifier(**grid.best_params_, random_state=42, n_jobs=-1)
best_rf_final.fit(X, y)

# === Save Final Model ===
joblib.dump(best_rf_final, "rf_final_model.pkl")
print("💾 Final model saved as rf_final_model.pkl")
