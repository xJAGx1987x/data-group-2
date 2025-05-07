import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# === Configuration ===
model_file = "xgb_final_model.pkl"
data_file = r"Chicago_cleaned_output\Chicago_balanced_encoded.csv"
target_column = 'is_violent'

# === Load Model ===
print(f"📦 Loading model from: {model_file}")
model = joblib.load(model_file)

# === Load Data ===
print(f"📂 Loading dataset from: {data_file}")
df = pd.read_csv(data_file)

# === Drop Non-Feature or Derived Columns ===
drop_cols = ['IUCR', 'Date', 'Time', 'Latitude', 'Longitude', 'Crime_Type', 'Crime_Subtype']
df = df.drop(columns=drop_cols, errors='ignore')

# === One-hot Encode Categorical Columns ===
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [col for col in cat_cols if col != target_column]
df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

# === Prepare Features and Labels ===
X = df.drop(columns=[target_column])
y = df[target_column]

# === Predict and Evaluate ===
print("🔮 Making predictions...")
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]

print("\n=== Final XGBoost Model Results ===")
print(f"Accuracy: {accuracy_score(y, preds):.4f}")
print(classification_report(y, preds))

# === Confusion Matrix ===
cm = confusion_matrix(y, preds)
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
fpr, tpr, _ = roc_curve(y, probs)
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
