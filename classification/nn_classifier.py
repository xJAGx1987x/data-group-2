import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# === Load and clean data ===
df = pd.read_csv("Chicago_cleaned_output/Chicago_balanced_encoded.csv")
df = df.drop(columns=['IUCR', 'Date', 'Time', 'Latitude', 'Longitude', 'Crime_Type', 'Crime_Subtype'], errors='ignore')
df = pd.get_dummies(df, drop_first=True)

# Sample 50k per class
df = df.groupby('is_violent', group_keys=False).apply(lambda x: x.sample(n=50000, random_state=42)).reset_index(drop=True)

X = df.drop(columns=['is_violent'])
y = df['is_violent']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Train MLP ===
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, alpha=0.0005,
                    solver='adam', random_state=42, verbose=True)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

print("\n=== scikit-learn MLPClassifier Results ===")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_probs))

# === Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Violent", "Violent"], yticklabels=["Non-Violent", "Violent"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()
