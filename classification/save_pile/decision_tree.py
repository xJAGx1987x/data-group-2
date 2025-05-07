import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# === Load Fully Encoded Data ===
data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_encoded.csv"
df = pd.read_csv(data_file)

# === Define New Feature Set (location + time only) ===
features = ['District', 'Ward', 'Beat', 'Hour', 'Month', 'AM_PM', 'Season', 'Location Description']
target = 'is_violent'

# === Drop rows with missing data in used columns
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train a Single Decision Tree ===
tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)  # keep depth limited for clarity
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

# === Evaluation
print("\n🔍 Decision Tree Classifier (Location + Time only):")
print(classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Textual Tree Rules
print("\n🛠 Decision Tree Rules:")
tree_rules = export_text(tree_model, feature_names=features)
print(tree_rules)

# === Save Plot
output_dir = "classification/output"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=features, class_names=["Non-Violent", "Violent"], filled=True, rounded=True)
plot_path = os.path.join(output_dir, "decision_tree_location_time_only.png")
plt.title("Decision Tree (Location + Time Only)")
plt.savefig(plot_path)
plt.show()

print(f"✅ Decision tree plot saved to: {plot_path}")
