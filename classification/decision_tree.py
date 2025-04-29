# decision_tree.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# === Load Fully Encoded Data ===
data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_encoded.csv"
df = pd.read_csv(data_file)

# === Define Features and Target ===
features = ['District', 'Latitude', 'Longitude', 'Hour', 'Month', 'Location Description', 'AM_PM', 'Season', 'Crime_Type', 'Crime_Subtype']
target = 'is_violent'

X = df[features]
y = df[target]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train a Single Decision Tree ===
tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limit depth for better visualization
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

# === Evaluation ===
print("\n🔍 Decision Tree Classifier Results:")
print(classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Text Visualization (Console) ===
print("\n🛠 Textual Decision Tree:")
tree_rules = export_text(tree_model, feature_names=features)
print(tree_rules)

# === Graphical Visualization (Plot) ===
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=features, class_names=["Non-Violent", "Violent"], filled=True, rounded=True)
plt.title("Decision Tree Visualization (Limited Depth)")
plt.savefig(r"classification/output/decision_tree_visualization.png")  # Save the plot
plt.show()

print("✅ Decision tree plot saved to: classification/output/decision_tree_visualization.png")