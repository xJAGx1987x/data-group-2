# k_nearest_neighbors_dask.py

import dask.dataframe as dd
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# === Load Cleaned Dataset with Dask ===
data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_encoded.csv"
df = dd.read_csv(data_file, dtype={'IUCR': 'object'})

# === Define target and features (still Dask here)
target_column = 'is_violent'
drop_cols = ['Date', 'Time', 'Crime_Type', 'Crime_Subtype', target_column]
X = df.drop(columns=drop_cols)
y = df[target_column]

# === Compute (Dask -> pandas)
X = X.compute()
y = y.compute()

# === Combine X and y to clean missing values
full_df = pd.concat([X, y], axis=1)
full_df = full_df.dropna()

# === Split again
X = full_df.drop(columns=[target_column])
y = full_df[target_column]

# === Now X and y are pandas DataFrames (good)
# === Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === K-Nearest Neighbors Classifier ===
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)  # n_jobs=-1 = use all cores
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

# === Evaluation ===
print("\n=== K-Nearest Neighbors Classifier Results ===")
print(f"Accuracy: {accuracy_score(y_test, knn_preds):.4f}")
print(classification_report(y_test, knn_preds))
