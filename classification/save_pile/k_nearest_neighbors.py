import dask.dataframe as dd
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# === Load Cleaned Dataset ===
data_file = r"Chicago_cleaned_output\Chicago_encoded.csv"
df = dd.read_csv(data_file, dtype={'IUCR': 'object'})

# === Define target and features
target_column = 'is_violent'
drop_cols = ['IUCR', 'Crime_Type', 'Crime_Subtype', 'Date', 'Time', 'Latitude', 'Longitude', target_column]
X = df.drop(columns=drop_cols)
y = df[target_column]

# === Convert Dask to pandas
X = X.compute()
y = y.compute()

# === Combine and drop missing
full_df = pd.concat([X, y], axis=1).dropna()
X = full_df.drop(columns=[target_column])
y = full_df[target_column]

# === Encode Categorical Features
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna("Unknown"))
        label_encoders[col] = le

# === Subsample 100,000 rows for faster training
sample_df = pd.concat([X, y], axis=1).sample(n=250_000, random_state=42)
X_sample = sample_df.drop(columns=[target_column])
y_sample = sample_df[target_column]

# === Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_sample, test_size=0.2, random_state=42)

# === Grid Search for Best k
print("\n🔍 Running GridSearchCV on sample...")
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid = GridSearchCV(KNeighborsClassifier(n_jobs=-1), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

best_k = grid.best_params_['n_neighbors']
print(f"✅ Best k found: {best_k}")

# === Final Model with Best k
knn_model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

# === Evaluation
print("\n=== KNN Results on 100K Sample ===")
print(f"Accuracy: {accuracy_score(y_test, knn_preds):.4f}")
print(classification_report(y_test, knn_preds))
