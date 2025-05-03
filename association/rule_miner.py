import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# === Load and sample data ===
df = pd.read_csv("Chicago_cleaned_output/Chicago_encoded.csv", nrows=500_000)

# === Use top features only ===
top_features = ['Beat', 'Hour', 'Location Description', 'Month', 'Ward', 'District', 'Season', 'AM_PM', 'is_violent']
df = df[top_features].dropna()

# === Convert numeric to label-style format: 'Feature_Value'
for col in df.columns:
    df[col] = df[col].astype(str).apply(lambda val: f"{col}_{val}")

# === Convert rows to list of feature-value pairs
transactions = df.apply(lambda row: set(row), axis=1)

# === One-hot encode for mlxtend
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# === Mine frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# === Extract association rules where is_violent=1 is the outcome
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules_violent = rules[rules['consequents'].apply(lambda x: 'is_violent_1' in x)]

# === Sort and display top rules
top_rules = rules_violent.sort_values(by=["confidence", "lift"], ascending=False)
print("\n🔍 Top Association Rules Predicting is_violent=1:\n")
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
