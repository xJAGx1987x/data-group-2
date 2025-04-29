import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# === Load Pre-Encoded Data ===
data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_cleaned.csv"
df = pd.read_csv(data_file)

# === Select and LIMIT Features ===
features = ['Location Description', 'Season', 'Crime_Type']
df = df[features]

# --- Reduce cardinality
top_locations = df['Location Description'].value_counts().nlargest(10).index
df['Location Description'] = df['Location Description'].where(df['Location Description'].isin(top_locations), other='Other')

# (Optional: reduce others similarly if necessary)

# === One-hot encode ===
df_encoded = pd.get_dummies(df)

# === Find Frequent Itemsets ===
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# === Generate Association Rules ===
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# === Sort by Lift
rules = rules.sort_values(by="lift", ascending=False)

# === Output
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
