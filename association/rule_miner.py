import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data_file = r"C:\Users\trend\PycharmProjects\data-group-2\Chicago_cleaned_output\Chicago_cleaned.csv"
df = pd.read_csv(data_file)

# === Select Features ===
features = ['Location Description', 'Season', 'Crime_Type']
df = df[features]

# === Collapse Rare Categories ===
top_locations = df['Location Description'].value_counts().nlargest(10).index
df['Location Description'] = df['Location Description'].where(df['Location Description'].isin(top_locations), other='Other')

# (Optional: Same for Crime_Type or others)

# === One-Hot Encode ===
df_encoded = pd.get_dummies(df)

# === Find Frequent Itemsets ===
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# === Generate Association Rules ===
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# === Sort Rules ===
rules = rules.sort_values(by="lift", ascending=False)

# === Output Results ===
for idx, rule in rules.iterrows():
    antecedents = ', '.join([str(a) for a in rule['antecedents']])
    consequents = ', '.join([str(c) for c in rule['consequents']])
    support = round(rule['support'], 3)
    confidence = round(rule['confidence'], 3)
    lift = round(rule['lift'], 3)
    
    print(f"If [{antecedents}], then [{consequents}] (support: {support}, confidence: {confidence}, lift: {lift})")

