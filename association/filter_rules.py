# Load the global rules
import pandas as pd

rules = pd.read_csv("association/output/global_fuzzy_rules.csv")

# Filter: Lift > 1.2
strong_rules = rules[(rules['Lift'] >= 0.9) & (rules['Confidence'] > 0.005)]

# Sort by Lift descending
strong_rules = strong_rules.sort_values(by='Lift', ascending=False)

# Save strong rules separately
strong_rules.to_csv("association/output/strong_fuzzy_rules.csv", index=False)

print(f"✅ Strong fuzzy rules saved to association/output/strong_fuzzy_rules.csv")
print(f"🔥 Top 10 strong fuzzy rules:\n")
print(strong_rules.head(10).to_string(index=False))
