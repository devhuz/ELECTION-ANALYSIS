import pandas as pd

ge = pd.read_csv("ge_data.csv", low_memory=False)
w = ge[ge["Position"] == 1]

# Which party labels contain "INC"?
inc_labels = w[w["Party"].str.contains("INC", na=False)]["Party"]
print("INC-type labels:", sorted(inc_labels.unique()))

# Top 5 winning labels in the years that look wrong
for y in [1977, 1980, 1984, 1989, 1991]:
    counts = w[w["Year"] == y]["Party"].value_counts().head(5)
    print(y, dict(counts))