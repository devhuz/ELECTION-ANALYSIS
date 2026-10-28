import pandas as pd

ge = pd.read_csv("ge_data.csv", low_memory=False)
ge.loc[ge["Turnout_Percentage"] > 100, "Turnout_Percentage"] = None
# Merge the INC(I) label into INC (same party lineage)
ge["Party"] = ge["Party"].replace({"INC(I)": "INC"})

# One row per constituency per election: the winner
winners = ge[ge["Position"] == 1].copy()

# Order each constituency's history by year
keys = ["State_Name", "Constituency_Name"]
winners = winners.sort_values(keys + ["Year"])

# Bring in the previous election's values for the same constituency
g = winners.groupby(keys)
winners["prev_party"] = g["Party"].shift(1)
winners["prev_margin"] = g["Margin_Percentage"].shift(1)
winners["prev_turnout"] = g["Turnout_Percentage"].shift(1)

# Target: did the same party win again? (1 = yes, 0 = seat flipped)
data = winners.dropna(subset=["prev_party"]).copy()
data["retained"] = (data["Party"] == data["prev_party"]).astype(int)

print("Rows with a previous election:", len(data))
print("Share of seats retained, by year:")
print(data.groupby("Year")["retained"].mean().round(2))
print("Overall retained share:", round(data["retained"].mean(), 3))

data.to_csv("model_data.csv", index=False)