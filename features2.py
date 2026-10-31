import pandas as pd

ge = pd.read_csv("ge_data.csv", low_memory=False)
ge.loc[ge["Turnout_Percentage"] > 100, "Turnout_Percentage"] = None
ge["Party"] = ge["Party"].replace({"INC(I)": "INC"})

winners = ge[ge["Position"] == 1].copy()
keys = ["State_Name", "Constituency_Name"]
winners = winners.sort_values(keys + ["Year"])

# Streak: how many elections in a row the same party has held the seat
g = winners.groupby(keys)
winners["changed"] = (winners["Party"] != g["Party"].shift(1)).astype(int)
winners["run_id"] = winners.groupby(keys)["changed"].cumsum()
winners["streak"] = winners.groupby(keys + ["run_id"]).cumcount() + 1

# Everything below is what we knew BEFORE this election
g = winners.groupby(keys)
winners["prev_party"] = g["Party"].shift(1)
winners["prev_margin"] = g["Margin_Percentage"].shift(1)
winners["prev_turnout"] = g["Turnout_Percentage"].shift(1)
winners["prev_vote_share"] = g["Vote_Share_Percentage"].shift(1)
winners["prev_n_cand"] = g["N_Cand"].shift(1)
winners["prev_streak"] = g["streak"].shift(1)

winners["prev_is_INC"] = (winners["prev_party"] == "INC").astype(int)
winners["prev_is_BJP"] = (winners["prev_party"] == "BJP").astype(int)
winners["is_reserved"] = (winners["Constituency_Type"] != "GEN").astype(int)

data = winners.dropna(subset=["prev_party"]).copy()
data["retained"] = (data["Party"] == data["prev_party"]).astype(int)

print("Rows:", len(data))
print("Constituency_Type values:")
print(data["Constituency_Type"].value_counts())

print("Retained share by length of previous streak (5 means 5 or more):")
streak = data["prev_streak"].clip(upper=5)
print(data.groupby(streak)["retained"].agg(["mean", "count"]).round(2))

print("Retained share by previous winner:")
two = data[data["prev_party"].isin(["INC", "BJP"])]
print(two.groupby("prev_party")["retained"].agg(["mean", "count"]).round(2))

data.to_csv("model_data2.csv", index=False)