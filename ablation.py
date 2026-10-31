import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("candidate_data.csv")
all_features = ["prior_contests", "prior_wins", "is_incumbent", "party_held_seat",
                "party_streak", "seat_margin", "seat_n_cand", "party_prev_share",
                "state_prev_share", "party_seat_prev_vote", "is_reserved"]
data = data.dropna(subset=all_features)
keys = ["State_Name", "Constituency_Name"]
years = [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]


def top1(test, scores):
    t = test[keys + ["won"]].copy()
    t["score"] = scores
    picks = t.groupby(keys)["score"].idxmax()
    return t.loc[picks, "won"].mean()


sets = {
    "all": all_features,
    "no_party_strength": [f for f in all_features
                          if f not in ["party_prev_share", "state_prev_share"]],
    "no_person_history": [f for f in all_features
                          if f not in ["prior_contests", "prior_wins", "is_incumbent"]],
    "seat_and_party_only": ["party_held_seat", "party_streak", "party_seat_prev_vote",
                            "seat_margin", "seat_n_cand", "is_reserved"],
}

results = {"prev_party": []}
for name in sets:
    results[name] = []

for year in years:
    train = data[data["Year"] < year]
    test = data[data["Year"] == year]
    results["prev_party"].append(top1(test, test["party_held_seat"].values))
    for name, feats in sets.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train[feats], train["won"])
        results[name].append(top1(test, model.predict_proba(test[feats])[:, 1]))

table = pd.DataFrame(results, index=years).round(3)
print(table)
print("Average:")
print(table.mean().round(3))
print("Years better than prev_party:")
for name in sets:
    print(name, (table[name] > table["prev_party"]).sum())