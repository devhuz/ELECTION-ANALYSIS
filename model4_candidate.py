import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("candidate_data.csv")
features = ["prior_contests", "prior_wins", "is_incumbent", "party_held_seat",
            "party_streak", "seat_margin", "seat_n_cand", "party_prev_share",
            "state_prev_share", "party_seat_prev_vote", "is_reserved"]
data = data.dropna(subset=features)
keys = ["State_Name", "Constituency_Name"]


def top1(test, scores):
    # In each seat, pick the highest-scoring candidate. Share of seats where that pick won.
    t = test[keys + ["won"]].copy()
    t["score"] = scores
    picks = t.groupby(keys)["score"].idxmax()
    return t.loc[picks, "won"].mean()


rows = []
for year in [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]:
    train = data[data["Year"] < year]
    test = data[data["Year"] == year]

    # Random pick: on average 1 divided by the number of candidates in the seat
    random_pick = test.groupby(keys)["won"].mean().mean()
    prev_party = top1(test, test["party_held_seat"].values)
    inc_party = top1(test, (test["is_incumbent"] * 2 + test["party_held_seat"]).values)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(train[features], train["won"])
    lr_pick = top1(test, lr.predict_proba(test[features])[:, 1])

    forest = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    min_samples_leaf=20, random_state=0, n_jobs=-1)
    forest.fit(train[features], train["won"])
    forest_pick = top1(test, forest.predict_proba(test[features])[:, 1])

    rows.append({
        "year": year,
        "seats": test.groupby(keys).ngroups,
        "random": round(random_pick, 3),
        "prev_party": round(prev_party, 3),
        "inc_else_party": round(inc_party, 3),
        "logreg": round(lr_pick, 3),
        "forest": round(forest_pick, 3),
    })

results = pd.DataFrame(rows)
print(results.to_string(index=False))
print("Average over all years:")
print(results[["random", "prev_party", "inc_else_party", "logreg", "forest"]].mean().round(3))