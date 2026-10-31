import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("candidate_data.csv")
features = ["prior_contests", "prior_wins", "is_incumbent", "party_held_seat",
            "party_streak", "seat_margin", "seat_n_cand", "party_prev_share",
            "state_prev_share", "party_seat_prev_vote", "is_reserved"]
data = data.dropna(subset=features)
keys = ["State_Name", "Constituency_Name"]
years = [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]


def seat_probs(test, raw):
    # Turn raw scores into probabilities that add up to 1 inside each seat
    t = test[keys + ["won"]].copy()
    t["raw"] = raw
    t["p"] = t["raw"] / t.groupby(keys)["raw"].transform("sum")
    return t


rows = []
calib = []
for year in years:
    train = data[data["Year"] < year]
    test = data[data["Year"] == year]

    rates = train.groupby("party_held_seat")["won"].mean()
    models = {
        "uniform": np.ones(len(test)),
        "prev_party_rate": test["party_held_seat"].map(rates).values,
    }
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(train[features], train["won"])
    models["logreg"] = lr.predict_proba(test[features])[:, 1]

    row = {"year": year}
    for name, raw in models.items():
        t = seat_probs(test, raw)
        pw = t[t["won"] == 1]["p"].clip(lower=1e-6)
        row[name] = round(-np.log(pw).mean(), 3)
        if name == "logreg":
            calib.append(t)
    rows.append(row)

res = pd.DataFrame(rows)
print("Log loss on the winner (lower is better):")
print(res.to_string(index=False))
print("Average:")
print(res[["uniform", "prev_party_rate", "logreg"]].mean().round(3))

allp = pd.concat(calib)
allp["band"] = pd.cut(allp["p"], [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0])
print("Calibration (logreg, all test years pooled):")
print(allp.groupby("band", observed=True).agg(
    predicted=("p", "mean"), actual=("won", "mean"), n=("won", "size")).round(3))