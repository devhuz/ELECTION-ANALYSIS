import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("candidate_data.csv")
features = ["prior_contests", "prior_wins", "is_incumbent", "party_held_seat",
            "party_streak", "seat_margin", "seat_n_cand", "party_prev_share",
            "state_prev_share", "party_seat_prev_vote", "is_reserved"]
data = data.dropna(subset=features)
keys = ["State_Name", "Constituency_Name"]


def seat_probs(df, raw):
    t = df[keys + ["won"]].copy()
    t["p"] = raw
    t["p"] = t["p"] / t.groupby(keys)["p"].transform("sum")
    return t


def renorm(t, col):
    return t[col] / t.groupby(keys)[col].transform("sum")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


# Out-of-sample predictions for each election from 1989 on
oos = {}
for y in [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]:
    train = data[data["Year"] < y]
    test = data[data["Year"] == y]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[features], train["won"])
    oos[y] = seat_probs(test, model.predict_proba(test[features])[:, 1])

rows = []
pooled = []
for year in [1996, 1998, 1999, 2004, 2009, 2014, 2019]:
    # Learn the correction from all earlier elections' predictions
    past = pd.concat([oos[y] for y in oos if y < year])
    t = oos[year].copy()

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(past["p"], past["won"])
    t["isotonic"] = iso.predict(t["p"])
    t["isotonic"] = renorm(t, "isotonic")

    sig = LogisticRegression()
    sig.fit(logit(past["p"].values), past["won"])
    t["sigmoid"] = sig.predict_proba(logit(t["p"].values))[:, 1]
    t["sigmoid"] = renorm(t, "sigmoid")

    row = {"year": year, "past_elections": len([y for y in oos if y < year])}
    for label, col in [("uncalibrated", "p"), ("isotonic", "isotonic"),
                       ("sigmoid", "sigmoid")]:
        pw = t[t["won"] == 1][col].clip(lower=1e-6)
        row[label] = round(-np.log(pw).mean(), 3)
    rows.append(row)
    pooled.append(t)

res = pd.DataFrame(rows)
print("Log loss on the winner (lower is better):")
print(res.to_string(index=False))
print("Average:")
print(res[["uncalibrated", "isotonic", "sigmoid"]].mean().round(3))

allp = pd.concat(pooled)
bands = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
for col in ["p", "isotonic", "sigmoid"]:
    allp["band"] = pd.cut(allp[col], bands)
    print("Calibration using", col)
    print(allp.groupby("band", observed=True).agg(
        predicted=(col, "mean"), actual=("won", "mean"),
        n=("won", "size")).round(3))