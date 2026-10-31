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
all_years = sorted(data["Year"].unique())


def seat_probs(df, raw):
    t = df[keys + ["won"]].copy()
    t["p"] = raw
    t["p"] = t["p"] / t.groupby(keys)["p"].transform("sum")
    return t


rows = []
pooled = []
for year in [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]:
    c = max(y for y in all_years if y < year)
    train = data[data["Year"] < c]
    calib = data[data["Year"] == c]
    test = data[data["Year"] == year]

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[features], train["won"])

    # Learn the correction on the previous election
    tc = seat_probs(calib, model.predict_proba(calib[features])[:, 1])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(tc["p"], tc["won"])

    # Apply to the test year
    t = seat_probs(test, model.predict_proba(test[features])[:, 1])
    t["p_cal"] = iso.predict(t["p"])
    t["p_cal"] = t["p_cal"] / t.groupby(keys)["p_cal"].transform("sum")

    row = {"year": year}
    for name, col in [("uncalibrated", "p"), ("calibrated", "p_cal")]:
        pw = t[t["won"] == 1][col].clip(lower=1e-6)
        row[name] = round(-np.log(pw).mean(), 3)
    rows.append(row)
    pooled.append(t)

res = pd.DataFrame(rows)
print("Log loss on the winner (lower is better):")
print(res.to_string(index=False))
print("Average:")
print(res[["uncalibrated", "calibrated"]].mean().round(3))

allp = pd.concat(pooled)
bands = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
for col in ["p", "p_cal"]:
    allp["band"] = pd.cut(allp[col], bands)
    print("Calibration using", col)
    print(allp.groupby("band", observed=True).agg(
        predicted=(col, "mean"), actual=("won", "mean"),
        n=("won", "size")).round(3))