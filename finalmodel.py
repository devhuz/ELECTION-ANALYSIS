import joblib
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


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def seat_norm(df, raw):
    t = df[keys + ["won"]].copy()
    t["p"] = raw
    t["p"] = t["p"] / t.groupby(keys)["p"].transform("sum")
    return t


# 1. Honest predictions for each election, from a model trained only on earlier years
past = []
for y in [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]:
    train = data[data["Year"] < y]
    test = data[data["Year"] == y]
    m = make_model().fit(train[features], train["won"])
    past.append(seat_norm(test, m.predict_proba(test[features])[:, 1]))
past = pd.concat(past)

# 2. Learn the correction from all of them pooled
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
iso.fit(past["p"], past["won"])

# 3. Final model: trained on every election we have
final = make_model().fit(data[features], data["won"])

joblib.dump({"model": final, "calibrator": iso, "features": features},
            "election_model.joblib")
print("Saved election_model.joblib")

# 4. Demo only: load it back and score one seat.
# 2019 was part of the training data, so this checks the file works, not accuracy.
saved = joblib.load("election_model.joblib")
d19 = data[data["Year"] == 2019]
first = d19[keys].drop_duplicates().iloc[0]
s = d19[(d19["State_Name"] == first["State_Name"]) &
        (d19["Constituency_Name"] == first["Constituency_Name"])].copy()
raw = saved["model"].predict_proba(s[saved["features"]])[:, 1]
s["p"] = raw / raw.sum()
s["chance"] = saved["calibrator"].predict(s["p"])
s["chance"] = s["chance"] / s["chance"].sum()
print("Demo seat:", first["Constituency_Name"], "-", first["State_Name"])
print(s[["Candidate", "Party", "won", "p", "chance"]].round(3).to_string(index=False))
print("Chances add up to:", round(s["chance"].sum(), 3))