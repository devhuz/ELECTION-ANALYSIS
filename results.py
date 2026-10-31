import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
years = [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def seat_norm(df, raw):
    t = df[keys + ["won"]].copy()
    t["p"] = raw
    t["p"] = t["p"] / t.groupby(keys)["p"].transform("sum")
    return t


def top1(test, scores):
    t = test[keys + ["won"]].copy()
    t["score"] = scores
    picks = t.groupby(keys)["score"].idxmax()
    return t.loc[picks, "won"].mean()


def loss(t, col):
    pw = t[t["won"] == 1][col].clip(lower=1e-6)
    return -np.log(pw).mean()


oos = {}
rows = []
for year in years:
    train = data[data["Year"] < year]
    test = data[data["Year"] == year]
    m = make_model().fit(train[features], train["won"])
    raw = m.predict_proba(test[features])[:, 1]
    oos[year] = seat_norm(test, raw)

    rates = train.groupby("party_held_seat")["won"].mean()
    rule = seat_norm(test, test["party_held_seat"].map(rates).values)
    uniform = seat_norm(test, np.ones(len(test)))

    rows.append({
        "year": year,
        "seats": test.groupby(keys).ngroups,
        "random": test.groupby(keys)["won"].mean().mean(),
        "prev_party": top1(test, test["party_held_seat"].values),
        "model": top1(test, raw),
        "loss_uniform": loss(uniform, "p"),
        "loss_rule": loss(rule, "p"),
        "loss_model": loss(oos[year], "p"),
    })
res = pd.DataFrame(rows)

# Recalibration from all earlier elections pooled (test years 1996 onward)
cal = []
res["loss_calibrated"] = np.nan
for i, year in enumerate(years):
    if year < 1996:
        continue
    past = pd.concat([oos[y] for y in years if y < year])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(past["p"], past["won"])
    t = oos[year].copy()
    t["iso"] = iso.predict(t["p"])
    t["iso"] = t["iso"] / t.groupby(keys)["iso"].transform("sum")
    res.loc[i, "loss_calibrated"] = loss(t, "iso")
    cal.append(t)

print(res.round(3).to_string(index=False))
print("Average over all years (loss_calibrated: 1996-2019 only):")
print(res.drop(columns=["year", "seats"]).mean().round(3))
late = res[res["year"] >= 1996]
print("1996-2019 only:")
print(late[["loss_model", "loss_calibrated"]].mean().round(3))
res.round(3).to_csv("results_table.csv", index=False)

# Chart 1: picking the winner
avg = res[["random", "prev_party", "model"]].mean()
plt.figure(figsize=(7, 4))
plt.bar(["Random pick", "Previous party", "Model"], avg.values)
plt.ylabel("Seats where the pick won")
plt.title("Picking the winner: average over 1989-2019")
plt.ylim(0, 0.7)
plt.savefig("chart_top1_average.png", dpi=150, bbox_inches="tight")
plt.close()

# Chart 2: by year (wave years show up as dips)
plt.figure(figsize=(8, 4.5))
plt.plot(res["year"], res["prev_party"], marker="o", label="Previous party")
plt.plot(res["year"], res["model"], marker="o", label="Model")
plt.xlabel("Election year")
plt.ylabel("Seats where the pick won")
plt.title("Previous party rule vs model, by election year")
plt.legend()
plt.grid(True)
plt.savefig("chart_top1_by_year.png", dpi=150, bbox_inches="tight")
plt.close()

# Chart 3: are the percentages honest?
allp = pd.concat(cal)
bands = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
plt.figure(figsize=(5.5, 5.5))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
for col, label in [("p", "Before correction"), ("iso", "After correction")]:
    band = pd.cut(allp[col], bands)
    g = allp.groupby(band, observed=True).agg(
        pred=(col, "mean"), actual=("won", "mean"))
    plt.plot(g["pred"], g["actual"], marker="o", label=label)
plt.xlabel("Chance the model gave")
plt.ylabel("Share that actually won")
plt.title("Calibration, 1996-2019")
plt.legend()
plt.grid(True)
plt.savefig("chart_calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 3 charts and results_table.csv")