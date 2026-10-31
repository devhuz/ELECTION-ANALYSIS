import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("model_data2.csv")

old = ["prev_margin", "prev_turnout"]
new = ["prev_margin", "prev_turnout", "prev_vote_share", "prev_n_cand",
       "prev_streak", "prev_is_INC", "prev_is_BJP", "is_reserved"]
data = data.dropna(subset=new)

rows = []
for year in [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]:
    train = data[data["Year"] < year]
    test = data[data["Year"] == year]
    y_train = train["retained"]
    y_test = test["retained"]

    # Fair baseline: the more common answer in the training years
    majority = 1 if y_train.mean() >= 0.5 else 0
    baseline = (y_test == majority).mean()

    # Logistic regression, old two features
    lr_old = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr_old.fit(train[old], y_train)
    acc_old = (lr_old.predict(test[old]) == y_test).mean()

    # Logistic regression, all new features
    lr_new = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr_new.fit(train[new], y_train)
    acc_new = (lr_new.predict(test[new]) == y_test).mean()

    # Random forest, all new features
    forest = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=0
    )
    forest.fit(train[new], y_train)
    acc_forest = (forest.predict(test[new]) == y_test).mean()

    rows.append({
        "year": year,
        "test_rows": len(test),
        "baseline": round(baseline, 3),
        "lr_old": round(acc_old, 3),
        "lr_new": round(acc_new, 3),
        "forest": round(acc_forest, 3),
    })

results = pd.DataFrame(rows)
print(results.to_string(index=False))
print("Average over all years:")
print(results[["baseline", "lr_old", "lr_new", "forest"]].mean().round(3))