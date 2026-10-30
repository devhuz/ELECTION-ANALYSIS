import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("model_data.csv")
features = ["prev_margin", "prev_turnout"]
data = data.dropna(subset=features)

rows = []
for year in [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]:
    # Study only elections before this year
    train = data[data["Year"] < year]
    test = data[data["Year"] == year]

    model = LogisticRegression(max_iter=1000)
    model.fit(train[features], train["retained"])
    pred = model.predict(test[features])

    # Fair baseline: guess whatever was more common in the training years
    majority = 1 if train["retained"].mean() >= 0.5 else 0
    base_acc = (test["retained"] == majority).mean()

    rows.append({
        "year": year,
        "test_rows": len(test),
        "actual_retained": round(test["retained"].mean(), 2),
        "baseline": round(base_acc, 2),
        "model": round(accuracy_score(test["retained"], pred), 2),
        "model_says_retained": round(pred.mean(), 2),
    })

print(pd.DataFrame(rows).to_string(index=False))