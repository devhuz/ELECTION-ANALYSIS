import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("model_data.csv")

# Inputs (features) and the answer (target)
features = ["prev_margin", "prev_turnout"]
data = data.dropna(subset=features)

# Split by time: study up to 2014, be graded on 2019
train = data[data["Year"] <= 2014]
test = data[data["Year"] == 2019]

X_train = train[features]
y_train = train["retained"]
X_test = test[features]
y_test = test["retained"]

print("Training rows:", len(train))
print("Test rows:", len(test))

# Baselines on the test set
print("Always guess retained:", round(y_test.mean(), 3))
print("Always guess flipped:", round(1 - y_test.mean(), 3))

# Train the model, then ask it to predict 2019
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Model accuracy:", round(accuracy_score(y_test, pred), 3))