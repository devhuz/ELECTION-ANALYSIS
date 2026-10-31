import pandas as pd

df = pd.read_csv("data.csv", low_memory=False)
ge_years = [1962, 1967, 1971, 1977, 1980, 1984, 1989, 1991,
            1996, 1998, 1999, 2004, 2009, 2014, 2019]

# 1. Follow one person by ID, not by name
one = df[df["pid"] == "GEDL60727"]
print("All rows for one person (by pid):")
print(one[["Year", "Constituency_Name", "Candidate", "Party",
           "Position", "Incumbent", "Contested", "No_Terms"]]
      .sort_values("Year").to_string(index=False))

# 2. Where are the rows with no person info?
missing = df[df["pid"].isnull()]
print("Rows with no pid, top years:")
print(missing["Year"].value_counts().head(8))

# 3. How lopsided is the target?
print("Share of candidates who won:", round((df["Position"] == 1).mean(), 3))
print("Share of people with only one row:",
      round((df["pid"].value_counts() == 1).mean(), 3))

# 4. Incumbent: how many per general election, and how often do they win?
ge = df[df["Year"].isin(ge_years)]
print("Incumbents per general election:")
print(ge[ge["Incumbent"] == True].groupby("Year").size())
print("Win rate by Incumbent flag:")
print(ge.groupby("Incumbent")["Position"].apply(lambda s: (s == 1).mean()).round(3))