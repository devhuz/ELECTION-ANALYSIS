import pandas as pd

df = pd.read_csv("data.csv", low_memory=False)
cols = ["pid", "Incumbent", "Contested", "No_Terms", "Turncoat",
        "Recontest", "Last_Party", "Same_Constituency", "Same_Party"]

print("Missing values:")
print(df[cols].isnull().sum())

print("Unique people (pid):", df["pid"].nunique())
print("Incumbent values:")
print(df["Incumbent"].value_counts(dropna=False))

# Follow one politician across elections
person = df[df["Candidate"].str.contains("VAJPAYEE", na=False)]
print(person[["Year", "Constituency_Name", "Candidate", "Party", "Position",
              "pid", "Incumbent", "Contested", "No_Terms"]]
      .sort_values("Year").to_string(index=False))