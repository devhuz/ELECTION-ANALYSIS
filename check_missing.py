import pandas as pd

df = pd.read_csv("data.csv", low_memory=False)

# Problem A: who are the rows with no pid?
missing = df[df["pid"].isnull()]
print("Position of rows with no pid:")
print(missing["Position"].value_counts().head(5))

winners = df[df["Position"] == 1]
share = winners.groupby("Year")["pid"].apply(lambda s: s.isnull().mean())
print("Share of each year's winners with no pid (only years above 0):")
print(share[share > 0].round(2))

# Problem B: the 1984 Gwalior contest
g84 = df[(df["Year"] == 1984) &
         (df["Constituency_Name"].str.contains("GWALIOR", case=False, na=False))]
print("Gwalior 1984, top 3 candidates:")
print(g84.sort_values("Position")[["Candidate", "Party", "Position",
                                   "pid", "Contested"]].head(3).to_string(index=False))