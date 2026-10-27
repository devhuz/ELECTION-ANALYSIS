import pandas as pd

# 1. Load raw data
df = pd.read_csv("data.csv", low_memory=False)

# 2. Keep only the useful columns
keep = [
    "State_Name", "Constituency_No", "Constituency_Name",
    "Constituency_Type", "Year", "Election_Type", "Position",
    "Candidate", "Sex", "Party", "Votes", "Valid_Votes",
    "Electors", "Turnout_Percentage", "Vote_Share_Percentage",
    "Margin", "Margin_Percentage", "N_Cand", "ENOP",
    "Incumbent", "Party_Type_TCPD",
]
df = df[keep]

# 3. Drop the column that has only one value
df = df.drop(columns=["Election_Type"])

# 4. Fill blank text values
df["Sex"] = df["Sex"].fillna("Unknown")
df["Party"] = df["Party"].fillna("Unknown")

# 5. Split general elections from by-elections
ge_years = [1962, 1967, 1971, 1977, 1980, 1984, 1989, 1991,
            1996, 1998, 1999, 2004, 2009, 2014, 2019]
ge = df[df["Year"].isin(ge_years)]
other = df[~df["Year"].isin(ge_years)]

ge.to_csv("ge_data.csv", index=False)
other.to_csv("other_data.csv", index=False)

print("General election rows:", len(ge))
print("Other rows:", len(other))