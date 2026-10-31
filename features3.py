import pandas as pd

df = pd.read_csv("data.csv", low_memory=False)
ge_years = [1962, 1967, 1971, 1977, 1980, 1984, 1989, 1991,
            1996, 1998, 1999, 2004, 2009, 2014, 2019]
df = df[df["Year"].isin(ge_years)].copy()
df["Party"] = df["Party"].replace({"INC(I)": "INC"})

# Rows with no person ID: who are they?
print("Rows with no pid:", df["pid"].isnull().sum())
print(df[df["pid"].isnull()]["Candidate"].value_counts().head(3))
df = df.dropna(subset=["pid"]).copy()

df["won"] = (df["Position"] == 1).astype(int)
prev_year = dict(zip(ge_years[1:], ge_years[:-1]))
df["prev_year"] = df["Year"].map(prev_year)

# Person history, counting only earlier elections
person = df.groupby(["pid", "Year"])["won"].max().reset_index()
person = person.sort_values(["pid", "Year"])
pg = person.groupby("pid")
person["prior_contests"] = pg.cumcount()
person["prior_wins"] = pg["won"].cumsum() - person["won"]
df = df.merge(person[["pid", "Year", "prior_contests", "prior_wins"]],
              on=["pid", "Year"], how="left")

# Seat history: who won, margin, crowd size, streak (one winner per seat-year)
keys = ["State_Name", "Constituency_Name"]
seat = df[df["won"] == 1].sort_values(keys + ["Year"])
seat = seat.drop_duplicates(subset=keys + ["Year"]).copy()
g = seat.groupby(keys)
seat["changed"] = (seat["Party"] != g["Party"].shift(1)).astype(int)
seat["run_id"] = seat.groupby(keys)["changed"].cumsum()
seat["seat_streak"] = seat.groupby(keys + ["run_id"]).cumcount() + 1
seat = seat.rename(columns={"Party": "seat_party",
                            "Margin_Percentage": "seat_margin",
                            "N_Cand": "seat_n_cand"})
prev = seat[keys + ["Year", "seat_party", "seat_margin",
                    "seat_n_cand", "seat_streak"]]
prev = prev.rename(columns={"Year": "prev_year"})
df = df.merge(prev, on=keys + ["prev_year"], how="left")

# Party strength last election: share of seats won, nationally and in the state
winners = df[df["won"] == 1].drop_duplicates(subset=keys + ["Year"])
nat = winners.groupby(["Year", "Party"]).size().reset_index(name="seats")
nat["party_prev_share"] = nat["seats"] / nat.groupby("Year")["seats"].transform("sum")
nat = nat.rename(columns={"Year": "prev_year"})
df = df.merge(nat[["prev_year", "Party", "party_prev_share"]],
              on=["prev_year", "Party"], how="left")

st = winners.groupby(["Year", "State_Name", "Party"]).size().reset_index(name="seats")
st["state_prev_share"] = st["seats"] / st.groupby(["Year", "State_Name"])["seats"].transform("sum")
st = st.rename(columns={"Year": "prev_year"})
df = df.merge(st[["prev_year", "State_Name", "Party", "state_prev_share"]],
              on=["prev_year", "State_Name", "Party"], how="left")

# This party's vote share in this seat last time (win or lose)
ps = df.groupby(keys + ["Party", "Year"])["Vote_Share_Percentage"].max().reset_index()
ps = ps.rename(columns={"Year": "prev_year",
                        "Vote_Share_Percentage": "party_seat_prev_vote"})
df = df.merge(ps, on=keys + ["Party", "prev_year"], how="left")

# Missing means "no earlier record", which we treat as zero
for col in ["party_prev_share", "state_prev_share", "party_seat_prev_vote"]:
    df[col] = df[col].fillna(0)

df["is_incumbent"] = (df["Incumbent"] == True).astype(int)
df["is_reserved"] = (df["Constituency_Type"] != "GEN").astype(int)
df["party_held_seat"] = (df["Party"] == df["seat_party"]).astype(int)
df["party_streak"] = df["seat_streak"] * df["party_held_seat"]

features = ["prior_contests", "prior_wins", "is_incumbent", "party_held_seat",
            "party_streak", "seat_margin", "seat_n_cand", "party_prev_share",
            "state_prev_share", "party_seat_prev_vote", "is_reserved"]
data = df.dropna(subset=["seat_party", "seat_margin", "seat_n_cand"]).copy()

print("Candidate rows:", len(data))
print("Winners per year:")
print(data.groupby("Year")["won"].sum())
print("Win rate by is_incumbent:")
print(data.groupby("is_incumbent")["won"].mean().round(3))
print("Win rate by party_held_seat:")
print(data.groupby("party_held_seat")["won"].mean().round(3))

cols = ["Year", "State_Name", "Constituency_Name", "Candidate", "Party",
        "pid"] + features + ["won"]
data[cols].to_csv("candidate_data.csv", index=False)