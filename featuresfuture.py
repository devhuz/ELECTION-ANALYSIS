import pandas as pd

# Builds the starting point for "the next election after 2019".
# Seat history and party strength come from 2019, the latest election in the
# data. The pretend field is everyone who stood in 2019, standing again.

df = pd.read_csv("data.csv", low_memory=False)
ge_years = [1962, 1967, 1971, 1977, 1980, 1984, 1989, 1991,
            1996, 1998, 1999, 2004, 2009, 2014, 2019]
FUTURE = 2024  # only a label for the pretend next election

df = df[df["Year"].isin(ge_years)].copy()
df["Party"] = df["Party"].replace({"INC(I)": "INC"})
df = df.dropna(subset=["pid"]).copy()  # removes NOTA rows

# Pretend next election: the 2019 candidates stand again, result unknown.
# Sitting MPs are the 2019 winners.
last = df[df["Year"] == 2019].copy()
fut = last.copy()
fut["Year"] = FUTURE
fut["Incumbent"] = (last["Position"] == 1)
fut["Position"] = 0
df = pd.concat([df, fut], ignore_index=True)

df["won"] = (df["Position"] == 1).astype(int)
all_years = ge_years + [FUTURE]
prev_year = dict(zip(all_years[1:], all_years[:-1]))
df["prev_year"] = df["Year"].map(prev_year)

# Person history: only earlier elections count
person = df.groupby(["pid", "Year"])["won"].max().reset_index()
person = person.sort_values(["pid", "Year"])
pg = person.groupby("pid")
person["prior_contests"] = pg.cumcount()
person["prior_wins"] = pg["won"].cumsum() - person["won"]
df = df.merge(person[["pid", "Year", "prior_contests", "prior_wins"]],
              on=["pid", "Year"], how="left")

# Seat history: winner, margin, crowd size and streak
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

# Party strength in the previous election
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

ps = df.groupby(keys + ["Party", "Year"])["Vote_Share_Percentage"].max().reset_index()
ps = ps.rename(columns={"Year": "prev_year",
                        "Vote_Share_Percentage": "party_seat_prev_vote"})
df = df.merge(ps, on=keys + ["Party", "prev_year"], how="left")

for col in ["party_prev_share", "state_prev_share", "party_seat_prev_vote"]:
    df[col] = df[col].fillna(0)

df["is_incumbent"] = (df["Incumbent"] == True).astype(int)
df["is_reserved"] = (df["Constituency_Type"] != "GEN").astype(int)
df["party_held_seat"] = (df["Party"] == df["seat_party"]).astype(int)
df["party_streak"] = df["seat_streak"] * df["party_held_seat"]

features = ["prior_contests", "prior_wins", "is_incumbent", "party_held_seat",
            "party_streak", "seat_margin", "seat_n_cand", "party_prev_share",
            "state_prev_share", "party_seat_prev_vote", "is_reserved"]

field = df[df["Year"] == FUTURE]
field = field.dropna(subset=["seat_party", "seat_margin", "seat_n_cand"]).copy()
cols = ["State_Name", "Constituency_Name", "Candidate", "Party", "pid",
        "seat_party", "seat_streak"] + features
field[cols].to_csv("forecast_field.csv", index=False)

print("Candidates in the pretend next election:", len(field))
print("Seats:", field.groupby(keys).ngroups)
print("Sitting MPs (2019 winners):", int(field["is_incumbent"].sum()))
print("Seat history comes from the 2019 election.")