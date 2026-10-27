import pandas as pd

ge = pd.read_csv("ge_data.csv", low_memory=False)

# Fix 1: turnout above 100% is impossible, so mark it as missing
ge.loc[ge["Turnout_Percentage"] > 100, "Turnout_Percentage"] = None

# One row per constituency: keep only winners
winners = ge[ge["Position"] == 1]

# Question 1: average turnout in each election
turnout = winners.groupby("Year")["Turnout_Percentage"].mean().round(1)
print("Average turnout by year:")
print(turnout)

# Question 2: narrowest wins, contested seats only
contested = winners[winners["N_Cand"] > 1]
close = contested.sort_values("Margin").head(10)
print("10 narrowest wins (contested seats):")
print(close[["Year", "State_Name", "Constituency_Name",
             "Candidate", "Party", "Margin", "N_Cand"]])