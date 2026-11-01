import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Winning-chance analyzer", layout="wide")

FEATURES = ["prior_contests", "prior_wins", "is_incumbent", "party_held_seat",
            "party_streak", "seat_margin", "seat_n_cand", "party_prev_share",
            "state_prev_share", "party_seat_prev_vote", "is_reserved"]
KEYS = ["State_Name", "Constituency_Name"]
TEST_YEARS = [1989, 1991, 1996, 1998, 1999, 2004, 2009, 2014, 2019]


@st.cache_data
def load_data():
    df = pd.read_csv("candidate_data.csv")
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def seat_norm(df, raw):
    # Turn raw scores into chances that add up to 1 inside each seat
    t = df[KEYS + ["won"]].copy()
    t["p"] = raw
    t["p"] = t["p"] / t.groupby(KEYS)["p"].transform("sum")
    return t


@st.cache_resource(show_spinner="Training on earlier elections...")
def honest_predictions(year):
    # Predictions for one election from a model that only saw earlier elections
    data = load_data()
    train = data[data["Year"] < year]
    test = data[data["Year"] == year]
    model = make_model().fit(train[FEATURES], train["won"])
    return seat_norm(test, model.predict_proba(test[FEATURES])[:, 1])


@st.cache_resource(show_spinner="Training on earlier elections...")
def model_for(year):
    data = load_data()
    train = data[data["Year"] < year]
    model = make_model().fit(train[FEATURES], train["won"])
    earlier = [y for y in TEST_YEARS if y < year]
    iso = None
    if len(earlier) >= 2:
        past = pd.concat([honest_predictions(y) for y in earlier])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        iso.fit(past["p"], past["won"])
    return model, iso


def chances(seat_rows, year):
    # Chance (in percent) for every candidate in one seat
    model, iso = model_for(year)
    t = seat_norm(seat_rows, model.predict_proba(seat_rows[FEATURES])[:, 1])
    out = t["p"]
    if iso is not None:
        out = pd.Series(iso.predict(t["p"]), index=t.index)
        out = out / out.sum()
    return out * 100


NEW_PARTY = "New or other party"


@st.cache_data
def load_field():
    return pd.read_csv("forecast_field.csv")


@st.cache_resource
def load_saved_model():
    return joblib.load("election_model.joblib")


def forecast_chances(rows):
    # Chance (in percent) for every candidate in one seat, final model
    saved = load_saved_model()
    raw = saved["model"].predict_proba(rows[saved["features"]])[:, 1]
    p = raw / raw.sum()
    cal = saved["calibrator"].predict(p)
    cal = cal / cal.sum()
    return pd.Series(cal * 100, index=rows.index)


def build_user_row(field, field_seat, party, contests, wins, sitting):
    base = field_seat.iloc[0]
    held = int(party == base["seat_party"])
    same_seat = field_seat[field_seat["Party"] == party]
    same_state = field[(field["State_Name"] == base["State_Name"]) &
                       (field["Party"] == party)]
    anywhere = field[field["Party"] == party]
    return {
        "State_Name": base["State_Name"],
        "Constituency_Name": base["Constituency_Name"],
        "Candidate": "You", "Party": party, "pid": "you",
        "seat_party": base["seat_party"], "seat_streak": base["seat_streak"],
        "prior_contests": contests, "prior_wins": wins,
        "is_incumbent": int(sitting), "party_held_seat": held,
        "party_streak": base["seat_streak"] * held,
        "seat_margin": base["seat_margin"], "seat_n_cand": base["seat_n_cand"],
        "is_reserved": base["is_reserved"],
        "party_seat_prev_vote": same_seat["party_seat_prev_vote"].max() if len(same_seat) else 0.0,
        "party_prev_share": anywhere["party_prev_share"].max() if len(anywhere) else 0.0,
        "state_prev_share": same_state["state_prev_share"].max() if len(same_state) else 0.0,
    }


def with_user(field_seat, user_row):
    rivals = field_seat
    if user_row["Party"] != "IND":
        rivals = rivals[rivals["Party"] != user_row["Party"]]
    return pd.concat([rivals, pd.DataFrame([user_row])], ignore_index=True)


if not os.path.exists("candidate_data.csv"):
    st.error("candidate_data.csv was not found. Run features3.py first, then "
             "start this app from the project folder.")
    st.stop()

data = load_data()

st.title("Winning-chance analyzer")
st.caption("Lok Sabha general elections, 1962 to 2019. Estimates from past "
           "data, using the 2019 election as the latest starting point. They are "
           "not guarantees.")

with st.sidebar:
    st.header("Choose a seat")
    year = st.selectbox("Election year", sorted(TEST_YEARS, reverse=True))
    year_rows = data[data["Year"] == year]
    state = st.selectbox("State", sorted(year_rows["State_Name"].unique()))
    seats = sorted(year_rows[year_rows["State_Name"] == state]["Constituency_Name"].unique())
    seat = st.selectbox("Seat", seats)
    st.write("This panel applies to the Seat replay and What-if tabs. The "
             f"model is trained only on elections before {year}, so it never "
             "sees the result it is scoring.")

seat_rows = year_rows[(year_rows["State_Name"] == state) &
                      (year_rows["Constituency_Name"] == seat)].copy()

section = st.radio(
    "Section",
    ["Next election", "Seat replay", "What-if", "Model results", "Limits"],
    horizontal=True, label_visibility="collapsed")

progress = st.column_config.ProgressColumn(
    "Chance (%)", min_value=0, max_value=100, format="%.1f")

if section == "Next election":
    st.subheader("Chances in the next election")
    st.write("This starts from the 2019 election, the latest in the data. It "
             "assumes the 2019 candidates stand again for the same parties. "
             "Add yourself below to see your own chance.")
    if not (os.path.exists("forecast_field.csv")
            and os.path.exists("election_model.joblib")):
        st.warning("Files are missing. Run features_future.py and "
                   "final_model.py first, then restart this app.")
    else:
        field = load_field()
        f1, f2 = st.columns(2)
        f_state = f1.selectbox("State", sorted(field["State_Name"].unique()),
                               key="f_state")
        f_seats = sorted(field[field["State_Name"] == f_state]["Constituency_Name"].unique())
        f_seat = f2.selectbox("Seat", f_seats, key="f_seat")
        field_seat = field[(field["State_Name"] == f_state) &
                           (field["Constituency_Name"] == f_seat)].reset_index(drop=True)

        outlook = field_seat[["Candidate", "Party", "is_incumbent"]].copy()
        outlook["Sitting MP"] = outlook["is_incumbent"].map({1: "Yes", 0: ""})
        outlook["Chance (%)"] = forecast_chances(field_seat).round(1)
        outlook = outlook.sort_values("Chance (%)", ascending=False)
        outlook = outlook.drop(columns="is_incumbent").reset_index(drop=True)
        st.markdown("**Outlook if the 2019 candidates stand again**")
        st.dataframe(outlook.head(8), hide_index=True,
                     column_config={"Chance (%)": progress})
        first = field_seat.iloc[0]
        st.write(f"In 2019 this seat was won by {first['seat_party']}, by a "
                 f"margin of {first['seat_margin']:.1f}%, with "
                 f"{int(first['seat_n_cand'])} candidates. That party has held "
                 f"it for {int(first['seat_streak'])} election(s) in a row.")

        st.markdown("**Add yourself**")
        seat_parties = (field_seat.sort_values("party_seat_prev_vote", ascending=False)
                        ["Party"].astype(str).drop_duplicates().tolist())
        other_parties = sorted(set(field["Party"].astype(str)) - set(seat_parties))
        p1, p2, p3, p4 = st.columns(4)
        party = p1.selectbox("Your party", seat_parties + other_parties + [NEW_PARTY],
                             key="f_party")
        contests = p2.number_input("Elections contested before", 0, 20, 0,
                                   key="f_contests")
        wins = p3.number_input("Elections won before", 0, 20, 0, key="f_wins")
        sitting = p4.checkbox("Sitting MP of this seat", key="f_sitting")
        if wins > contests:
            st.warning("Wins cannot be higher than contests. Using contests = wins.")
            contests = wins
        if sitting and wins < 1:
            st.warning("A sitting MP has won at least once. Using 1 win and 1 contest.")
            wins, contests = 1, max(contests, 1)

        user = build_user_row(field, field_seat, party, contests, wins, sitting)
        rows = with_user(field_seat, user)
        ch = forecast_chances(rows)
        mine = float(ch[rows["pid"] == "you"].iloc[0])
        rank = int((ch > mine).sum()) + 1
        rival = rows.assign(chance=ch)
        rival = rival[rival["pid"] != "you"].sort_values("chance", ascending=False).iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Your chance", f"{mine:.1f}%")
        m2.metric("Rank in this seat", f"{rank} of {len(rows)}")
        m3.metric("Strongest rival", f"{rival['Candidate']} ({rival['Party']})",
                  f"{rival['chance']:.1f}%", delta_color="off")

        st.markdown("**Your chance on different party tickets**")
        tickets = []
        for t_party in seat_parties[:6] + [NEW_PARTY]:
            u = build_user_row(field, field_seat, t_party, contests, wins, sitting)
            r = with_user(field_seat, u)
            c = forecast_chances(r)
            tickets.append({"Party": t_party,
                            "Your chance (%)": round(float(c[r["pid"] == "you"].iloc[0]), 1)})
        tickets = pd.DataFrame(tickets).sort_values("Your chance (%)", ascending=False)
        st.dataframe(tickets, hide_index=True)
        st.caption("Each row uses your same personal record on that party's "
                   "ticket. It does not include how switching parties changes "
                   "votes, and it assumes the rivals shown above.")

if section == "Seat replay":
    st.subheader(f"{seat}, {state}, {year}")
    view = seat_rows[["Candidate", "Party", "won"]].copy()
    view["Chance (%)"] = chances(seat_rows, year).round(1)
    view["Result"] = view["won"].map({1: "Won", 0: ""})
    view = view.sort_values("Chance (%)", ascending=False).drop(columns="won")
    view = view.reset_index(drop=True)
    st.dataframe(view.head(8), hide_index=True,
                 column_config={"Chance (%)": progress})

    results = view["Result"].tolist()
    if "Won" in results:
        rank = results.index("Won") + 1
        st.info(f"The actual winner was ranked {rank} of {len(view)} candidates "
                "by the model.")

    first = seat_rows.iloc[0]
    kind = "reserved" if first["is_reserved"] == 1 else "general"
    st.write(f"Going into this election: the last margin in this seat was "
             f"{first['seat_margin']:.1f}%, there were {int(first['seat_n_cand'])} "
             f"candidates last time, and it is a {kind} seat. The longest current "
             f"run for one party is {int(seat_rows['party_streak'].max())} "
             "election(s).")

if section == "What-if":
    st.subheader("Change a candidate's history")
    st.write("Pick a candidate in the seat chosen on the left, then edit their "
             "personal record. Everything else about the seat stays as it was.")
    labels = (seat_rows["Candidate"] + " (" + seat_rows["Party"].astype(str) + ")").tolist()
    choice = st.selectbox("Candidate", labels)
    pos = labels.index(choice)
    idx = seat_rows.index[pos]
    row = seat_rows.loc[idx]

    c1, c2, c3 = st.columns(3)
    contests = c1.number_input("Past elections contested", 0, 20, int(row["prior_contests"]))
    wins = c2.number_input("Past elections won", 0, 20, int(row["prior_wins"]))
    sitting = c3.checkbox("Sitting MP of this seat", value=bool(row["is_incumbent"]))
    if wins > contests:
        st.warning("Wins cannot be higher than contests. Using wins = contests.")
        wins = contests

    edited = seat_rows.copy()
    edited.loc[idx, ["prior_contests", "prior_wins", "is_incumbent"]] = [
        contests, wins, int(sitting)]
    before = chances(seat_rows, year)
    after = chances(edited, year)

    m1, m2 = st.columns(2)
    m1.metric("Chance with original record", f"{before[idx]:.1f}%")
    m2.metric("Chance with your edits", f"{after[idx]:.1f}%",
              delta=f"{after[idx] - before[idx]:+.1f} points")

    cmp = pd.DataFrame({"Candidate": seat_rows["Candidate"],
                        "Party": seat_rows["Party"],
                        "Before (%)": before.round(1),
                        "After (%)": after.round(1)})
    cmp = cmp.sort_values("After (%)", ascending=False).reset_index(drop=True)
    st.dataframe(cmp.head(8), hide_index=True)

if section == "Model results":
    st.subheader("How well does the model do?")
    st.write("Tested year by year: for each election from 1989 to 2019 the model "
             "was trained only on earlier elections. About 510 seats per year.")
    if os.path.exists("results_table.csv"):
        st.dataframe(pd.read_csv("results_table.csv"), hide_index=True)
    st.markdown(
        "**Picking the winner (average of 9 years):** a random candidate is "
        "right 10.4% of the time, the party that held the seat last time 54.8%, "
        "and the model 52.7%. The model does not beat the simple rule here.\n\n"
        "**Quality of the percentages (log loss, lower is better):** equal "
        "chance for everyone 2.447, the simple rule 1.911, the model 1.447. "
        "Recalibration lowers the model's average from 1.380 to 1.249 over "
        "1996 to 2019.")
    pictures = [("chart_top1_by_year.png", "Previous-party rule and model, by year"),
                ("chart_calibration.png", "Do the percentages match reality?"),
                ("chart_top1_average.png", "Average accuracy at picking winners")]
    cols = st.columns(len(pictures))
    for col, (path, caption) in zip(cols, pictures):
        if os.path.exists(path):
            col.image(path, caption=caption)

if section == "Limits":
    st.subheader("What this tool can and cannot tell you")
    st.markdown(
        "**How it works.** For each candidate, the model uses only what was "
        "known before the election: their past contests and wins, whether they "
        "were the sitting MP, whether their party held the seat and for how "
        "long, the party's vote share in the seat and its seat share last "
        "election, and the seat's last margin and field size. It uses no "
        "party names, so it can score any candidate.\n\n"
        "**Limits.**\n"
        "- National swings, such as 1996 and 2014, cannot be seen in past data. "
        "They are the main reason winner-picking stops near 55%.\n"
        "- Chances above about 70% run high: about 79% predicted, 69% actual.\n"
        "- Seats with no earlier election, for example after the 2008 boundary "
        "changes, have no seat history and are left out.\n"
        "- The same person can appear under two IDs, so some histories are "
        "undercounted. Party renamings can look like changes of party.\n"
        "- The Next election tab assumes everyone who stood in 2019 stands "
        "again for the same party, and it ignores the effect of switching parties.\n"
        "- The data ends in 2019. A chance means a candidate like this, in a "
        "typical field in this seat, because rivals are not known in advance.")