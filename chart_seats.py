import pandas as pd
import matplotlib.pyplot as plt

ge = pd.read_csv("ge_data.csv", low_memory=False)
winners = ge[ge["Position"] == 1]

seats = (
    winners[winners["Party"].isin(["INC", "BJP"])]
    .groupby(["Year", "Party"])
    .size()
    .unstack(fill_value=0)
)
print(seats)

seats.plot(marker="o", figsize=(10, 5))
plt.title("Lok Sabha seats won: INC vs BJP")
plt.xlabel("Election year")
plt.ylabel("Seats won")
plt.grid(True)
plt.savefig("seats_inc_bjp.png")