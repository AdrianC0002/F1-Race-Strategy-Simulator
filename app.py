import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json

# Model and metadata loaded
model = joblib.load("final_xgb_model.pkl")

with open("metadata.json", "r") as f:
    metadata = json.load(f)

track_laps = metadata["track_laps"]
default_driver = metadata["driver"]
default_team = metadata["team"]
pit_loss = metadata["pit_loss"]

# Same columns used in training
cat_cols = ["gp", "compound", "driver", "team"]
num_cols = ["lap_number", "stint", "tyre_life", "tyre_life_sq", "stops_so_far"]


# Function to build race laps
def build_race_rows(gp, driver, team, total_laps, pit_laps, compounds):
    rows = []

    current_stint = 1
    tyre_life = 1
    pit_index = 0

    for lap in range(1, total_laps + 1):
        current_compound = compounds[current_stint - 1]

        rows.append({
            "gp": gp,
            "compound": current_compound,
            "driver": driver,
            "team": team,
            "lap_number": lap,
            "stint": current_stint,
            "tyre_life": tyre_life,
            "tyre_life_sq": tyre_life ** 2,
            "stops_so_far": current_stint - 1
        })

        tyre_life += 1

        if pit_index < len(pit_laps) and lap == pit_laps[pit_index]:
            current_stint += 1
            tyre_life = 1
            pit_index += 1

    return pd.DataFrame(rows)


# Function to simulate strategy
def simulate_strategy(gp, driver, team, total_laps, pit_laps, compounds):
    sim_df = build_race_rows(gp, driver, team, total_laps, pit_laps, compounds)

    sim_df["predicted_lap_time"] = model.predict(sim_df[cat_cols + num_cols])
    sim_df["cumulative_time"] = sim_df["predicted_lap_time"].cumsum()

    total_time = sim_df["predicted_lap_time"].sum() + pit_loss * len(pit_laps)

    return sim_df, total_time


# Streamlit page
st.title("Formula 1 Race Strategy Simulator")
st.write("Explore and compare Formula 1 race strategy scenarios.")
st.write(f"Fixed driver/team: {default_driver} / {default_team}")

# Track selection
selected_track = st.selectbox(
    "Select track",
    ["Bahrain", "Saudi", "Spain", "Monza", "AbuDhabi"]
)

total_laps = track_laps[selected_track]

# Strategy inputs
st.subheader("Enter strategy inputs")

compound_text = st.text_input(
    "Compounds in order (example: SOFT,HARD)",
    value="SOFT,HARD"
)

pit_lap_text = st.text_input(
    "Pit laps (example: 12 or 10,32)",
    value="12"
)

# Button
if st.button("Run Strategy"):
    compounds = [x.strip().upper() for x in compound_text.split(",") if x.strip()]

    if pit_lap_text.strip() == "":
        pit_laps = []
    else:
        pit_laps = [int(x.strip()) for x in pit_lap_text.split(",") if x.strip()]

    # Simple checks
    if len(compounds) != len(pit_laps) + 1:
        st.error("Number of compounds must equal number of stints.")
    elif any(c not in ["SOFT", "MEDIUM", "HARD"] for c in compounds):
        st.error("Only SOFT, MEDIUM, and HARD are allowed.")
    elif any(lap < 1 or lap >= total_laps for lap in pit_laps):
        st.error("Pit laps must be inside the race length.")
    else:
        sim_df, total_time = simulate_strategy(
            gp=selected_track,
            driver=default_driver,
            team=default_team,
            total_laps=total_laps,
            pit_laps=pit_laps,
            compounds=compounds
        )

        st.success(f"Predicted total race time: {total_time:.2f} seconds")

        st.subheader("Lap-by-lap predictions")
        st.dataframe(
            sim_df[["lap_number", "compound", "stint", "tyre_life", "predicted_lap_time"]].round(2),
            use_container_width=True
        )

        # Plot 1: predicted lap time
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(sim_df["lap_number"], sim_df["predicted_lap_time"])
        ax1.set_xlabel("Lap Number")
        ax1.set_ylabel("Predicted Lap Time (s)")
        ax1.set_title("Predicted Lap Time by Lap")
        ax1.grid(True)
        st.pyplot(fig1)

        # Plot 2: cumulative time
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(sim_df["lap_number"], sim_df["cumulative_time"])
        ax2.set_xlabel("Lap Number")
        ax2.set_ylabel("Cumulative Predicted Lap Time (s)")
        ax2.set_title("Cumulative Race Time")
        ax2.grid(True)
        st.pyplot(fig2)