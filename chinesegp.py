import os
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

import fastf1
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

YEAR = 2026
TARGET_RACE = "Chinese Grand Prix"

# Races before the target race
PREVIOUS_RACES = [
    "Australian Grand Prix"
]

# Track type score: low=1, medium=2, high=3
TRACK_DOWNFORCE = {
    "Australian Grand Prix": 2,
    "Chinese Grand Prix": 2
}

CACHE_FOLDER = "f1_cache"


# Creates cache folder and enables FastF1 caching for faster data loading
def setup_fastf1_cache():
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_FOLDER)


# Loads race session data for a given year and race
def get_race_session(year, race_name):
    print(f"Loading race session: {race_name}")
    session = fastf1.get_session(year, race_name, "R")
    session.load()
    return session


# Loads qualifying session data for a given race
def get_qualifying_session(year, race_name):
    print(f"Loading qualifying session: {race_name}")
    session = fastf1.get_session(year, race_name, "Q")
    session.load()
    return session


# Cleans lap data and converts lap and sector times into seconds
def clean_lap_data(session):
    laps = session.laps.copy()

    needed_columns = [
        "Driver",
        "LapNumber",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Stint",
        "TyreLife",
        "Compound"
    ]

    available_columns = [col for col in needed_columns if col in laps.columns]
    laps = laps[available_columns].copy()

    laps = laps.dropna(subset=["Driver", "LapTime", "Stint"])
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    for sector in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if sector in laps.columns:
            laps[f"{sector}_s"] = laps[sector].dt.total_seconds()

    return laps


# Calculates tire degradation by comparing early vs late laps in each stint
def calculate_tire_degradation(driver_laps):
    degradation_values = []

    for _, stint_laps in driver_laps.groupby("Stint"):
        stint_laps = stint_laps.sort_values("LapNumber")

        if len(stint_laps) >= 6:
            first_three = stint_laps.head(3)["LapTime_s"].mean()
            last_three = stint_laps.tail(3)["LapTime_s"].mean()
            degradation_values.append(last_three - first_three)

    if degradation_values:
        return float(np.mean(degradation_values))

    return np.nan


# Calculates average pace across all stints for a driver
def calculate_average_stint_pace(driver_laps):
    stint_averages = []

    for _, stint_laps in driver_laps.groupby("Stint"):
        if len(stint_laps) >= 3:
            stint_averages.append(stint_laps["LapTime_s"].mean())

    if stint_averages:
        return float(np.mean(stint_averages))

    return np.nan


# Builds performance metrics for each driver from a single race
def get_driver_features_from_race(session, race_name):
    laps = clean_lap_data(session)
    driver_rows = []

    for driver_code in laps["Driver"].dropna().unique():
        driver_laps = laps[laps["Driver"] == driver_code].copy()

        driver_row = {
            "Driver": driver_code,
            "Race": race_name,
            "AvgLapTime_s": driver_laps["LapTime_s"].mean(),
            "LapStd_s": driver_laps["LapTime_s"].std(),
            "LapVar_s": driver_laps["LapTime_s"].var(),
            "TireDeg_s": calculate_tire_degradation(driver_laps),
            "AvgStintPace_s": calculate_average_stint_pace(driver_laps),
            "TotalLaps": len(driver_laps),
            "DownforceScore": TRACK_DOWNFORCE.get(race_name, 2)
        }

        for sector_col in ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]:
            if sector_col in driver_laps.columns:
                driver_row[f"Mean_{sector_col}"] = driver_laps[sector_col].mean()
            else:
                driver_row[f"Mean_{sector_col}"] = np.nan

        driver_rows.append(driver_row)

    results = pd.DataFrame(driver_rows)
    results["LapStd_s"] = results["LapStd_s"].fillna(0.0)
    results["LapVar_s"] = results["LapVar_s"].fillna(0.0)

    return results


# Combines driver data from multiple previous races into one dataset
def build_previous_race_dataset(year, race_list):
    all_race_data = []

    for race_name in race_list:
        try:
            race_session = get_race_session(year, race_name)
            race_features = get_driver_features_from_race(race_session, race_name)
            all_race_data.append(race_features)
        except Exception as error:
            print(f"Skipping {race_name}: {error}")

    if not all_race_data:
        raise ValueError("No previous race data could be loaded.")

    return pd.concat(all_race_data, ignore_index=True)


# Creates overall season form for each driver using previous races
def summarize_driver_form(previous_race_data):
    return (
        previous_race_data.groupby("Driver")
        .agg(
            SeasonAvgLap_s=("AvgLapTime_s", "mean"),
            SeasonLapStd_s=("LapStd_s", "mean"),
            SeasonLapVar_s=("LapVar_s", "mean"),
            SeasonTireDeg_s=("TireDeg_s", "mean"),
            SeasonAvgStintPace_s=("AvgStintPace_s", "mean"),
            SeasonMeanSector1_s=("Mean_Sector1Time_s", "mean"),
            SeasonMeanSector2_s=("Mean_Sector2Time_s", "mean"),
            SeasonMeanSector3_s=("Mean_Sector3Time_s", "mean"),
            SeasonTotalLaps=("TotalLaps", "sum"),
            RacesUsed=("Race", "nunique")
        )
        .reset_index()
    )


# Picks the best qualifying time using Q3 first, then Q2, then Q1
def get_best_qualifying_time(row):
    for session_part in ["Q3", "Q2", "Q1"]:
        if session_part in row.index and pd.notna(row[session_part]):
            return row[session_part]
    return pd.NaT


# Fetches qualifying results and driver details from FastF1
def fetch_qualifying_results(year, race_name):
    quali_session = get_qualifying_session(year, race_name)
    results = quali_session.results.copy()

    useful_columns = [
        "Abbreviation",
        "FullName",
        "DriverNumber",
        "TeamName",
        "Position",
        "Q1",
        "Q2",
        "Q3"
    ]

    existing_columns = [col for col in useful_columns if col in results.columns]
    results = results[existing_columns].copy()

    results["BestQualiTime"] = results.apply(get_best_qualifying_time, axis=1)
    results = results.dropna(subset=["BestQualiTime"]).copy()
    results["QualifyingTime_s"] = results["BestQualiTime"].dt.total_seconds()

    results = results.rename(columns={"Abbreviation": "Driver"})
    results = results.sort_values("Position").reset_index(drop=True)

    return results


# Calculates team performance based on average driver pace
def build_team_form(previous_race_data, qualifying_data):
    merged = previous_race_data.merge(
        qualifying_data[["Driver", "TeamName"]],
        on="Driver",
        how="left"
    )

    team_form = (
        merged.groupby("TeamName")["AvgLapTime_s"]
        .mean()
        .reset_index()
        .rename(columns={"AvgLapTime_s": "TeamForm_s"})
    )

    return team_form


# Trains the machine learning model using selected features
def train_prediction_model(training_data, feature_columns):
    training_rows = training_data.dropna(subset=["SeasonAvgLap_s"]).copy()

    if len(training_rows) < 5:
        raise ValueError(
            "Not enough driver data to train the model. "
            "Make sure previous race data and qualifying data are available."
        )

    X = training_rows[feature_columns]
    y = training_rows["SeasonAvgLap_s"]

    imputer = SimpleImputer(strategy="median")
    X_filled = imputer.fit_transform(X)

    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )
    model.fit(X_filled, y)

    return model, imputer


# Predicts race pace and sorts drivers into finishing order
def predict_race_order(data, model, imputer, feature_columns):
    X_all = data[feature_columns]
    X_all_filled = imputer.transform(X_all)

    data = data.copy()
    data["PredictedRacePace_s"] = model.predict(X_all_filled)

    final_results = data.sort_values(
        by=["PredictedRacePace_s", "QualifyingTime_s"]
    ).reset_index(drop=True)

    return final_results


# Displays predicted results in one popup window
def show_results_window(final_results):
    root = tk.Tk()
    root.title(f"{YEAR} {TARGET_RACE} Prediction")
    root.geometry("920x720")
    root.configure(bg="#111111")

    style = ttk.Style()
    style.theme_use("default")

    style.configure(
        "Treeview",
        background="#1b1b1b",
        foreground="white",
        fieldbackground="#1b1b1b",
        rowheight=28,
        font=("Arial", 10)
    )
    style.configure(
        "Treeview.Heading",
        background="#d00000",
        foreground="white",
        font=("Arial", 10, "bold")
    )

    main_frame = tk.Frame(root, bg="#111111", padx=16, pady=16)
    main_frame.pack(fill="both", expand=True)

    title = tk.Label(
        main_frame,
        text=f"🏁 Predicted {YEAR} {TARGET_RACE} 🏁",
        font=("Arial", 20, "bold"),
        bg="#111111",
        fg="white"
    )
    title.pack(pady=(0, 16))

    podium_box = tk.LabelFrame(
        main_frame,
        text="🏆 Podium",
        font=("Arial", 13, "bold"),
        bg="#111111",
        fg="white",
        padx=12,
        pady=12
    )
    podium_box.pack(fill="x", pady=(0, 16))

    p1 = final_results.iloc[0]
    p2 = final_results.iloc[1]
    p3 = final_results.iloc[2]

    podium_lines = [
        f"🥇 P1: {p1['Driver']} - {p1['FullName']} ({p1['TeamName']})",
        f"🥈 P2: {p2['Driver']} - {p2['FullName']} ({p2['TeamName']})",
        f"🥉 P3: {p3['Driver']} - {p3['FullName']} ({p3['TeamName']})"
    ]

    for line in podium_lines:
        tk.Label(
            podium_box,
            text=line,
            font=("Arial", 12, "bold"),
            bg="#111111",
            fg="white",
            anchor="w"
        ).pack(fill="x", pady=3)

    top10_box = tk.LabelFrame(
        main_frame,
        text="Top 10 Finishers",
        font=("Arial", 13, "bold"),
        bg="#111111",
        fg="white",
        padx=10,
        pady=10
    )
    top10_box.pack(fill="both", pady=(0, 16))

    columns = ("Pos", "Driver", "Name", "Team", "Predicted Pace (s)")
    table = ttk.Treeview(top10_box, columns=columns, show="headings", height=10)

    column_widths = {
        "Pos": 60,
        "Driver": 80,
        "Name": 240,
        "Team": 200,
        "Predicted Pace (s)": 150
    }

    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=column_widths[col], anchor="center")

    top10 = final_results.head(10)

    for position, (_, row) in enumerate(top10.iterrows(), start=1):
        table.insert(
            "",
            "end",
            values=(
                position,
                row["Driver"],
                row["FullName"],
                row["TeamName"],
                f"{row['PredictedRacePace_s']:.3f}"
            )
        )

    table.pack(fill="x", expand=True)

    full_order_box = tk.LabelFrame(
        main_frame,
        text="Full Predicted Order",
        font=("Arial", 13, "bold"),
        bg="#111111",
        fg="white",
        padx=10,
        pady=10
    )
    full_order_box.pack(fill="both", expand=True)

    text_area = tk.Text(
        full_order_box,
        height=14,
        bg="#1b1b1b",
        fg="white",
        font=("Courier New", 10),
        relief="flat"
    )
    text_area.pack(fill="both", expand=True)

    for position, (_, row) in enumerate(final_results.iterrows(), start=1):
        line = (
            f"{position:>2}. "
            f"{row['Driver']:<3} | "
            f"{row['FullName']:<24} | "
            f"{row['TeamName']:<18} | "
            f"{row['PredictedRacePace_s']:.3f}s\n"
        )
        text_area.insert("end", line)

    text_area.config(state="disabled")
    root.mainloop()


# Runs the full pipeline: load data, train model, predict, and show results
def main():
    try:
        setup_fastf1_cache()

        print("Building previous race dataset...")
        previous_race_data = build_previous_race_dataset(YEAR, PREVIOUS_RACES)

        print("Summarizing driver form...")
        driver_form = summarize_driver_form(previous_race_data)

        print("Fetching qualifying results...")
        qualifying_data = fetch_qualifying_results(YEAR, TARGET_RACE)

        print("Building team form...")
        team_form = build_team_form(previous_race_data, qualifying_data)

        print("Merging datasets...")
        prediction_data = qualifying_data.merge(driver_form, on="Driver", how="left")
        prediction_data = prediction_data.merge(team_form, on="TeamName", how="left")
        prediction_data["DownforceScore"] = TRACK_DOWNFORCE.get(TARGET_RACE, 2)

        feature_columns = [
            "QualifyingTime_s",
            "TeamForm_s",
            "SeasonAvgLap_s",
            "SeasonLapStd_s",
            "SeasonLapVar_s",
            "SeasonTireDeg_s",
            "SeasonAvgStintPace_s",
            "SeasonMeanSector1_s",
            "SeasonMeanSector2_s",
            "SeasonMeanSector3_s",
            "DownforceScore",
            "SeasonTotalLaps",
            "RacesUsed"
        ]

        print("Training model...")
        model, imputer = train_prediction_model(prediction_data, feature_columns)

        print("Predicting race order...")
        final_results = predict_race_order(
            prediction_data,
            model,
            imputer,
            feature_columns
        )

        print("Opening results window...")
        show_results_window(final_results)

    except Exception as error:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Prediction Error", str(error))
        root.destroy()


if __name__ == "__main__":
    main()