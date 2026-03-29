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

# =========================================================
# 1) BASIC SETUP
# =========================================================
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

YEAR = 2026
TARGET_RACE = "Chinese Grand Prix"

# Use races before China in the same season.
# Adjust this list if your available 2026 calendar data differs.
PREVIOUS_RACES = [
    "Australian Grand Prix",
]

# Manual track-type feature
# low = 1, medium = 2, high = 3
TRACK_DOWNFORCE = {
    "Australian Grand Prix": 2,
    "Chinese Grand Prix": 2
}


# =========================================================
# 2) DATA HELPERS
# =========================================================
def load_race_session(year: int, race_name: str):
    session = fastf1.get_session(year, race_name, "R")
    session.load()
    return session


def load_qualifying_session(year: int, race_name: str):
    session = fastf1.get_session(year, race_name, "Q")
    session.load()
    return session


def prepare_laps(session) -> pd.DataFrame:
    laps = session.laps.copy()

    keep_cols = [
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
    available = [c for c in keep_cols if c in laps.columns]
    laps = laps[available].copy()

    # Need these core columns for the model
    needed = ["Driver", "LapTime", "Stint"]
    laps = laps.dropna(subset=[c for c in needed if c in laps.columns])

    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps = laps.dropna(subset=[col])
            laps[f"{col}_s"] = laps[col].dt.total_seconds()

    return laps


def compute_tire_degradation(driver_laps: pd.DataFrame) -> float:
    """
    Average tire degradation across stints:
    mean(last 3 laps of stint) - mean(first 3 laps of stint)
    Higher = more degradation
    """
    deg_values = []

    for _, stint_df in driver_laps.groupby("Stint"):
        stint_df = stint_df.sort_values("LapNumber")

        if len(stint_df) >= 6:
            first_avg = stint_df.head(3)["LapTime_s"].mean()
            last_avg = stint_df.tail(3)["LapTime_s"].mean()
            deg_values.append(last_avg - first_avg)

    if deg_values:
        return float(np.mean(deg_values))
    return np.nan


def compute_avg_stint_performance(driver_laps: pd.DataFrame) -> float:
    """
    Average of mean lap time from each stint
    """
    stint_means = []

    for _, stint_df in driver_laps.groupby("Stint"):
        if len(stint_df) >= 3:
            stint_means.append(stint_df["LapTime_s"].mean())

    if stint_means:
        return float(np.mean(stint_means))
    return np.nan


def build_driver_features_for_race(session, race_name: str) -> pd.DataFrame:
    laps = prepare_laps(session)

    rows = []

    for driver_code in laps["Driver"].dropna().unique():
        driver_laps = laps[laps["Driver"] == driver_code].copy()

        row = {
            "Driver": driver_code,
            "Race": race_name,
            "AvgLapTime_s": driver_laps["LapTime_s"].mean(),
            "LapStd_s": driver_laps["LapTime_s"].std(),
            "LapVar_s": driver_laps["LapTime_s"].var(),
            "TireDeg_s": compute_tire_degradation(driver_laps),
            "AvgStintPace_s": compute_avg_stint_performance(driver_laps),
            "TotalLaps": len(driver_laps),
            "DownforceScore": TRACK_DOWNFORCE.get(race_name, 2)
        }

        # Sector means if available
        for sector_col in ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]:
            if sector_col in driver_laps.columns:
                row[f"Mean_{sector_col}"] = driver_laps[sector_col].mean()
            else:
                row[f"Mean_{sector_col}"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    df["LapStd_s"] = df["LapStd_s"].fillna(0.0)
    df["LapVar_s"] = df["LapVar_s"].fillna(0.0)

    return df


def fetch_qualifying_results(year: int, race_name: str) -> pd.DataFrame:
    """
    Pull qualifying results + driver details directly from FastF1.
    Session.results contains columns like driver name, abbreviation,
    team name, driver number, Q1/Q2/Q3, etc.
    """
    quali = load_qualifying_session(year, race_name)
    results = quali.results.copy()

    cols_needed = [
        "Abbreviation",
        "FullName",
        "FirstName",
        "LastName",
        "BroadcastName",
        "DriverNumber",
        "CountryCode",
        "TeamName",
        "TeamColor",
        "Position",
        "Q1",
        "Q2",
        "Q3"
    ]
    existing = [c for c in cols_needed if c in results.columns]
    results = results[existing].copy()

    def best_quali_time(row):
        for col in ["Q3", "Q2", "Q1"]:
            if col in row.index and pd.notna(row[col]):
                return row[col]
        return pd.NaT

    results["BestQualiTime"] = results.apply(best_quali_time, axis=1)
    results = results.dropna(subset=["BestQualiTime"]).copy()
    results["QualifyingTime_s"] = results["BestQualiTime"].dt.total_seconds()

    results = results.rename(columns={"Abbreviation": "Driver"})
    results = results.sort_values("Position").reset_index(drop=True)

    return results


def build_previous_race_features(year: int, race_names: list[str]) -> pd.DataFrame:
    all_features = []

    for race in race_names:
        try:
            session = load_race_session(year, race)
            race_features = build_driver_features_for_race(session, race)
            all_features.append(race_features)
        except Exception as exc:
            print(f"Skipping {race}: {exc}")

    if not all_features:
        raise ValueError("No previous-race data could be loaded.")

    combined = pd.concat(all_features, ignore_index=True)
    return combined


def build_driver_form(previous_features: pd.DataFrame) -> pd.DataFrame:
    driver_form = (
        previous_features.groupby("Driver")
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

    return driver_form


def build_team_form(previous_features: pd.DataFrame, qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """
    Team form = average driver pace by team from previous races
    Lower = better
    """
    tmp = previous_features.merge(
        qualifying_df[["Driver", "TeamName"]],
        on="Driver",
        how="left"
    )

    team_form = (
        tmp.groupby("TeamName")["AvgLapTime_s"]
        .mean()
        .reset_index()
        .rename(columns={"AvgLapTime_s": "TeamForm_s"})
    )

    return team_form


# =========================================================
# 3) UI
# =========================================================
def show_results_popup(final_results: pd.DataFrame):
    root = tk.Tk()
    root.title("2026 Chinese Grand Prix Prediction")
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
    style.map("Treeview", background=[("selected", "#444444")])

    main = tk.Frame(root, bg="#111111", padx=16, pady=16)
    main.pack(fill="both", expand=True)

    title = tk.Label(
        main,
        text="🏁 Predicted 2026 Chinese Grand Prix 🏁",
        font=("Arial", 20, "bold"),
        bg="#111111",
        fg="white"
    )
    title.pack(pady=(0, 16))

    podium_frame = tk.LabelFrame(
        main,
        text="🏆 Podium",
        font=("Arial", 13, "bold"),
        bg="#111111",
        fg="white",
        padx=12,
        pady=12
    )
    podium_frame.pack(fill="x", pady=(0, 16))

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
            podium_frame,
            text=line,
            font=("Arial", 12, "bold"),
            bg="#111111",
            fg="white",
            anchor="w",
            justify="left"
        ).pack(fill="x", pady=3)

    top10_frame = tk.LabelFrame(
        main,
        text="Top 10 Finishers",
        font=("Arial", 13, "bold"),
        bg="#111111",
        fg="white",
        padx=10,
        pady=10
    )
    top10_frame.pack(fill="both", expand=False, pady=(0, 16))

    columns = ("Pos", "Driver", "Name", "Team", "Predicted Pace (s)")
    tree = ttk.Treeview(top10_frame, columns=columns, show="headings", height=10)

    widths = {
        "Pos": 60,
        "Driver": 80,
        "Name": 240,
        "Team": 200,
        "Predicted Pace (s)": 150
    }

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=widths[col], anchor="center")

    top10 = final_results.head(10).copy()

    for i, (_, row) in enumerate(top10.iterrows(), start=1):
        tree.insert(
            "",
            "end",
            values=(
                i,
                row["Driver"],
                row["FullName"],
                row["TeamName"],
                f"{row['PredictedRacePace_s']:.3f}"
            )
        )

    tree.pack(fill="x", expand=True)

    full_frame = tk.LabelFrame(
        main,
        text="Full Predicted Order",
        font=("Arial", 13, "bold"),
        bg="#111111",
        fg="white",
        padx=10,
        pady=10
    )
    full_frame.pack(fill="both", expand=True)

    text = tk.Text(
        full_frame,
        height=14,
        bg="#1b1b1b",
        fg="white",
        insertbackground="white",
        font=("Courier New", 10),
        relief="flat"
    )
    text.pack(fill="both", expand=True)

    for i, (_, row) in enumerate(final_results.iterrows(), start=1):
        line = (
            f"{i:>2}. "
            f"{row['Driver']:<3} | "
            f"{row['FullName']:<24} | "
            f"{row['TeamName']:<18} | "
            f"{row['PredictedRacePace_s']:.3f}s\n"
        )
        text.insert("end", line)

    text.config(state="disabled")

    root.mainloop()


# =========================================================
# 4) MAIN PIPELINE
# =========================================================
def main():
    try:
        # Build previous-race features
        previous_features = build_previous_race_features(YEAR, PREVIOUS_RACES)

        # Driver form from previous races
        driver_form = build_driver_form(previous_features)

        # Chinese GP qualifying pulled from API/data source
        china_quali = fetch_qualifying_results(YEAR, TARGET_RACE)

        # Team form from previous races
        team_form = build_team_form(previous_features, china_quali)

        # Merge
        prediction_df = china_quali.merge(driver_form, on="Driver", how="left")
        prediction_df = prediction_df.merge(team_form, on="TeamName", how="left")

        # Add target-race context
        prediction_df["DownforceScore"] = TRACK_DOWNFORCE.get(TARGET_RACE, 2)

        # Model features
        feature_cols = [
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

        # We use season avg lap pace as a simple pace proxy target.
        train_df = prediction_df.dropna(subset=["SeasonAvgLap_s"]).copy()

        if len(train_df) < 5:
            raise ValueError(
                "Not enough merged driver data to train the model. "
                "Check that the previous 2026 races and 2026 Chinese GP qualifying are available."
            )

        X = train_df[feature_cols]
        y = train_df["SeasonAvgLap_s"]

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        model = XGBRegressor(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        )
        model.fit(X_imputed, y)

        # Predict for all drivers in qualifying
        all_X = prediction_df[feature_cols]
        all_X_imputed = imputer.transform(all_X)

        prediction_df["PredictedRacePace_s"] = model.predict(all_X_imputed)

        final_results = prediction_df.sort_values(
            by=["PredictedRacePace_s", "QualifyingTime_s"]
        ).reset_index(drop=True)

        show_results_popup(final_results)

    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Prediction Error", str(exc))
        root.destroy()


if __name__ == "__main__":
    main()