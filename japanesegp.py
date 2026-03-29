import os
import warnings

import fastf1
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

YEAR = 2026
TARGET_RACE = "Japanese Grand Prix"

# Add all races before Japan that are available
PREVIOUS_RACES = [
    "Australian Grand Prix",
    "Chinese Grand Prix"
]

CACHE_FOLDER = "f1_cache"

# Track features taken from the table/image you shared
# Values are per-track and are used through interaction features
TRACK_PERFORMANCE = {
    "Australian Grand Prix": {
        "Energy_Lap_pct": 35.4,
        "FullThrottle_pct": 71.5,
        "FullThrottle_Recovery_Ratio": 2.02,
        "Recovery_Index": 10.66
    },
    "Chinese Grand Prix": {
        "Energy_Lap_pct": 66.3,
        "FullThrottle_pct": 57.4,
        "FullThrottle_Recovery_Ratio": 0.87,
        "Recovery_Index": 4.72
    },
    "Japanese Grand Prix": {
        "Energy_Lap_pct": 37.5,
        "FullThrottle_pct": 68.3,
        "FullThrottle_Recovery_Ratio": 1.82,
        "Recovery_Index": 10.58
    }
}


def setup_fastf1_cache():
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_FOLDER)


def get_session(year, race_name, session_type):
    print(f"Loading {session_type} session: {race_name}")
    session = fastf1.get_session(year, race_name, session_type)
    session.load()
    return session


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

    laps = laps.dropna(subset=["Driver", "LapTime", "Stint"]).copy()
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    for sector in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if sector in laps.columns:
            laps[f"{sector}_s"] = laps[sector].dt.total_seconds()

    return laps


def get_best_qualifying_time(row):
    for part in ["Q3", "Q2", "Q1"]:
        if part in row.index and pd.notna(row[part]):
            return row[part]
    return pd.NaT


def minmax_scale(series, reverse=False):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(0.5, index=series.index)

    min_val = valid.min()
    max_val = valid.max()

    if min_val == max_val:
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = (s - min_val) / (max_val - min_val)

    if reverse:
        scaled = 1 - scaled

    fill_value = scaled.dropna().mean() if not scaled.dropna().empty else 0.5
    return scaled.fillna(fill_value)


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


def calculate_average_stint_pace(driver_laps):
    stint_averages = []

    for _, stint_laps in driver_laps.groupby("Stint"):
        if len(stint_laps) >= 3:
            stint_averages.append(stint_laps["LapTime_s"].mean())

    if stint_averages:
        return float(np.mean(stint_averages))

    return np.nan


def extract_race_results(session, race_name):
    results = session.results.copy()

    useful_columns = [
        "Abbreviation",
        "FullName",
        "DriverNumber",
        "TeamName",
        "Position"
    ]

    existing_columns = [col for col in useful_columns if col in results.columns]
    results = results[existing_columns].copy()

    results = results.rename(columns={
        "Abbreviation": "Driver",
        "Position": "FinishPosition"
    })

    results["Race"] = race_name
    results["FinishPosition"] = pd.to_numeric(results["FinishPosition"], errors="coerce")

    return results


def get_driver_features_from_race(session, race_name):
    laps = clean_lap_data(session)
    race_results = extract_race_results(session, race_name)

    driver_rows = []

    for driver_code in laps["Driver"].dropna().unique():
        driver_laps = laps[laps["Driver"] == driver_code].copy()

        row = {
            "Driver": driver_code,
            "Race": race_name,
            "AvgLapTime_s": driver_laps["LapTime_s"].mean(),
            "LapStd_s": driver_laps["LapTime_s"].std(),
            "LapVar_s": driver_laps["LapTime_s"].var(),
            "TireDeg_s": calculate_tire_degradation(driver_laps),
            "AvgStintPace_s": calculate_average_stint_pace(driver_laps),
            "TotalLaps": len(driver_laps),
        }

        for sector_col in ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]:
            if sector_col in driver_laps.columns:
                row[f"Mean_{sector_col}"] = driver_laps[sector_col].mean()
            else:
                row[f"Mean_{sector_col}"] = np.nan

        driver_rows.append(row)

    features = pd.DataFrame(driver_rows)

    features["LapStd_s"] = features["LapStd_s"].fillna(0.0)
    features["LapVar_s"] = features["LapVar_s"].fillna(0.0)

    merged = features.merge(
        race_results[["Driver", "FullName", "DriverNumber", "TeamName", "FinishPosition"]],
        on="Driver",
        how="left"
    )

    return merged


def build_previous_race_dataset(year, race_list):
    all_race_data = []

    for race_name in race_list:
        try:
            race_session = get_session(year, race_name, "R")
            race_features = get_driver_features_from_race(race_session, race_name)
            all_race_data.append(race_features)
        except Exception as error:
            print(f"Skipping {race_name}: {error}")

    if not all_race_data:
        raise ValueError("No previous race data could be loaded.")

    return pd.concat(all_race_data, ignore_index=True)


def summarize_driver_form(previous_race_data):
    driver_form = (
        previous_race_data.groupby("Driver")
        .agg(
            FullName=("FullName", "last"),
            DriverNumber=("DriverNumber", "last"),
            TeamName=("TeamName", "last"),
            SeasonAvgLap_s=("AvgLapTime_s", "mean"),
            SeasonLapStd_s=("LapStd_s", "mean"),
            SeasonLapVar_s=("LapVar_s", "mean"),
            SeasonTireDeg_s=("TireDeg_s", "mean"),
            SeasonAvgStintPace_s=("AvgStintPace_s", "mean"),
            SeasonMeanSector1_s=("Mean_Sector1Time_s", "mean"),
            SeasonMeanSector2_s=("Mean_Sector2Time_s", "mean"),
            SeasonMeanSector3_s=("Mean_Sector3Time_s", "mean"),
            SeasonTotalLaps=("TotalLaps", "sum"),
            AvgFinishPosition=("FinishPosition", "mean"),
            BestFinishPosition=("FinishPosition", "min"),
            RacesUsed=("Race", "nunique")
        )
        .reset_index()
    )

    return driver_form


def build_team_strength(previous_race_data):
    team_strength = (
        previous_race_data.groupby("TeamName")
        .agg(
            TeamAvgLap_s=("AvgLapTime_s", "mean"),
            TeamAvgStintPace_s=("AvgStintPace_s", "mean"),
            TeamAvgFinishPos=("FinishPosition", "mean"),
            TeamBestFinishPos=("FinishPosition", "min"),
            TeamTotalLaps=("TotalLaps", "sum"),
            TeamRaceCount=("Race", "nunique")
        )
        .reset_index()
    )

    return team_strength


def fetch_qualifying_results(year, race_name):
    quali_session = get_session(year, race_name, "Q")
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
    results["QualifyingPosition"] = pd.to_numeric(results["Position"], errors="coerce")

    results = results.rename(columns={"Abbreviation": "Driver"})
    results = results.sort_values("QualifyingPosition").reset_index(drop=True)

    return results


def build_prediction_dataset(qualifying_data, driver_form, team_strength, target_race):
    prediction_data = qualifying_data.merge(
        driver_form,
        on="Driver",
        how="left",
        suffixes=("", "_driver_form")
    )

    prediction_data["TeamName"] = prediction_data["TeamName"].fillna(
        prediction_data["TeamName_driver_form"]
    )

    prediction_data["FullName"] = prediction_data["FullName"].fillna(
        prediction_data["FullName_driver_form"]
    )

    prediction_data["DriverNumber"] = prediction_data["DriverNumber"].fillna(
        prediction_data["DriverNumber_driver_form"]
    )

    prediction_data = prediction_data.merge(
        team_strength,
        on="TeamName",
        how="left"
    )

    track_info = TRACK_PERFORMANCE.get(target_race, {})
    prediction_data["Track_Energy_Lap"] = track_info.get("Energy_Lap_pct", np.nan)
    prediction_data["Track_FullThrottle"] = track_info.get("FullThrottle_pct", np.nan)
    prediction_data["Track_Recovery_Index"] = track_info.get("Recovery_Index", np.nan)
    prediction_data["Track_FT_Recovery_Ratio"] = track_info.get("FullThrottle_Recovery_Ratio", np.nan)

    return prediction_data


def add_component_scores(prediction_data):
    data = prediction_data.copy()

    # Base scores
    data["QualiTimeScore"] = minmax_scale(data["QualifyingTime_s"], reverse=True)
    data["QualiPosScore"] = minmax_scale(data["QualifyingPosition"], reverse=True)
    data["QualifyingScore"] = (
        0.65 * data["QualiTimeScore"] +
        0.35 * data["QualiPosScore"]
    )

    data["DriverPaceScore"] = minmax_scale(data["SeasonAvgLap_s"], reverse=True)
    data["DriverStintScore"] = minmax_scale(data["SeasonAvgStintPace_s"], reverse=True)
    data["DriverFinishScore"] = minmax_scale(data["AvgFinishPosition"], reverse=True)

    data["DriverFormScore"] = (
        0.45 * data["DriverPaceScore"] +
        0.35 * data["DriverStintScore"] +
        0.20 * data["DriverFinishScore"]
    )

    data["TeamPaceScore"] = minmax_scale(data["TeamAvgLap_s"], reverse=True)
    data["TeamStintScore"] = minmax_scale(data["TeamAvgStintPace_s"], reverse=True)
    data["TeamFinishScore"] = minmax_scale(data["TeamAvgFinishPos"], reverse=True)

    data["TeamStrengthScore"] = (
        0.45 * data["TeamPaceScore"] +
        0.25 * data["TeamStintScore"] +
        0.30 * data["TeamFinishScore"]
    )

    data["TireDegScore"] = minmax_scale(data["SeasonTireDeg_s"], reverse=True)
    data["ConsistencyScore"] = minmax_scale(data["SeasonLapStd_s"], reverse=True)
    data["VarianceScore"] = minmax_scale(data["SeasonLapVar_s"], reverse=True)

    data["RaceManagementScore"] = (
        0.45 * data["TireDegScore"] +
        0.30 * data["ConsistencyScore"] +
        0.25 * data["VarianceScore"]
    )

    data["ReliabilityScore"] = minmax_scale(data["SeasonTotalLaps"], reverse=True)

    # Sector-based track fit
    data["Sector1Score"] = minmax_scale(data["SeasonMeanSector1_s"], reverse=True)
    data["Sector2Score"] = minmax_scale(data["SeasonMeanSector2_s"], reverse=True)
    data["Sector3Score"] = minmax_scale(data["SeasonMeanSector3_s"], reverse=True)

    data["DriverTrackFitScore"] = (
        0.30 * data["Sector1Score"] +
        0.40 * data["Sector2Score"] +
        0.30 * data["Sector3Score"]
    )

    # Normalize track values globally using track dictionary
    all_energy = pd.Series([v["Energy_Lap_pct"] for v in TRACK_PERFORMANCE.values()])
    all_full_throttle = pd.Series([v["FullThrottle_pct"] for v in TRACK_PERFORMANCE.values()])
    all_ratio = pd.Series([v["FullThrottle_Recovery_Ratio"] for v in TRACK_PERFORMANCE.values()])
    all_recovery_index = pd.Series([v["Recovery_Index"] for v in TRACK_PERFORMANCE.values()])

    target_energy = data["Track_Energy_Lap"].iloc[0]
    target_full_throttle = data["Track_FullThrottle"].iloc[0]
    target_ratio = data["Track_FT_Recovery_Ratio"].iloc[0]
    target_recovery_index = data["Track_Recovery_Index"].iloc[0]

    energy_norm = (target_energy - all_energy.min()) / (all_energy.max() - all_energy.min()) if all_energy.max() != all_energy.min() else 0.5
    full_throttle_norm = (target_full_throttle - all_full_throttle.min()) / (all_full_throttle.max() - all_full_throttle.min()) if all_full_throttle.max() != all_full_throttle.min() else 0.5
    ratio_norm = (target_ratio - all_ratio.min()) / (all_ratio.max() - all_ratio.min()) if all_ratio.max() != all_ratio.min() else 0.5
    recovery_index_norm = (target_recovery_index - all_recovery_index.min()) / (all_recovery_index.max() - all_recovery_index.min()) if all_recovery_index.max() != all_recovery_index.min() else 0.5

    # Interaction features: this is where the track actually affects ranking
    data["EnergyDemandImpact"] = energy_norm * (
        0.55 * data["TeamStrengthScore"] +
        0.45 * data["DriverFormScore"]
    )

    data["ThrottleDemandImpact"] = full_throttle_norm * (
        0.70 * data["TeamPaceScore"] +
        0.30 * data["QualifyingScore"]
    )

    data["RecoveryDemandImpact"] = (1 - recovery_index_norm) * (
        0.60 * data["RaceManagementScore"] +
        0.40 * data["ReliabilityScore"]
    )

    data["BalanceDemandImpact"] = (1 - ratio_norm) * (
        0.55 * data["DriverTrackFitScore"] +
        0.45 * data["ConsistencyScore"]
    )

    data["TrackAdjustedScore"] = (
        0.30 * data["EnergyDemandImpact"] +
        0.30 * data["ThrottleDemandImpact"] +
        0.20 * data["RecoveryDemandImpact"] +
        0.20 * data["BalanceDemandImpact"]
    )

    # Final score
    data["FinalHybridScore"] = (
        0.32 * data["QualifyingScore"] +
        0.24 * data["TeamStrengthScore"] +
        0.19 * data["DriverFormScore"] +
        0.08 * data["RaceManagementScore"] +
        0.04 * data["ReliabilityScore"] +
        0.13 * data["TrackAdjustedScore"]
    )

    return data


def apply_realism_guardrails(scored_data):
    data = scored_data.copy()
    data["GuardrailPenalty"] = 0.0

    weak_team_mask = data["TeamStrengthScore"] < data["TeamStrengthScore"].median()
    poor_quali_mask = data["QualifyingPosition"] > 8
    weak_driver_mask = data["DriverFormScore"] < data["DriverFormScore"].median()

    data.loc[weak_team_mask & poor_quali_mask, "GuardrailPenalty"] += 0.035
    data.loc[weak_team_mask & weak_driver_mask, "GuardrailPenalty"] += 0.020

    strong_case_mask = (
        (data["QualifyingPosition"] <= 4) &
        (data["TeamStrengthScore"] >= data["TeamStrengthScore"].quantile(0.65)) &
        (data["DriverFormScore"] >= data["DriverFormScore"].quantile(0.65))
    )
    data.loc[strong_case_mask, "GuardrailPenalty"] -= 0.015

    data["AdjustedRaceScore"] = data["FinalHybridScore"] - data["GuardrailPenalty"]

    return data


def predict_race_order(prediction_data):
    scored = add_component_scores(prediction_data)
    scored = apply_realism_guardrails(scored)

    final_results = scored.sort_values(
        by=["AdjustedRaceScore", "QualifyingPosition", "QualifyingTime_s"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    final_results["PredictedPosition"] = range(1, len(final_results) + 1)
    return final_results


def print_results(final_results):
    print("\n" + "=" * 95)
    print(f"🏁 {YEAR} {TARGET_RACE} - TRACK-AWARE REALISTIC PREDICTION")
    print("=" * 95)

    print("\n🏆 PODIUM")
    for i in range(min(3, len(final_results))):
        row = final_results.iloc[i]
        medal = ["🥇", "🥈", "🥉"][i]
        print(
            f"{medal} P{i+1}: {row['Driver']} | {row['FullName']} | "
            f"{row['TeamName']} | Score: {row['AdjustedRaceScore']:.4f}"
        )

    print("\n🔟 TOP 10")
    print("-" * 95)
    for i in range(min(10, len(final_results))):
        row = final_results.iloc[i]
        qpos = int(row["QualifyingPosition"]) if pd.notna(row["QualifyingPosition"]) else "NA"
        print(
            f"{i+1:>2}. {row['Driver']:<3} | "
            f"{row['FullName']:<24} | "
            f"{row['TeamName']:<18} | "
            f"Q Pos: {qpos:>2} | "
            f"Score: {row['AdjustedRaceScore']:.4f}"
        )

    print("\n📋 FULL ORDER")
    print("-" * 95)
    for i, row in final_results.iterrows():
        print(
            f"{i+1:>2}. {row['Driver']:<3} | "
            f"{row['FullName']:<24} | "
            f"{row['TeamName']:<18} | "
            f"Score: {row['AdjustedRaceScore']:.4f}"
        )


def main():
    try:
        setup_fastf1_cache()

        print("Building previous race dataset...")
        previous_race_data = build_previous_race_dataset(YEAR, PREVIOUS_RACES)

        print("Summarizing driver form...")
        driver_form = summarize_driver_form(previous_race_data)

        print("Building team strength...")
        team_strength = build_team_strength(previous_race_data)

        print("Fetching qualifying results...")
        qualifying_data = fetch_qualifying_results(YEAR, TARGET_RACE)

        print("Building prediction dataset...")
        prediction_data = build_prediction_dataset(
            qualifying_data,
            driver_form,
            team_strength,
            TARGET_RACE
        )

        print("Predicting race order...")
        final_results = predict_race_order(prediction_data)

        print_results(final_results)

    except Exception as error:
        print("\nPrediction Error:")
        print(str(error))


if __name__ == "__main__":
    main()