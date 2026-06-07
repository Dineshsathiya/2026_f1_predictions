"""
F1 race prediction script.

Uses FastF1 session data, recent form, track profile values, standings,
and practice/qualifying signals to rank the target race grid.
"""

import os
import warnings
import tempfile
import webbrowser
from typing import Dict, List, Optional, Tuple

import fastf1
import numpy as np
import pandas as pd
import requests

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# settings

YEAR = 2026
TARGET_RACE = "Monaco Grand Prix"
CACHE_FOLDER = "f1_cache"

USE_JOLPICA = True
JOLPICA_TIMEOUT = 15

RECENCY_DECAY = 0.75
RECENCY_BY_GROUP = {
    "pace": 0.76,
    "finish": 0.68,
    "practice": 0.80,
    "reliability": 0.92,
    "standings": 0.88,
    "teammate": 0.78
}

RECENCY_BLEND = {
    "season": 0.55,
    "recent": 0.30,
    "track": 0.15
}

RECENCY_CAPS = {
    "time_s": 0.06,
    "position": 1.75,
    "ratio": 0.12
}

TRACK_PERFORMANCE = {
    "Australian Grand Prix": {
        "Energy_Lap_pct": 35.4,
        "FullThrottle_pct": 71.5,
        "FullThrottle_Recovery_Ratio": 2.02,
        "Recovery_Index": 10.66,
        "TrackPositionImportance": 0.62,
        "PitLossImpact": 0.55,
        "OvertakingDifficulty": 0.58
    },
    "Chinese Grand Prix": {
        "Energy_Lap_pct": 66.3,
        "FullThrottle_pct": 57.4,
        "FullThrottle_Recovery_Ratio": 0.87,
        "Recovery_Index": 4.72,
        "TrackPositionImportance": 0.55,
        "PitLossImpact": 0.50,
        "OvertakingDifficulty": 0.52
    },
    "Japanese Grand Prix": {
        "Energy_Lap_pct": 37.5,
        "FullThrottle_pct": 68.3,
        "FullThrottle_Recovery_Ratio": 1.82,
        "Recovery_Index": 10.58,
        "TrackPositionImportance": 0.70,
        "PitLossImpact": 0.62,
        "OvertakingDifficulty": 0.68
    },
    "Miami Grand Prix": {
        "Energy_Lap_pct": 45.0,
        "FullThrottle_pct": 59.0,
        "FullThrottle_Recovery_Ratio": 1.25,
        "Recovery_Index": 7.0,
        "TrackPositionImportance": 0.66,
        "PitLossImpact": 0.58,
        "OvertakingDifficulty": 0.62
    },
    "Canadian Grand Prix": {
        "Energy_Lap_pct": 63.0,
        "FullThrottle_pct": 66.0,
        "FullThrottle_Recovery_Ratio": 1.05,
        "Recovery_Index": 5.4,
        "TrackPositionImportance": 0.60,
        "PitLossImpact": 0.52,
        "OvertakingDifficulty": 0.50
    },
    "Monaco Grand Prix": {
        "Energy_Lap_pct": 28.0,
        "FullThrottle_pct": 45.0,
        "FullThrottle_Recovery_Ratio": 0.70,
        "Recovery_Index": 8.8,
        "TrackPositionImportance": 0.96,
        "PitLossImpact": 0.92,
        "OvertakingDifficulty": 0.97
    }
}

# helpers

def setup_fastf1_cache() -> None:
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_FOLDER)

def safe_total_seconds(value):
    if pd.isna(value):
        return np.nan
    try:
        return value.total_seconds()
    except Exception:
        return np.nan

def weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~np.isnan(values) & ~np.isnan(weights)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    if np.nansum(weights) == 0:
        return float(np.nanmean(values))

    return float(np.average(values, weights=weights))

def minmax_scale(series: pd.Series, reverse: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(0.5, index=series.index)

    lo = valid.min()
    hi = valid.max()

    if lo == hi:
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = (s - lo) / (hi - lo)

    if reverse:
        scaled = 1 - scaled

    fill_value = scaled.dropna().mean() if not scaled.dropna().empty else 0.5
    return scaled.fillna(fill_value)

def fill_feature_frame(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    X = df.copy()

    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan

    X = X[feature_columns].copy()

    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median_value = X[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        X[col] = X[col].fillna(median_value)

    return X

def get_track_vector(race_name: str) -> np.ndarray:
    track_info = TRACK_PERFORMANCE.get(race_name, {})
    keys = [
        "Energy_Lap_pct",
        "FullThrottle_pct",
        "FullThrottle_Recovery_Ratio",
        "Recovery_Index",
        "TrackPositionImportance",
        "PitLossImpact",
        "OvertakingDifficulty"
    ]
    values = [track_info.get(k, np.nan) for k in keys]
    return np.array(values, dtype=float)

def compute_track_similarity(source_race: str, target_race: str) -> float:
    src = get_track_vector(source_race)
    tgt = get_track_vector(target_race)

    if np.isnan(src).all() or np.isnan(tgt).all():
        return 0.5

    valid = ~np.isnan(src) & ~np.isnan(tgt)
    src = src[valid]
    tgt = tgt[valid]

    if len(src) == 0:
        return 0.5

    ranges = np.maximum(np.abs(tgt), 1.0)
    distance = np.mean(np.abs(src - tgt) / ranges)
    similarity = 1.0 - distance
    return float(np.clip(similarity, 0.05, 1.0))

def cap_against_baseline(blended_value: float, baseline_value: float, metric_type: str = "ratio") -> float:
    if pd.isna(blended_value):
        return baseline_value
    if pd.isna(baseline_value):
        return blended_value

    if metric_type == "time_s":
        cap = RECENCY_CAPS["time_s"] * max(abs(baseline_value), 1.0)
    elif metric_type == "position":
        cap = RECENCY_CAPS["position"]
    else:
        cap = RECENCY_CAPS["ratio"] * max(abs(baseline_value), 1.0)

    low = baseline_value - cap
    high = baseline_value + cap
    return float(np.clip(blended_value, low, high))

def controlled_weighted_metric(
    grp: pd.DataFrame,
    col: str,
    group_key: str,
    metric_type: str = "ratio",
    recent_window: int = 2
) -> Tuple[float, float]:
    if col not in grp.columns:
        return np.nan, 0.0

    data = grp[[col, "RaceOrder", "TrackSimilarityToTarget"]].copy()
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[col])

    if data.empty:
        return np.nan, 0.0

    latest_order = data["RaceOrder"].max()
    data["RacesAgo"] = latest_order - data["RaceOrder"]

    baseline = float(data[col].mean())

    decay = RECENCY_BY_GROUP.get(group_key, RECENCY_DECAY)
    data["RecentWeight"] = decay ** data["RacesAgo"]
    recent = weighted_mean(data[col], data["RecentWeight"])

    recent_slice = data.sort_values("RaceOrder").tail(recent_window).copy()
    if recent_slice.empty:
        track_recent = recent
    else:
        recent_slice["TrackWeight"] = recent_slice["RecentWeight"] * recent_slice["TrackSimilarityToTarget"].fillna(0.5)
        track_recent = weighted_mean(recent_slice[col], recent_slice["TrackWeight"])

    components = []
    weights = []
    if not pd.isna(baseline):
        components.append(baseline)
        weights.append(RECENCY_BLEND["season"])
    if not pd.isna(recent):
        components.append(recent)
        weights.append(RECENCY_BLEND["recent"])
    if not pd.isna(track_recent):
        components.append(track_recent)
        weights.append(RECENCY_BLEND["track"])

    if not components:
        return np.nan, 0.0

    blended = float(np.average(components, weights=weights))
    capped = cap_against_baseline(blended, baseline, metric_type=metric_type)
    impact = 0.0 if pd.isna(baseline) else float(capped - baseline)
    return capped, impact

def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)

def driver_last_name(full_name: str, fallback: str = "") -> str:
    name = safe_text(full_name).strip()
    if not name:
        return safe_text(fallback).upper()
    parts = name.split()
    return parts[-1].upper()

def team_color(team_name: str) -> str:
    colors = {
        "McLaren": "#ff8000",
        "Ferrari": "#dc0000",
        "Mercedes": "#00d2be",
        "Red Bull Racing": "#1e5bc6",
        "Racing Bulls": "#6692ff",
        "Aston Martin": "#229971",
        "Alpine": "#0090ff",
        "Williams": "#005aff",
        "Sauber": "#52e252",
        "Audi": "#9b0000",
        "Haas F1 Team": "#b6babd",
        "Haas": "#b6babd"
    }
    return colors.get(str(team_name), "#888888")

# schedule and sessions

def get_event_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year).copy()

    if "EventName" not in schedule.columns:
        raise ValueError("FastF1 event schedule does not contain 'EventName'.")

    if "RoundNumber" not in schedule.columns:
        schedule["RoundNumber"] = range(1, len(schedule) + 1)

    return schedule

def is_real_race_weekend(event_name: str) -> bool:
    if pd.isna(event_name):
        return False

    text = str(event_name).lower()
    excluded_terms = ["testing", "pre-season", "launch", "presentation"]

    return not any(term in text for term in excluded_terms)

def get_previous_races(year: int, target_race: str) -> Tuple[List[str], pd.DataFrame]:
    schedule = get_event_schedule(year)
    schedule = schedule[schedule["EventName"].apply(is_real_race_weekend)].copy()
    schedule = schedule.drop_duplicates(subset=["EventName"]).reset_index(drop=True)

    target_matches = schedule[schedule["EventName"].str.lower() == target_race.lower()]
    if target_matches.empty:
        raise ValueError(f"Target race '{target_race}' was not found in the filtered {year} schedule.")

    target_idx = target_matches.index[0]
    previous_rows = schedule.loc[: target_idx - 1].copy()

    if previous_rows.empty:
        raise ValueError(f"No previous real race weekends found before {target_race}.")

    return previous_rows["EventName"].tolist(), schedule

def build_round_map(schedule: pd.DataFrame) -> Dict[str, int]:
    return dict(zip(schedule["EventName"], schedule["RoundNumber"]))

def get_session(year: int, race_name: str, session_type: str):
    print(f"Loading {session_type} session: {race_name}")
    session = fastf1.get_session(year, race_name, session_type)
    session.load()
    return session

def try_get_session(year: int, race_name: str, session_types: List[str]):
    for stype in session_types:
        try:
            session = get_session(year, race_name, stype)
            return stype, session
        except Exception as error:
            print(f"Skipping {stype} for {race_name}: {error}")
    return None, None

def detect_weekend_type(year: int, race_name: str) -> str:
    label, _ = try_get_session(year, race_name, ["SQ", "S"])
    return "sprint" if label is not None else "normal"

# session data

def clean_lap_data(session) -> pd.DataFrame:
    laps = session.laps.copy()

    if laps.empty:
        return pd.DataFrame()

    keep_cols = [
        "Driver",
        "LapNumber",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Stint",
        "TyreLife",
        "Compound",
        "PitOutTime",
        "PitInTime",
        "IsAccurate"
    ]

    existing = [col for col in keep_cols if col in laps.columns]
    laps = laps[existing].copy()

    if "Driver" not in laps.columns or "LapTime" not in laps.columns:
        return pd.DataFrame()

    laps = laps.dropna(subset=["Driver", "LapTime"]).copy()
    laps["LapTime_s"] = laps["LapTime"].apply(safe_total_seconds)

    for sector in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        if sector in laps.columns:
            laps[f"{sector}_s"] = laps[sector].apply(safe_total_seconds)

    if "IsAccurate" in laps.columns:
        laps = laps[(laps["IsAccurate"].isna()) | (laps["IsAccurate"] == True)].copy()

    if "PitOutTime" in laps.columns:
        laps = laps[laps["PitOutTime"].isna()].copy()

    if "PitInTime" in laps.columns:
        laps = laps[laps["PitInTime"].isna()].copy()

    laps = laps.dropna(subset=["LapTime_s"]).copy()
    laps = laps[laps["LapTime_s"] > 0].copy()

    return laps

def calculate_tire_degradation(driver_laps: pd.DataFrame) -> float:
    if driver_laps.empty or "Stint" not in driver_laps.columns:
        return np.nan

    degradation_values = []

    for _, stint_laps in driver_laps.groupby("Stint"):
        stint_laps = stint_laps.sort_values("LapNumber")

        if len(stint_laps) >= 6:
            first_three = stint_laps.head(3)["LapTime_s"].mean()
            last_three = stint_laps.tail(3)["LapTime_s"].mean()
            degradation_values.append(last_three - first_three)

    return float(np.mean(degradation_values)) if degradation_values else np.nan

def calculate_average_stint_pace(driver_laps: pd.DataFrame) -> float:
    if driver_laps.empty or "Stint" not in driver_laps.columns:
        return np.nan

    stint_averages = []

    for _, stint_laps in driver_laps.groupby("Stint"):
        if len(stint_laps) >= 3:
            stint_averages.append(stint_laps["LapTime_s"].mean())

    return float(np.mean(stint_averages)) if stint_averages else np.nan

def extract_race_results(session, race_name: str) -> pd.DataFrame:
    results = session.results.copy()

    useful_columns = ["Abbreviation", "FullName", "DriverNumber", "TeamName", "Position"]
    existing = [col for col in useful_columns if col in results.columns]
    results = results[existing].copy()

    results = results.rename(columns={"Abbreviation": "Driver", "Position": "FinishPosition"})
    results["Race"] = race_name
    results["FinishPosition"] = pd.to_numeric(results["FinishPosition"], errors="coerce")

    return results

def extract_generic_session_results(session, pos_col_name: str) -> pd.DataFrame:
    results = session.results.copy()

    useful_columns = ["Abbreviation", "FullName", "DriverNumber", "TeamName", "Position"]
    existing = [col for col in useful_columns if col in results.columns]
    results = results[existing].copy()

    results = results.rename(columns={"Abbreviation": "Driver", "Position": pos_col_name})
    results[pos_col_name] = pd.to_numeric(results[pos_col_name], errors="coerce")

    return results

def get_best_qualifying_time(row) -> Optional[pd.Timedelta]:
    for part in ["Q3", "Q2", "Q1"]:
        if part in row.index and pd.notna(row[part]):
            return row[part]
    return pd.NaT

def fetch_qualifying_results(year: int, race_name: str) -> pd.DataFrame:
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

    existing = [col for col in useful_columns if col in results.columns]
    results = results[existing].copy()

    results["BestQualiTime"] = results.apply(get_best_qualifying_time, axis=1)
    results = results.dropna(subset=["BestQualiTime"]).copy()
    results["QualifyingTime_s"] = results["BestQualiTime"].apply(safe_total_seconds)
    results["QualifyingPosition"] = pd.to_numeric(results["Position"], errors="coerce")
    results["QualiGapToPole_s"] = results["QualifyingTime_s"] - results["QualifyingTime_s"].min()

    results = results.rename(columns={"Abbreviation": "Driver"})
    results = results.sort_values("QualifyingPosition").reset_index(drop=True)

    return results

def get_driver_features_from_race(session, race_name: str) -> pd.DataFrame:
    laps = clean_lap_data(session)
    race_results = extract_race_results(session, race_name)

    if laps.empty:
        return pd.DataFrame()

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
            "TotalLaps": len(driver_laps)
        }

        for sector_col in ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]:
            row[f"Mean_{sector_col}"] = (
                driver_laps[sector_col].mean() if sector_col in driver_laps.columns else np.nan
            )

        driver_rows.append(row)

    features = pd.DataFrame(driver_rows)
    features["LapStd_s"] = features["LapStd_s"].fillna(0.0)
    features["LapVar_s"] = features["LapVar_s"].fillna(0.0)

    merged = features.merge(
        race_results[["Driver", "FullName", "DriverNumber", "TeamName", "FinishPosition"]],
        on="Driver",
        how="left"
    )

    track_info = TRACK_PERFORMANCE.get(race_name, {})
    merged["Track_Energy_Lap"] = track_info.get("Energy_Lap_pct", np.nan)
    merged["Track_FullThrottle"] = track_info.get("FullThrottle_pct", np.nan)
    merged["Track_Recovery_Index"] = track_info.get("Recovery_Index", np.nan)
    merged["Track_FT_Recovery_Ratio"] = track_info.get("FullThrottle_Recovery_Ratio", np.nan)
    merged["TrackPositionImportance"] = track_info.get("TrackPositionImportance", np.nan)
    merged["PitLossImpact"] = track_info.get("PitLossImpact", np.nan)
    merged["OvertakingDifficulty"] = track_info.get("OvertakingDifficulty", np.nan)

    return merged

def get_practice_driver_metrics(session, mode: str) -> pd.DataFrame:
    laps = clean_lap_data(session)
    if laps.empty:
        return pd.DataFrame(columns=["Driver"])

    rows = []

    for driver_code in laps["Driver"].dropna().unique():
        dlaps = laps[laps["Driver"] == driver_code].copy().sort_values("LapNumber")

        if mode in {"FP1", "FP3", "SQ"}:
            quick = dlaps.nsmallest(min(10, len(dlaps)), "LapTime_s")
            avg_pace = quick["LapTime_s"].mean() if not quick.empty else np.nan
            rows.append({"Driver": driver_code, f"{mode}_AvgLap_s": avg_pace})

        elif mode == "FP2":
            long_run_paces = []
            long_run_degs = []

            if "Stint" in dlaps.columns:
                for _, stint_laps in dlaps.groupby("Stint"):
                    stint_laps = stint_laps.sort_values("LapNumber").copy()

                    if len(stint_laps) >= 5:
                        core = stint_laps.iloc[1:-1].copy() if len(stint_laps) >= 7 else stint_laps.copy()
                        long_run_paces.append(core["LapTime_s"].mean())

                        if len(stint_laps) >= 6:
                            first_three = stint_laps.head(3)["LapTime_s"].mean()
                            last_three = stint_laps.tail(3)["LapTime_s"].mean()
                            long_run_degs.append(last_three - first_three)

            rows.append({
                "Driver": driver_code,
                "FP2_LongRunAvg_s": float(np.mean(long_run_paces)) if long_run_paces else np.nan,
                "FP2_LongRunDeg_s": float(np.mean(long_run_degs)) if long_run_degs else np.nan
            })

        elif mode == "SPRINT":
            rows.append({
                "Driver": driver_code,
                "SprintAvgLap_s": dlaps["LapTime_s"].mean() if not dlaps.empty else np.nan,
                "SprintLapStd_s": dlaps["LapTime_s"].std() if not dlaps.empty else np.nan
            })

    return pd.DataFrame(rows)

def get_sprint_features_for_event(year: int, race_name: str) -> pd.DataFrame:
    merged = pd.DataFrame(columns=["Driver"])

    _, sq_session = try_get_session(year, race_name, ["SQ"])
    if sq_session is not None:
        sq_metrics = get_practice_driver_metrics(sq_session, "SQ")
        merged = sq_metrics if merged.empty else merged.merge(sq_metrics, on="Driver", how="outer")

        sq_results = extract_generic_session_results(sq_session, "SprintShootoutPos")
        merged = merged.merge(sq_results[["Driver", "SprintShootoutPos"]], on="Driver", how="left")

        if {"Q1", "Q2", "Q3"}.intersection(set(sq_session.results.columns)):
            temp = sq_session.results.copy()
            temp["BestSQTime"] = temp.apply(get_best_qualifying_time, axis=1)
            temp = temp.rename(columns={"Abbreviation": "Driver"})
            temp["SprintShootoutBestTime_s"] = temp["BestSQTime"].apply(safe_total_seconds)
            merged = merged.merge(temp[["Driver", "SprintShootoutBestTime_s"]], on="Driver", how="left")

    _, sprint_session = try_get_session(year, race_name, ["S"])
    if sprint_session is not None:
        sprint_metrics = get_practice_driver_metrics(sprint_session, "SPRINT")
        merged = sprint_metrics if merged.empty else merged.merge(sprint_metrics, on="Driver", how="outer")

        sprint_results = extract_generic_session_results(sprint_session, "SprintFinishPos")
        merged = merged.merge(sprint_results[["Driver", "SprintFinishPos"]], on="Driver", how="left")

    expected_cols = [
        "Driver",
        "SQ_AvgLap_s",
        "SprintShootoutPos",
        "SprintShootoutBestTime_s",
        "SprintAvgLap_s",
        "SprintLapStd_s",
        "SprintFinishPos"
    ]

    for col in expected_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged

def get_practice_features_for_event(year: int, race_name: str) -> pd.DataFrame:
    merged = None
    weekend_type = detect_weekend_type(year, race_name)

    _, session = try_get_session(year, race_name, ["FP1"])
    if session is not None:
        merged = get_practice_driver_metrics(session, "FP1")

    if weekend_type == "normal":
        _, session = try_get_session(year, race_name, ["FP2"])
        if session is not None:
            df = get_practice_driver_metrics(session, "FP2")
            merged = df if merged is None else merged.merge(df, on="Driver", how="outer")

        _, session = try_get_session(year, race_name, ["FP3"])
        if session is not None:
            df = get_practice_driver_metrics(session, "FP3")
            merged = df if merged is None else merged.merge(df, on="Driver", how="outer")

    sprint_features = get_sprint_features_for_event(year, race_name)
    if merged is None:
        merged = sprint_features
    else:
        merged = merged.merge(sprint_features, on="Driver", how="outer")

    if merged is None:
        merged = pd.DataFrame(columns=["Driver"])

    expected_cols = [
        "FP1_AvgLap_s",
        "FP2_LongRunAvg_s",
        "FP2_LongRunDeg_s",
        "FP3_AvgLap_s",
        "SQ_AvgLap_s",
        "SprintShootoutPos",
        "SprintShootoutBestTime_s",
        "SprintAvgLap_s",
        "SprintLapStd_s",
        "SprintFinishPos"
    ]

    for col in expected_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    if weekend_type == "sprint":
        merged["EffectiveRacePrep_s"] = merged["SprintAvgLap_s"].combine_first(merged["FP2_LongRunAvg_s"])
        merged["EffectiveRaceDeg_s"] = merged["SprintLapStd_s"].combine_first(merged["FP2_LongRunDeg_s"])
        merged["PracticeRaceScoreRaw"] = (
            0.30 * merged["FP1_AvgLap_s"] +
            0.30 * merged["SprintAvgLap_s"] +
            0.20 * merged["SQ_AvgLap_s"] +
            0.20 * merged["SprintShootoutBestTime_s"]
        )
    else:
        merged["EffectiveRacePrep_s"] = merged["FP2_LongRunAvg_s"]
        merged["EffectiveRaceDeg_s"] = merged["FP2_LongRunDeg_s"]
        merged["PracticeRaceScoreRaw"] = (
            0.20 * merged["FP1_AvgLap_s"] +
            0.50 * merged["FP2_LongRunAvg_s"] +
            0.30 * merged["FP3_AvgLap_s"]
        )

    merged["WeekendType"] = weekend_type

    return merged

# standings

def fetch_constructor_standings_before_round(year: int, round_number: int) -> pd.DataFrame:
    cols = ["TeamName", "ConstructorsPointsBefore", "ConstructorsPosBefore"]

    if not USE_JOLPICA or round_number <= 1:
        return pd.DataFrame(columns=cols)

    previous_round = round_number - 1
    url = f"https://api.jolpi.ca/ergast/f1/{year}/{previous_round}/constructorstandings.json"

    try:
        response = requests.get(url, timeout=JOLPICA_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        standings_lists = (
            payload.get("MRData", {})
            .get("StandingsTable", {})
            .get("StandingsLists", [])
        )

        if not standings_lists:
            return pd.DataFrame(columns=cols)

        rows = []
        for item in standings_lists[0].get("ConstructorStandings", []):
            constructor = item.get("Constructor", {})
            rows.append({
                "TeamName": constructor.get("name"),
                "ConstructorsPointsBefore": pd.to_numeric(item.get("points"), errors="coerce"),
                "ConstructorsPosBefore": pd.to_numeric(item.get("position"), errors="coerce")
            })

        df = pd.DataFrame(rows)
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan

        return df[cols]

    except Exception as error:
        print(f"Could not fetch constructor standings before round {round_number}: {error}")
        return pd.DataFrame(columns=cols)

def fetch_driver_standings_before_round(year: int, round_number: int) -> pd.DataFrame:
    cols = ["Driver", "DriverPointsBefore", "DriverPosBefore"]

    if not USE_JOLPICA or round_number <= 1:
        return pd.DataFrame(columns=cols)

    previous_round = round_number - 1
    url = f"https://api.jolpi.ca/ergast/f1/{year}/{previous_round}/driverstandings.json"

    try:
        response = requests.get(url, timeout=JOLPICA_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        standings_lists = (
            payload.get("MRData", {})
            .get("StandingsTable", {})
            .get("StandingsLists", [])
        )

        if not standings_lists:
            return pd.DataFrame(columns=cols)

        rows = []
        for item in standings_lists[0].get("DriverStandings", []):
            driver = item.get("Driver", {})
            rows.append({
                "Driver": driver.get("code"),
                "DriverPointsBefore": pd.to_numeric(item.get("points"), errors="coerce"),
                "DriverPosBefore": pd.to_numeric(item.get("position"), errors="coerce")
            })

        df = pd.DataFrame(rows)
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan

        return df[cols]

    except Exception as error:
        print(f"Could not fetch driver standings before round {round_number}: {error}")
        return pd.DataFrame(columns=cols)

# features

def add_teammate_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    def safe_team_mean(source_col: str, mean_col: str):
        if source_col in data.columns:
            data[mean_col] = data.groupby("TeamName")[source_col].transform("mean")
        else:
            data[mean_col] = np.nan

    safe_team_mean("QualifyingTime_s", "TeamMeanQualiTime_s")
    safe_team_mean("FP1_AvgLap_s", "TeamMeanFP1_s")
    safe_team_mean("FP2_LongRunAvg_s", "TeamMeanFP2LR_s")
    safe_team_mean("FP3_AvgLap_s", "TeamMeanFP3_s")
    safe_team_mean("SprintAvgLap_s", "TeamMeanSprintAvg_s")
    safe_team_mean("AvgFinishPosition", "TeamMeanAvgFinish")
    safe_team_mean("SeasonAvgLap_s", "TeamMeanSeasonLap_s")
    safe_team_mean("SprintFinishPos", "TeamMeanSprintFinishPos")

    data["QualiGapToTeammate_s"] = (
        data["QualifyingTime_s"] - data["TeamMeanQualiTime_s"]
        if "QualifyingTime_s" in data.columns else np.nan
    )
    data["FP1GapToTeammate_s"] = (
        data["FP1_AvgLap_s"] - data["TeamMeanFP1_s"]
        if "FP1_AvgLap_s" in data.columns else np.nan
    )
    data["FP2LongRunGapToTeammate_s"] = (
        data["FP2_LongRunAvg_s"] - data["TeamMeanFP2LR_s"]
        if "FP2_LongRunAvg_s" in data.columns else np.nan
    )
    data["FP3GapToTeammate_s"] = (
        data["FP3_AvgLap_s"] - data["TeamMeanFP3_s"]
        if "FP3_AvgLap_s" in data.columns else np.nan
    )
    data["SprintPaceVsTeammate"] = (
        data["SprintAvgLap_s"] - data["TeamMeanSprintAvg_s"]
        if "SprintAvgLap_s" in data.columns else np.nan
    )
    data["AvgFinishVsTeammate"] = (
        data["AvgFinishPosition"] - data["TeamMeanAvgFinish"]
        if "AvgFinishPosition" in data.columns else np.nan
    )
    data["SeasonLapVsTeammate"] = (
        data["SeasonAvgLap_s"] - data["TeamMeanSeasonLap_s"]
        if "SeasonAvgLap_s" in data.columns else np.nan
    )
    data["SprintFinishVsTeammate"] = (
        data["SprintFinishPos"] - data["TeamMeanSprintFinishPos"]
        if "SprintFinishPos" in data.columns else np.nan
    )

    return data

def add_rolling_last2_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    needed = [
        "Driver",
        "RaceOrder",
        "FinishPosition",
        "AvgLapTime_s",
        "TireDeg_s",
        "FP2_LongRunAvg_s",
        "QualiGapToTeammate_s",
        "SprintFinishPos"
    ]

    for col in needed:
        if col not in data.columns:
            data[col] = np.nan

    data = data.sort_values(["Driver", "RaceOrder"]).copy()

    data["Last2_AvgFinish"] = (
        data.groupby("Driver")["FinishPosition"]
        .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    )
    data["Last2_AvgLap_s"] = (
        data.groupby("Driver")["AvgLapTime_s"]
        .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    )
    data["Last2_QualiGapToTeammate_s"] = (
        data.groupby("Driver")["QualiGapToTeammate_s"]
        .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    )
    data["Last2_TireDeg_s"] = (
        data.groupby("Driver")["TireDeg_s"]
        .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    )
    data["Last2_FP2LongRunAvg_s"] = (
        data.groupby("Driver")["FP2_LongRunAvg_s"]
        .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    )
    data["Last2_SprintFinishPos"] = (
        data.groupby("Driver")["SprintFinishPos"]
        .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    )

    return data

def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["QualiDeltaVsTeam"] = (
        df["QualifyingTime_s"] -
        df.groupby("TeamName")["QualifyingTime_s"].transform("mean")
    )

    df["LapTimeVsTeam"] = (
        df["SeasonAvgLap_s"] -
        df.groupby("TeamName")["SeasonAvgLap_s"].transform("mean")
    )

    if "AvgFinishPosition" in df.columns:
        df["FinishVsTeam"] = (
            df["AvgFinishPosition"] -
            df.groupby("TeamName")["AvgFinishPosition"].transform("mean")
        )
    else:
        df["FinishVsTeam"] = np.nan

    df["Energy_x_TeamPace"] = df["Track_Energy_Lap"] * df["TeamAvgLap_s"]
    df["Throttle_x_Quali"] = df["Track_FullThrottle"] * df["QualifyingTime_s"]
    df["Recovery_x_TireDeg"] = df["Track_Recovery_Index"] * df["SeasonTireDeg_s"]
    df["Ratio_x_Consistency"] = df["Track_FT_Recovery_Ratio"] * df["SeasonLapStd_s"]
    df["Energy_x_StintPace"] = df["Track_Energy_Lap"] * df["SeasonAvgStintPace_s"]

    df["FP1_vs_FP3_Delta"] = df["FP3_AvgLap_s"] - df["FP1_AvgLap_s"]
    df["FP2_LongRun_vs_Quali"] = df["FP2_LongRunAvg_s"] - df["QualifyingTime_s"]
    df["FP2_Deg_x_TrackRecovery"] = df["FP2_LongRunDeg_s"] * df["Track_Recovery_Index"]

    df["TrackPositionScore"] = minmax_scale(df["QualifyingPosition"], reverse=True) * df["TrackPositionImportance"]
    df["OvertakePenaltyScore"] = minmax_scale(df["QualifyingPosition"], reverse=False) * df["OvertakingDifficulty"]
    df["PitStrategyRiskScore"] = (
        0.40 * minmax_scale(df["SeasonTireDeg_s"], reverse=False) +
        0.30 * minmax_scale(df["SeasonLapStd_s"], reverse=False) +
        0.30 * minmax_scale(df["QualifyingPosition"], reverse=False)
    ) * df["PitLossImpact"]

    weekend_type = df["WeekendType"].iloc[0] if "WeekendType" in df.columns and not df.empty else "normal"

    if weekend_type == "sprint":
        df["PracticeRaceScore"] = (
            0.28 * df["FP1_AvgLap_s"] +
            0.28 * df["SprintAvgLap_s"] +
            0.22 * df["SQ_AvgLap_s"] +
            0.22 * df["SprintShootoutGapToPole_s"]
        )
    else:
        df["PracticeRaceScore"] = (
            0.20 * df["FP1_AvgLap_s"] +
            0.50 * df["FP2_LongRunAvg_s"] +
            0.30 * df["FP3_AvgLap_s"]
        )

    return df

# dataset

def build_previous_race_dataset(
    year: int,
    race_list: List[str],
    round_map: Dict[str, int],
    target_race: str
) -> pd.DataFrame:
    all_race_data = []
    race_order_map = {race_name: idx + 1 for idx, race_name in enumerate(race_list)}

    for race_name in race_list:
        try:
            race_session = get_session(year, race_name, "R")
            race_features = get_driver_features_from_race(race_session, race_name)

            if race_features.empty:
                print(f"Skipping {race_name}: no race feature rows")
                continue

            try:
                quali_features = fetch_qualifying_results(year, race_name)[
                    ["Driver", "QualifyingTime_s", "QualifyingPosition", "QualiGapToPole_s"]
                ]
                race_features = race_features.merge(quali_features, on="Driver", how="left")
            except Exception as error:
                print(f"Could not fetch qualifying for {race_name}: {error}")

            practice_features = get_practice_features_for_event(year, race_name)
            race_features = race_features.merge(practice_features, on="Driver", how="left")

            round_number = int(round_map.get(race_name, race_order_map[race_name]))
            race_features["RoundNumber"] = round_number
            race_features["RaceOrder"] = race_order_map[race_name]
            race_features["SourceRaceName"] = race_name
            race_features["TrackSimilarityToTarget"] = compute_track_similarity(race_name, target_race)

            constructor_standings = fetch_constructor_standings_before_round(year, round_number)
            driver_standings = fetch_driver_standings_before_round(year, round_number)

            race_features = race_features.merge(constructor_standings, on="TeamName", how="left")
            race_features = race_features.merge(driver_standings, on="Driver", how="left")

            all_race_data.append(race_features)

        except Exception as error:
            print(f"Skipping {race_name}: {error}")

    if not all_race_data:
        raise ValueError("No previous race data could be loaded.")

    full_df = pd.concat(all_race_data, ignore_index=True)
    full_df = add_teammate_relative_features(full_df)
    full_df = add_rolling_last2_features(full_df)
    return full_df

def summarize_driver_form(previous_race_data: pd.DataFrame) -> pd.DataFrame:
    df = previous_race_data.copy()
    rows = []

    for driver, grp in df.groupby("Driver"):
        full_name_series = grp["FullName"].dropna()
        driver_num_series = grp["DriverNumber"].dropna()
        team_name_series = grp["TeamName"].dropna()

        season_avg_lap, lap_impact = controlled_weighted_metric(grp, "AvgLapTime_s", "pace", "time_s")
        season_stint_pace, stint_impact = controlled_weighted_metric(grp, "AvgStintPace_s", "pace", "time_s")
        avg_finish, finish_impact = controlled_weighted_metric(grp, "FinishPosition", "finish", "position")
        tire_deg, tire_impact = controlled_weighted_metric(grp, "TireDeg_s", "pace", "time_s")
        fp2_long_run, fp2_impact = controlled_weighted_metric(grp, "FP2_LongRunAvg_s", "practice", "time_s")
        driver_points, points_impact = controlled_weighted_metric(grp, "DriverPointsBefore", "standings", "position")

        rows.append({
            "Driver": driver,
            "FullName": full_name_series.iloc[-1] if not full_name_series.empty else np.nan,
            "DriverNumber": driver_num_series.iloc[-1] if not driver_num_series.empty else np.nan,
            "TeamName": team_name_series.iloc[-1] if not team_name_series.empty else np.nan,
            "SeasonAvgLap_s": season_avg_lap,
            "SeasonLapStd_s": controlled_weighted_metric(grp, "LapStd_s", "pace", "time_s")[0],
            "SeasonLapVar_s": controlled_weighted_metric(grp, "LapVar_s", "pace", "ratio")[0],
            "SeasonTireDeg_s": tire_deg,
            "SeasonAvgStintPace_s": season_stint_pace,
            "SeasonMeanSector1_s": controlled_weighted_metric(grp, "Mean_Sector1Time_s", "pace", "time_s")[0],
            "SeasonMeanSector2_s": controlled_weighted_metric(grp, "Mean_Sector2Time_s", "pace", "time_s")[0],
            "SeasonMeanSector3_s": controlled_weighted_metric(grp, "Mean_Sector3Time_s", "pace", "time_s")[0],
            "SeasonTotalLaps": controlled_weighted_metric(grp, "TotalLaps", "reliability", "position")[0],
            "AvgFinishPosition": avg_finish,
            "BestFinishPosition": np.nanmin(grp["FinishPosition"].values) if grp["FinishPosition"].notna().any() else np.nan,
            "RacesUsed": grp["Race"].nunique(),
            "FP1_AvgLap_s": controlled_weighted_metric(grp, "FP1_AvgLap_s", "practice", "time_s")[0],
            "FP2_LongRunAvg_s": fp2_long_run,
            "FP2_LongRunDeg_s": controlled_weighted_metric(grp, "FP2_LongRunDeg_s", "practice", "time_s")[0],
            "FP3_AvgLap_s": controlled_weighted_metric(grp, "FP3_AvgLap_s", "practice", "time_s")[0],
            "SQ_AvgLap_s": controlled_weighted_metric(grp, "SQ_AvgLap_s", "practice", "time_s")[0],
            "SprintAvgLap_s": controlled_weighted_metric(grp, "SprintAvgLap_s", "practice", "time_s")[0],
            "SprintLapStd_s": controlled_weighted_metric(grp, "SprintLapStd_s", "practice", "time_s")[0],
            "SprintFinishPos": controlled_weighted_metric(grp, "SprintFinishPos", "finish", "position")[0],
            "DriverPointsBefore": driver_points,
            "DriverPosBefore": controlled_weighted_metric(grp, "DriverPosBefore", "standings", "position")[0],
            "QualiGapToTeammate_s": controlled_weighted_metric(grp, "QualiGapToTeammate_s", "teammate", "time_s")[0],
            "FP1GapToTeammate_s": controlled_weighted_metric(grp, "FP1GapToTeammate_s", "teammate", "time_s")[0],
            "FP2LongRunGapToTeammate_s": controlled_weighted_metric(grp, "FP2LongRunGapToTeammate_s", "teammate", "time_s")[0],
            "FP3GapToTeammate_s": controlled_weighted_metric(grp, "FP3GapToTeammate_s", "teammate", "time_s")[0],
            "AvgFinishVsTeammate": controlled_weighted_metric(grp, "AvgFinishVsTeammate", "teammate", "position")[0],
            "SeasonLapVsTeammate": controlled_weighted_metric(grp, "SeasonLapVsTeammate", "teammate", "time_s")[0],
            "SprintFinishVsTeammate": controlled_weighted_metric(grp, "SprintFinishVsTeammate", "teammate", "position")[0],
            "SprintPaceVsTeammate": controlled_weighted_metric(grp, "SprintPaceVsTeammate", "teammate", "time_s")[0],
            "Last2_AvgFinish": grp["Last2_AvgFinish"].dropna().iloc[-1] if not grp["Last2_AvgFinish"].dropna().empty else np.nan,
            "Last2_AvgLap_s": grp["Last2_AvgLap_s"].dropna().iloc[-1] if not grp["Last2_AvgLap_s"].dropna().empty else np.nan,
            "Last2_QualiGapToTeammate_s": grp["Last2_QualiGapToTeammate_s"].dropna().iloc[-1] if not grp["Last2_QualiGapToTeammate_s"].dropna().empty else np.nan,
            "Last2_TireDeg_s": grp["Last2_TireDeg_s"].dropna().iloc[-1] if not grp["Last2_TireDeg_s"].dropna().empty else np.nan,
            "Last2_FP2LongRunAvg_s": grp["Last2_FP2LongRunAvg_s"].dropna().iloc[-1] if not grp["Last2_FP2LongRunAvg_s"].dropna().empty else np.nan,
            "Last2_SprintFinishPos": grp["Last2_SprintFinishPos"].dropna().iloc[-1] if not grp["Last2_SprintFinishPos"].dropna().empty else np.nan,
            "WeekendType": grp["WeekendType"].dropna().iloc[-1] if "WeekendType" in grp.columns and not grp["WeekendType"].dropna().empty else "normal",
            "TrackSimilarityAvg": grp["TrackSimilarityToTarget"].mean() if "TrackSimilarityToTarget" in grp.columns else np.nan,
            "RecencyImpact_LapTime_s": lap_impact,
            "RecencyImpact_FinishPos": finish_impact,
            "RecencyImpact_FP2LongRun_s": fp2_impact,
            "RecencyImpact_DriverPoints": points_impact,
            "RecencyImpact_TireDeg_s": tire_impact
        })

    return pd.DataFrame(rows)

def build_team_strength(previous_race_data: pd.DataFrame) -> pd.DataFrame:
    df = previous_race_data.copy()
    rows = []

    for team, grp in df.groupby("TeamName"):
        rows.append({
            "TeamName": team,
            "TeamAvgLap_s": controlled_weighted_metric(grp, "AvgLapTime_s", "pace", "time_s")[0],
            "TeamAvgStintPace_s": controlled_weighted_metric(grp, "AvgStintPace_s", "pace", "time_s")[0],
            "TeamAvgFinishPos": controlled_weighted_metric(grp, "FinishPosition", "finish", "position")[0],
            "TeamBestFinishPos": np.nanmin(grp["FinishPosition"].values) if grp["FinishPosition"].notna().any() else np.nan,
            "TeamTotalLaps": controlled_weighted_metric(grp, "TotalLaps", "reliability", "position")[0],
            "TeamRaceCount": grp["Race"].nunique(),
            "TeamFP2_LongRunAvg_s": controlled_weighted_metric(grp, "FP2_LongRunAvg_s", "practice", "time_s")[0],
            "TeamFP2_LongRunDeg_s": controlled_weighted_metric(grp, "FP2_LongRunDeg_s", "practice", "time_s")[0],
            "ConstructorsPointsBefore": controlled_weighted_metric(grp, "ConstructorsPointsBefore", "standings", "position")[0],
            "ConstructorsPosBefore": controlled_weighted_metric(grp, "ConstructorsPosBefore", "standings", "position")[0],
            "TeamTrackSimilarityAvg": grp["TrackSimilarityToTarget"].mean() if "TrackSimilarityToTarget" in grp.columns else np.nan
        })

    return pd.DataFrame(rows)

def prepare_training_data(previous_race_data: pd.DataFrame, team_strength: pd.DataFrame) -> pd.DataFrame:
    df = previous_race_data.copy()
    latest_race_order = df["RaceOrder"].max()
    df["RacesAgo"] = latest_race_order - df["RaceOrder"]
    df["RaceWeight"] = RECENCY_DECAY ** df["RacesAgo"]

    rows = []

    for (race, driver), grp in df.groupby(["Race", "Driver"]):
        weights = grp["RaceWeight"].values
        full_name_series = grp["FullName"].dropna()
        driver_num_series = grp["DriverNumber"].dropna()
        team_name_series = grp["TeamName"].dropna()

        rows.append({
            "Race": race,
            "Driver": driver,
            "QualifyingTime_s": weighted_mean(grp["QualifyingTime_s"], weights),
            "QualifyingPosition": weighted_mean(grp["QualifyingPosition"], weights),
            "QualiGapToPole_s": weighted_mean(grp["QualiGapToPole_s"], weights),
            "FullName": full_name_series.iloc[-1] if not full_name_series.empty else np.nan,
            "DriverNumber": driver_num_series.iloc[-1] if not driver_num_series.empty else np.nan,
            "TeamName": team_name_series.iloc[-1] if not team_name_series.empty else np.nan,
            "Track_Energy_Lap": grp["Track_Energy_Lap"].dropna().iloc[-1] if grp["Track_Energy_Lap"].notna().any() else np.nan,
            "Track_FullThrottle": grp["Track_FullThrottle"].dropna().iloc[-1] if grp["Track_FullThrottle"].notna().any() else np.nan,
            "Track_Recovery_Index": grp["Track_Recovery_Index"].dropna().iloc[-1] if grp["Track_Recovery_Index"].notna().any() else np.nan,
            "Track_FT_Recovery_Ratio": grp["Track_FT_Recovery_Ratio"].dropna().iloc[-1] if grp["Track_FT_Recovery_Ratio"].notna().any() else np.nan,
            "TrackPositionImportance": grp["TrackPositionImportance"].dropna().iloc[-1] if grp["TrackPositionImportance"].notna().any() else np.nan,
            "PitLossImpact": grp["PitLossImpact"].dropna().iloc[-1] if grp["PitLossImpact"].notna().any() else np.nan,
            "OvertakingDifficulty": grp["OvertakingDifficulty"].dropna().iloc[-1] if grp["OvertakingDifficulty"].notna().any() else np.nan,
            "SeasonAvgLap_s": weighted_mean(grp["AvgLapTime_s"], weights),
            "SeasonLapStd_s": weighted_mean(grp["LapStd_s"], weights),
            "SeasonLapVar_s": weighted_mean(grp["LapVar_s"], weights),
            "SeasonTireDeg_s": weighted_mean(grp["TireDeg_s"], weights),
            "SeasonAvgStintPace_s": weighted_mean(grp["AvgStintPace_s"], weights),
            "SeasonMeanSector1_s": weighted_mean(grp["Mean_Sector1Time_s"], weights),
            "SeasonMeanSector2_s": weighted_mean(grp["Mean_Sector2Time_s"], weights),
            "SeasonMeanSector3_s": weighted_mean(grp["Mean_Sector3Time_s"], weights),
            "SeasonTotalLaps": grp["TotalLaps"].sum(),
            "RacesUsed": grp["Race"].nunique(),
            "AvgFinishPosition": weighted_mean(grp["FinishPosition"], weights),
            "BestFinishPosition": np.nanmin(grp["FinishPosition"].values) if grp["FinishPosition"].notna().any() else np.nan,
            "TargetFinishPosition": weighted_mean(grp["FinishPosition"], weights),
            "FP1_AvgLap_s": weighted_mean(grp["FP1_AvgLap_s"], weights),
            "FP2_LongRunAvg_s": weighted_mean(grp["FP2_LongRunAvg_s"], weights),
            "FP2_LongRunDeg_s": weighted_mean(grp["FP2_LongRunDeg_s"], weights),
            "FP3_AvgLap_s": weighted_mean(grp["FP3_AvgLap_s"], weights),
            "SQ_AvgLap_s": weighted_mean(grp["SQ_AvgLap_s"], weights),
            "SprintAvgLap_s": weighted_mean(grp["SprintAvgLap_s"], weights),
            "SprintLapStd_s": weighted_mean(grp["SprintLapStd_s"], weights),
            "SprintFinishPos": weighted_mean(grp["SprintFinishPos"], weights),
            "SprintShootoutGapToPole_s": 0.0,
            "ConstructorsPointsBefore": weighted_mean(grp["ConstructorsPointsBefore"], weights),
            "ConstructorsPosBefore": weighted_mean(grp["ConstructorsPosBefore"], weights),
            "DriverPointsBefore": weighted_mean(grp["DriverPointsBefore"], weights),
            "DriverPosBefore": weighted_mean(grp["DriverPosBefore"], weights),
            "Last2_AvgFinish": grp["Last2_AvgFinish"].dropna().iloc[-1] if not grp["Last2_AvgFinish"].dropna().empty else np.nan,
            "Last2_AvgLap_s": grp["Last2_AvgLap_s"].dropna().iloc[-1] if not grp["Last2_AvgLap_s"].dropna().empty else np.nan,
            "Last2_QualiGapToTeammate_s": grp["Last2_QualiGapToTeammate_s"].dropna().iloc[-1] if not grp["Last2_QualiGapToTeammate_s"].dropna().empty else np.nan,
            "Last2_TireDeg_s": grp["Last2_TireDeg_s"].dropna().iloc[-1] if not grp["Last2_TireDeg_s"].dropna().empty else np.nan,
            "Last2_FP2LongRunAvg_s": grp["Last2_FP2LongRunAvg_s"].dropna().iloc[-1] if not grp["Last2_FP2LongRunAvg_s"].dropna().empty else np.nan,
            "Last2_SprintFinishPos": grp["Last2_SprintFinishPos"].dropna().iloc[-1] if not grp["Last2_SprintFinishPos"].dropna().empty else np.nan,
            "WeekendType": grp["WeekendType"].dropna().iloc[-1] if "WeekendType" in grp.columns and not grp["WeekendType"].dropna().empty else "normal"
        })

    train_df = pd.DataFrame(rows)
    train_df = add_teammate_relative_features(train_df)
    train_df = train_df.merge(team_strength, on="TeamName", how="left")
    train_df = add_engineered_features(train_df)

    return train_df

def build_prediction_dataset(
    qualifying_data: pd.DataFrame,
    driver_form: pd.DataFrame,
    team_strength: pd.DataFrame,
    practice_target: pd.DataFrame,
    target_round: int,
    year: int,
    target_race: str
) -> pd.DataFrame:
    prediction_data = qualifying_data.merge(
        driver_form,
        on="Driver",
        how="left",
        suffixes=("", "_driver_form")
    )

    prediction_data["TeamName"] = prediction_data["TeamName"].fillna(prediction_data["TeamName_driver_form"])
    prediction_data["FullName"] = prediction_data["FullName"].fillna(prediction_data["FullName_driver_form"])
    prediction_data["DriverNumber"] = prediction_data["DriverNumber"].fillna(prediction_data["DriverNumber_driver_form"])

    prediction_data = prediction_data.merge(team_strength, on="TeamName", how="left")
    prediction_data = prediction_data.merge(practice_target, on="Driver", how="left", suffixes=("", "_practice"))

    for col in [
        "FP1_AvgLap_s",
        "FP2_LongRunAvg_s",
        "FP2_LongRunDeg_s",
        "FP3_AvgLap_s",
        "SQ_AvgLap_s",
        "SprintShootoutPos",
        "SprintShootoutBestTime_s",
        "SprintAvgLap_s",
        "SprintLapStd_s",
        "SprintFinishPos",
        "EffectiveRacePrep_s",
        "EffectiveRaceDeg_s",
        "PracticeRaceScoreRaw",
        "WeekendType"
    ]:
        practice_col = f"{col}_practice"
        if practice_col in prediction_data.columns:
            if col in prediction_data.columns:
                prediction_data[col] = prediction_data[practice_col].combine_first(prediction_data[col])
            else:
                prediction_data[col] = prediction_data[practice_col]

    if "SprintShootoutBestTime_s" in prediction_data.columns and prediction_data["SprintShootoutBestTime_s"].notna().any():
        prediction_data["SprintShootoutGapToPole_s"] = (
            prediction_data["SprintShootoutBestTime_s"] -
            prediction_data["SprintShootoutBestTime_s"].min()
        )
    else:
        prediction_data["SprintShootoutGapToPole_s"] = np.nan

    constructor_standings = fetch_constructor_standings_before_round(year, target_round)
    driver_standings = fetch_driver_standings_before_round(year, target_round)

    prediction_data = prediction_data.merge(constructor_standings, on="TeamName", how="left", suffixes=("", "_latest"))
    prediction_data = prediction_data.merge(driver_standings, on="Driver", how="left", suffixes=("", "_latest"))

    for base_col in ["ConstructorsPointsBefore", "ConstructorsPosBefore", "DriverPointsBefore", "DriverPosBefore"]:
        latest_col = f"{base_col}_latest"
        if latest_col in prediction_data.columns:
            prediction_data[base_col] = prediction_data[latest_col].combine_first(prediction_data.get(base_col))
        elif base_col not in prediction_data.columns:
            prediction_data[base_col] = np.nan

    track_info = TRACK_PERFORMANCE.get(target_race, {})
    prediction_data["Track_Energy_Lap"] = track_info.get("Energy_Lap_pct", np.nan)
    prediction_data["Track_FullThrottle"] = track_info.get("FullThrottle_pct", np.nan)
    prediction_data["Track_Recovery_Index"] = track_info.get("Recovery_Index", np.nan)
    prediction_data["Track_FT_Recovery_Ratio"] = track_info.get("FullThrottle_Recovery_Ratio", np.nan)
    prediction_data["TrackPositionImportance"] = track_info.get("TrackPositionImportance", np.nan)
    prediction_data["PitLossImpact"] = track_info.get("PitLossImpact", np.nan)
    prediction_data["OvertakingDifficulty"] = track_info.get("OvertakingDifficulty", np.nan)

    prediction_data = add_teammate_relative_features(prediction_data)
    prediction_data = add_engineered_features(prediction_data)

    return prediction_data

# model

def get_feature_columns() -> List[str]:
    return [
        "QualifyingTime_s",
        "QualifyingPosition",
        "QualiGapToPole_s",
        "SeasonAvgLap_s",
        "SeasonLapStd_s",
        "SeasonLapVar_s",
        "SeasonTireDeg_s",
        "SeasonAvgStintPace_s",
        "SeasonMeanSector1_s",
        "SeasonMeanSector2_s",
        "SeasonMeanSector3_s",
        "SeasonTotalLaps",
        "RacesUsed",
        "AvgFinishPosition",
        "BestFinishPosition",
        "FP1_AvgLap_s",
        "FP2_LongRunAvg_s",
        "FP2_LongRunDeg_s",
        "FP3_AvgLap_s",
        "SQ_AvgLap_s",
        "SprintAvgLap_s",
        "SprintLapStd_s",
        "SprintFinishPos",
        "SprintShootoutGapToPole_s",
        "ConstructorsPointsBefore",
        "ConstructorsPosBefore",
        "DriverPointsBefore",
        "DriverPosBefore",
        "TeamAvgLap_s",
        "TeamAvgStintPace_s",
        "TeamAvgFinishPos",
        "TeamBestFinishPos",
        "TeamTotalLaps",
        "TeamFP2_LongRunAvg_s",
        "TeamFP2_LongRunDeg_s",
        "Track_Energy_Lap",
        "Track_FullThrottle",
        "Track_Recovery_Index",
        "Track_FT_Recovery_Ratio",
        "TrackPositionImportance",
        "PitLossImpact",
        "OvertakingDifficulty",
        "QualiDeltaVsTeam",
        "LapTimeVsTeam",
        "FinishVsTeam",
        "Energy_x_TeamPace",
        "Throttle_x_Quali",
        "Recovery_x_TireDeg",
        "Ratio_x_Consistency",
        "Energy_x_StintPace",
        "FP1_vs_FP3_Delta",
        "FP2_LongRun_vs_Quali",
        "FP2_Deg_x_TrackRecovery",
        "PracticeRaceScore",
        "TrackPositionScore",
        "OvertakePenaltyScore",
        "PitStrategyRiskScore",
        "QualiGapToTeammate_s",
        "FP1GapToTeammate_s",
        "FP2LongRunGapToTeammate_s",
        "FP3GapToTeammate_s",
        "AvgFinishVsTeammate",
        "SeasonLapVsTeammate",
        "SprintFinishVsTeammate",
        "SprintPaceVsTeammate",
        "Last2_AvgFinish",
        "Last2_AvgLap_s",
        "Last2_QualiGapToTeammate_s",
        "Last2_TireDeg_s",
        "Last2_FP2LongRunAvg_s",
        "Last2_SprintFinishPos"
    ]

def train_prediction_model(training_data: pd.DataFrame, feature_columns: List[str]):
    rows = training_data.dropna(subset=["TargetFinishPosition"]).copy()

    if len(rows) < 8:
        raise ValueError("Not enough rows to train the model. Add more completed previous races.")

    X = fill_feature_frame(rows, feature_columns)
    y = pd.to_numeric(rows["TargetFinishPosition"], errors="coerce")

    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) < 8:
        raise ValueError("Not enough valid training rows after filtering target values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    model = XGBRegressor(
        n_estimators=550,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train.to_numpy(), y_train.to_numpy())
    test_predictions = model.predict(X_test.to_numpy())

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    r2 = r2_score(y_test, test_predictions)

    raw_importances = getattr(model, "feature_importances_", np.zeros(len(feature_columns)))
    raw_importances = np.asarray(raw_importances).flatten()

    if len(raw_importances) < len(feature_columns):
        raw_importances = np.pad(raw_importances, (0, len(feature_columns) - len(raw_importances)), constant_values=0)
    elif len(raw_importances) > len(feature_columns):
        raw_importances = raw_importances[:len(feature_columns)]

    feature_importance_df = pd.DataFrame(
        list(zip(feature_columns, raw_importances)),
        columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    stats = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "total_rows": len(X),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "feature_importance_df": feature_importance_df
    }

    return model, stats

# scoring

def add_component_scores(prediction_data: pd.DataFrame) -> pd.DataFrame:
    data = prediction_data.copy()
    weekend_type = data["WeekendType"].iloc[0] if "WeekendType" in data.columns and not data.empty else "normal"

    data["QualifyingScore"] = (
        0.42 * minmax_scale(data["QualifyingTime_s"], reverse=True) +
        0.24 * minmax_scale(data["QualifyingPosition"], reverse=True) +
        0.20 * minmax_scale(data["QualiGapToPole_s"], reverse=True) +
        0.14 * minmax_scale(data["QualiGapToTeammate_s"], reverse=True)
    )

    data["DriverFormScore"] = (
        0.12 * minmax_scale(data["SeasonAvgLap_s"], reverse=True) +
        0.10 * minmax_scale(data["SeasonAvgStintPace_s"], reverse=True) +
        0.04 * minmax_scale(data["AvgFinishPosition"], reverse=True) +
        0.08 * minmax_scale(data["FP2_LongRunAvg_s"], reverse=True) +
        0.10 * minmax_scale(data["DriverPointsBefore"], reverse=True) +
        0.08 * minmax_scale(data["DriverPosBefore"], reverse=True) +
        0.12 * minmax_scale(data["Last2_AvgFinish"], reverse=True) +
        0.12 * minmax_scale(data["Last2_AvgLap_s"], reverse=True) +
        0.10 * minmax_scale(data["Last2_FP2LongRunAvg_s"], reverse=True) +
        0.08 * minmax_scale(data["SeasonLapVsTeammate"], reverse=True) +
        0.08 * minmax_scale(data["AvgFinishVsTeammate"], reverse=True) +
        0.08 * minmax_scale(data["Last2_QualiGapToTeammate_s"], reverse=True)
    )

    data["TeamStrengthScore"] = (
        0.32 * minmax_scale(data["TeamAvgLap_s"], reverse=True) +
        0.18 * minmax_scale(data["TeamAvgStintPace_s"], reverse=True) +
        0.15 * minmax_scale(data["TeamAvgFinishPos"], reverse=True) +
        0.23 * minmax_scale(data["ConstructorsPointsBefore"], reverse=True) +
        0.12 * minmax_scale(data["ConstructorsPosBefore"], reverse=True)
    )

    data["RaceManagementScore"] = (
        0.22 * minmax_scale(data["SeasonTireDeg_s"], reverse=True) +
        0.14 * minmax_scale(data["SeasonLapStd_s"], reverse=True) +
        0.10 * minmax_scale(data["SeasonLapVar_s"], reverse=True) +
        0.14 * minmax_scale(data["FP2_LongRunDeg_s"], reverse=True) +
        0.10 * minmax_scale(data["FP2LongRunGapToTeammate_s"], reverse=True) +
        0.15 * minmax_scale(data["Last2_TireDeg_s"], reverse=True) +
        0.15 * minmax_scale(data["Last2_QualiGapToTeammate_s"], reverse=True)
    )

    if weekend_type == "sprint":
        data["PracticeScore"] = (
            0.22 * minmax_scale(data["FP1_AvgLap_s"], reverse=True) +
            0.20 * minmax_scale(data["SprintAvgLap_s"], reverse=True) +
            0.18 * minmax_scale(data["SQ_AvgLap_s"], reverse=True) +
            0.15 * minmax_scale(data["SprintShootoutGapToPole_s"], reverse=True) +
            0.10 * minmax_scale(data["FP1GapToTeammate_s"], reverse=True) +
            0.15 * minmax_scale(data["Last2_SprintFinishPos"], reverse=True)
        )

        data["SprintScore"] = (
            0.34 * minmax_scale(data["SprintFinishPos"], reverse=True) +
            0.22 * minmax_scale(data["SprintAvgLap_s"], reverse=True) +
            0.16 * minmax_scale(data["SprintShootoutPos"], reverse=True) +
            0.14 * minmax_scale(data["SprintFinishVsTeammate"], reverse=True) +
            0.14 * minmax_scale(data["SprintShootoutGapToPole_s"], reverse=True)
        )
    else:
        data["PracticeScore"] = (
            0.18 * minmax_scale(data["FP1_AvgLap_s"], reverse=True) +
            0.44 * minmax_scale(data["FP2_LongRunAvg_s"], reverse=True) +
            0.28 * minmax_scale(data["FP3_AvgLap_s"], reverse=True) +
            0.10 * minmax_scale(data["FP1GapToTeammate_s"], reverse=True)
        )

        data["SprintScore"] = pd.Series(0.5, index=data.index)

    data["ReliabilityScore"] = minmax_scale(data["SeasonTotalLaps"], reverse=True)

    # Monaco-specific strategy model:
    data["TrackPositionComponent"] = minmax_scale(data["TrackPositionScore"], reverse=False)
    data["StrategyRiskComponent"] = minmax_scale(data["PitStrategyRiskScore"], reverse=True)
    data["OvertakePenaltyComponent"] = minmax_scale(data["OvertakePenaltyScore"], reverse=True)

    data["StrategyScore"] = (
        0.50 * data["TrackPositionComponent"] +
        0.30 * data["StrategyRiskComponent"] +
        0.20 * data["OvertakePenaltyComponent"]
    )

    data["TrackAdjustedScore"] = (
        0.20 * minmax_scale(data["Energy_x_TeamPace"], reverse=True) +
        0.18 * minmax_scale(data["Throttle_x_Quali"], reverse=True) +
        0.15 * minmax_scale(data["Recovery_x_TireDeg"], reverse=True) +
        0.12 * minmax_scale(data["Ratio_x_Consistency"], reverse=True) +
        0.15 * minmax_scale(data["FP2_Deg_x_TrackRecovery"], reverse=True) +
        0.20 * data["StrategyScore"]
    )

    return data

def apply_realism_guardrails(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["GuardrailPenalty"] = 0.0

    weak_team_mask = df["TeamStrengthScore"] < df["TeamStrengthScore"].median()
    poor_quali_mask = df["QualifyingPosition"] > 8
    weak_driver_mask = df["DriverFormScore"] < df["DriverFormScore"].median()

    monaco_multiplier = 1.5 if TARGET_RACE.lower() == "monaco grand prix" else 1.0

    df.loc[weak_team_mask & poor_quali_mask, "GuardrailPenalty"] += 0.040 * monaco_multiplier
    df.loc[weak_team_mask & weak_driver_mask, "GuardrailPenalty"] += 0.020
    df.loc[df["QualifyingPosition"] > 12, "GuardrailPenalty"] += 0.020 * monaco_multiplier

    strong_case_mask = (
        (df["QualifyingPosition"] <= 4) &
        (df["TeamStrengthScore"] >= df["TeamStrengthScore"].quantile(0.60)) &
        (df["DriverFormScore"] >= df["DriverFormScore"].quantile(0.60))
    )
    df.loc[strong_case_mask, "GuardrailPenalty"] -= 0.020 * monaco_multiplier

    return df

def predict_race_order(
    prediction_data: pd.DataFrame,
    model,
    feature_columns: List[str]
) -> pd.DataFrame:
    data = prediction_data.copy()
    data = add_component_scores(data)
    data = apply_realism_guardrails(data)

    X_pred = fill_feature_frame(data, feature_columns)
    data["PredictedFinishPosition_ML"] = model.predict(X_pred.to_numpy())

    data["MLScore"] = minmax_scale(data["PredictedFinishPosition_ML"], reverse=True)

    data["FinalRaceScore"] = (
        0.26 * data["MLScore"] +
        0.22 * data["QualifyingScore"] +
        0.12 * data["TeamStrengthScore"] +
        0.11 * data["DriverFormScore"] +
        0.09 * data["PracticeScore"] +
        0.06 * data["RaceManagementScore"] +
        0.06 * data["TrackAdjustedScore"] +
        0.05 * data["StrategyScore"] +
        0.03 * data["ReliabilityScore"]
    ) - data["GuardrailPenalty"]

    final_results = data.sort_values(
        by=["FinalRaceScore", "QualifyingPosition", "QualifyingTime_s"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    final_results["PredictedPosition"] = range(1, len(final_results) + 1)

    return final_results

# output

def print_ml_stats(stats: dict) -> None:
    print("\n" + "=" * 92)
    print("MODEL STATS")
    print("=" * 92)
    print(f"Total rows used: {stats['total_rows']}")
    print(f"Training rows : {stats['train_rows']}")
    print(f"Test rows     : {stats['test_rows']}")
    print(f"MAE           : {stats['mae']:.4f}")
    print(f"RMSE          : {stats['rmse']:.4f}")
    print(f"R²            : {stats['r2']:.4f}")

    print("\nTOP FEATURE IMPORTANCE")
    print("-" * 92)
    print(stats["feature_importance_df"].head(25).to_string(index=False))

def print_results(final_results: pd.DataFrame) -> None:
    print("\n" + "=" * 135)
    print(f" {YEAR} {TARGET_RACE} - MONACO GP PREDICTION")
    print("=" * 135)

    print("\n PODIUM")
    for i in range(min(3, len(final_results))):
        row = final_results.iloc[i]
        medal = ["🥇", "🥈", "🥉"][i]
        print(
            f"{medal} P{i + 1}: {row['Driver']} | {row['FullName']} | "
            f"{row['TeamName']} | Final Score: {row['FinalRaceScore']:.4f}"
        )

    print("\n TOP 10")
    print("-" * 135)

    for i in range(min(10, len(final_results))):
        row = final_results.iloc[i]
        qpos = int(row["QualifyingPosition"]) if pd.notna(row["QualifyingPosition"]) else "NA"
        dpos = int(row["DriverPosBefore"]) if pd.notna(row["DriverPosBefore"]) else "NA"
        qgap = f"{row['QualiGapToPole_s']:.3f}" if pd.notna(row["QualiGapToPole_s"]) else "NA"
        strat = f"{row['StrategyScore']:.3f}" if pd.notna(row["StrategyScore"]) else "NA"

        print(
            f"{i + 1:>2}. {row['Driver']:<3} | "
            f"{str(row['FullName']):<24} | "
            f"{str(row['TeamName']):<18} | "
            f"DrvPos: {dpos:>2} | "
            f"Q Pos: {qpos:>2} | "
            f"Q Gap: {qgap:<6} | "
            f"Strategy: {strat:<6} | "
            f"ML Pred: {row['PredictedFinishPosition_ML']:.2f} | "
            f"Score: {row['FinalRaceScore']:.4f}"
        )

    print("\n FULL ORDER")
    print("-" * 135)

    for i, row in final_results.iterrows():
        print(
            f"{i + 1:>2}. {row['Driver']:<3} | "
            f"{str(row['FullName']):<24} | "
            f"{str(row['TeamName']):<18} | "
            f"ML Pred: {row['PredictedFinishPosition_ML']:.2f} | "
            f"Score: {row['FinalRaceScore']:.4f}"
        )

def export_results_html(final_results: pd.DataFrame, output_file: str = None) -> str:
    display_df = final_results.copy().head(10)
    display_df["DriverNumber"] = display_df["DriverNumber"].fillna("")

    rows_html = []
    for _, row in display_df.iterrows():
        color = team_color(row.get("TeamName", ""))
        accent = color
        pos = int(row["PredictedPosition"])
        driver_code = safe_text(row.get("Driver", ""))[:3].upper()
        last_name = driver_last_name(row.get("FullName", ""), fallback=driver_code)
        team_name = safe_text(row.get("TeamName", ""))
        number = safe_text(row.get("DriverNumber", ""))
        score = f"{row['FinalRaceScore']:.3f}" if pd.notna(row.get("FinalRaceScore")) else ""

        rows_html.append(f"""
        <div class='rank-row'>
          <div class='pos-box'>{pos}</div>
          <div class='num-box'>{number}</div>
          <div class='bar' style='background:{accent};'>
            <div class='driver-chip'>{driver_code}</div>
            <div class='text-wrap'>
              <div class='driver-name'>{last_name}</div>
              <div class='team-name'>{team_name}</div>
            </div>
            <div class='score-pill'>{score}</div>
          </div>
        </div>
        """)

    title_words = TARGET_RACE.replace(" Grand Prix", "").upper()

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset='utf-8'>
      <title>{YEAR} {TARGET_RACE} Prediction</title>
      <style>
        * {{ box-sizing:border-box; }}
        body {{
          margin:0;
          background:#070707;
          color:#fff;
          font-family:Arial, Helvetica, sans-serif;
        }}
        .page {{
          min-height:100vh;
          display:flex;
          align-items:flex-start;
          justify-content:center;
          padding:18px;
          background:linear-gradient(180deg,#090909 0%,#121212 100%);
        }}
        .board {{
          width:100%;
          max-width:760px;
          min-height:1180px;
          background:#050505;
          border:2px solid #121212;
          display:grid;
          grid-template-columns:120px 1fr;
          overflow:hidden;
        }}
        .left-panel {{
          background:#060606;
          border-right:2px solid #111;
          position:relative;
          display:flex;
          flex-direction:column;
          align-items:center;
          justify-content:flex-start;
          padding-top:12px;
        }}
        .f1-logo {{
          font-size:24px;
          font-weight:900;
          letter-spacing:1px;
          margin-bottom:16px;
        }}
        .vertical-title {{
          writing-mode:vertical-rl;
          transform:rotate(180deg);
          font-size:76px;
          font-weight:900;
          line-height:0.9;
          letter-spacing:2px;
          margin-top:24px;
        }}
        .main-panel {{ padding:10px 14px 14px 14px; }}
        .header {{
          display:flex;
          justify-content:space-between;
          align-items:flex-start;
          padding:4px 2px 14px 2px;
        }}
        .round {{ font-size:14px; font-weight:800; color:#d0d0d0; letter-spacing:1px; }}
        .gp-name {{ font-size:30px; font-weight:900; line-height:0.95; margin-top:3px; }}
        .sub {{ color:#ff5a1f; font-size:13px; font-weight:700; margin-top:6px; letter-spacing:1px; }}
        .grid {{ display:flex; flex-direction:column; gap:10px; }}
        .rank-row {{ display:grid; grid-template-columns:46px 46px 1fr; gap:10px; align-items:center; }}
        .pos-box, .num-box {{
          height:50px;
          display:flex; align-items:center; justify-content:center;
          font-weight:900; font-size:24px;
          background:#0b0b0b; border-left:4px solid #222;
        }}
        .num-box {{ font-size:20px; color:#f0f0f0; }}
        .bar {{
          height:50px;
          display:grid;
          grid-template-columns:50px 1fr auto;
          align-items:center;
          padding:0 10px 0 0;
          color:#000;
          font-weight:900;
          text-transform:uppercase;
        }}
        .driver-chip {{
          width:42px; height:42px; border-radius:50%;
          background:rgba(0,0,0,0.24); color:#fff;
          display:flex; align-items:center; justify-content:center;
          margin-left:6px; font-size:13px; font-weight:900;
          border:2px solid rgba(255,255,255,0.35);
        }}
        .text-wrap {{ padding-left:10px; overflow:hidden; }}
        .driver-name {{ font-size:24px; letter-spacing:0.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        .team-name {{ font-size:11px; color:rgba(0,0,0,0.68); margin-top:2px; letter-spacing:0.8px; }}
        .score-pill {{
          min-width:62px; height:32px; border-radius:16px;
          padding:0 10px; background:rgba(255,255,255,0.22);
          display:flex; align-items:center; justify-content:center;
          font-size:13px; color:#fff; font-weight:800;
        }}
        .footer {{
          display:flex; justify-content:flex-end; padding-top:10px;
          color:#ff5a1f; font-size:12px; font-weight:800; letter-spacing:1px;
        }}
        @media (max-width:700px) {{
          .board {{ grid-template-columns:86px 1fr; min-height:auto; }}
          .vertical-title {{ font-size:54px; }}
          .gp-name {{ font-size:24px; }}
          .driver-name {{ font-size:18px; }}
          .rank-row {{ grid-template-columns:40px 40px 1fr; gap:8px; }}
          .pos-box, .num-box, .bar {{ height:44px; }}
        }}
      </style>
    </head>
    <body>
      <div class='page'>
        <div class='board'>
          <div class='left-panel'>
            <div class='f1-logo'>F1</div>
            <div class='vertical-title'>{title_words}</div>
          </div>
          <div class='main-panel'>
            <div class='header'>
              <div>
                <div class='round'>ROUND {len(display_df):02d} / {title_words}</div>
                <div class='gp-name'>THE GP</div>
                <div class='sub'>MODEL PREDICTION • {YEAR}</div>
              </div>
            </div>
            <div class='grid'>
              {''.join(rows_html)}
            </div>
            <div class='footer'>V1 MODEL</div>
          </div>
        </div>
      </div>
    </body>
    </html>
    """

    if output_file is None:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix="f1_prediction_", mode="w", encoding="utf-8")
        output_file = temp.name
        temp.write(html)
        temp.close()
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

    file_url = "file://" + os.path.abspath(output_file).replace("\\", "/")
    print(f"\nOpen preview: {file_url}")
    try:
        webbrowser.open(file_url)
    except Exception as e:
        print(f"Could not auto-open browser: {e}")
    return file_url

# run

def main():
    try:
        setup_fastf1_cache()

        print("Finding all races before target race...")
        previous_races, schedule = get_previous_races(YEAR, TARGET_RACE)
        round_map = build_round_map(schedule)
        target_round = int(round_map[TARGET_RACE])

        print(f"Previous races being used: {previous_races}")

        print("Building previous race dataset...")
        previous_race_data = build_previous_race_dataset(YEAR, previous_races, round_map, TARGET_RACE)

        print("Summarizing driver form with exponential recency weighting...")
        driver_form = summarize_driver_form(previous_race_data)

        print("Building team strength with exponential recency weighting...")
        team_strength = build_team_strength(previous_race_data)

        print("Preparing ML training data...")
        training_data = prepare_training_data(previous_race_data, team_strength)

        feature_columns = get_feature_columns()

        print("Training ML model...")
        model, stats = train_prediction_model(training_data, feature_columns)

        print("Fetching target qualifying results...")
        qualifying_data = fetch_qualifying_results(YEAR, TARGET_RACE)

        print("Fetching target practice/sprint features...")
        practice_target = get_practice_features_for_event(YEAR, TARGET_RACE)

        print("Building prediction dataset...")
        prediction_data = build_prediction_dataset(
            qualifying_data=qualifying_data,
            driver_form=driver_form,
            team_strength=team_strength,
            practice_target=practice_target,
            target_round=target_round,
            year=YEAR,
            target_race=TARGET_RACE
        )

        print("Predicting race order...")
        final_results = predict_race_order(
            prediction_data=prediction_data,
            model=model,
            feature_columns=feature_columns
        )

        print_ml_stats(stats)
        print_results(final_results)

        export_results_html(final_results)

        recency_cols = [
            "PredictedPosition", "Driver", "FullName", "TeamName",
            "RecencyImpact_LapTime_s", "RecencyImpact_FinishPos",
            "RecencyImpact_FP2LongRun_s", "TrackSimilarityAvg"
        ]
        existing_cols = [c for c in recency_cols if c in final_results.columns]
        if existing_cols:
            print("\nRECENCY IMPACT (top 10)")
            print("-" * 100)
            print(final_results[existing_cols].head(10).to_string(index=False))

    except Exception as error:
        print("\nPrediction Error:")
        print(str(error))

if __name__ == "__main__":
    main()
