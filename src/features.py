"""M3 — Feature engineering (FR-05).

Framing: one row = one segment and one TARGET window T. The model predicts T from information
available at the forecast origin t0 = T - h (h = horizon in windows, 1..4).

Leakage guards (checked by tests/test_features.py):
  * every traffic feature is taken at t0 or earlier (groupby('road_id').shift(k), k >= h);
  * rolling stats are computed on the already-shifted series ("shift before rolling");
  * lags are built from a CAUSAL fill of the raw observations (forward fill <= 2 windows,
    then the train-window median), never from the interpolated values, because interpolation
    at t0 would use the value at T;
  * only calendar and time-of-day features describe T itself (they are known in advance);
  * weather is the latest observed hour at t0, not the weather of T.
"""

import numpy as np
import pandas as pd

from src.clean import derive_congestion
from src.utils import assign_split, get_logger

KEY_COLUMNS = ["road_id", "timestamp", "origin_timestamp", "split", "horizon"]
TARGET_COLUMNS = ["y_volume", "y_travel_time", "y_congestion", "y_accident"]
MASK_COLUMNS = ["mask_volume", "mask_travel_time", "mask_congestion", "mask_accident"]
INFO_COLUMNS = ["road_capacity_half"]


def causal_series(clean, column, short_gap_max_windows):
    """Return a past-only filled version of `column` (one value per row of `clean`).

    Missing observations are forward-filled for at most `short_gap_max_windows` windows inside
    each segment; anything still missing takes the value written by clean.py, which at those
    positions is the segment x time-window median from the training dates.
    """
    observed = clean[f"{column}_obs"] if f"{column}_obs" in clean.columns else clean[column]
    forward_filled = observed.groupby(clean["road_id"]).ffill(limit=short_gap_max_windows)
    still_missing = forward_filled.isna()
    forward_filled[still_missing] = clean.loc[still_missing, column]
    return forward_filled


def congestion_codes(labels, order):
    """Convert congestion labels to integers 0..3 (Free-flow..Severe); unknown -> NaN."""
    code_lookup = {}
    for code, label in enumerate(order):
        code_lookup[label] = code
    return pd.Series(labels).map(code_lookup).astype(float).to_numpy()


def _peak_flag(time_strings, peak_windows):
    """1 if the HH:MM string falls inside any configured peak period."""
    is_peak = np.zeros(len(time_strings), dtype=int)
    for start, end in peak_windows:
        inside = (time_strings >= start) & (time_strings < end)
        is_peak = is_peak | inside.to_numpy().astype(int)
    return is_peak


def build_features(clean, config, horizon=1):
    """Build the modelling table for one forecast horizon.

    Args:
        clean: output of clean.run_cleaning (full 181,200-row grid).
        config: loaded configuration.
        horizon: windows ahead (1 = next 30-min window).

    Returns:
        DataFrame with KEY, feature, target and mask columns, one row per (segment, T).
    """
    logger = get_logger()
    feature_config = config["features"]
    short_gap = config["imputation"]["short_gap_max_windows"]
    order = config["congestion_order"]

    clean = clean.sort_values(["road_id", "timestamp"]).reset_index(drop=True)
    by_segment = clean.groupby("road_id")
    table = pd.DataFrame({"road_id": clean["road_id"], "timestamp": clean["timestamp"]})
    table["horizon"] = horizon
    table["origin_timestamp"] = by_segment["timestamp"].shift(horizon)

    # ---- causal (past-only) versions of every time-varying measurement
    causal = {}
    for column in ["traffic_volume", "avg_speed", "occupancy", "travel_time", "signal_timing",
                   "share_2w", "share_car", "share_lcv", "share_hcv"]:
        causal[column] = causal_series(clean, column, short_gap)
    causal_frame = pd.DataFrame(causal)
    causal_frame["road_id"] = clean["road_id"]
    causal_by_segment = causal_frame.groupby("road_id")

    def lag(column, windows_back):
        """Value of a causal series `windows_back` windows before T (within the segment)."""
        return causal_by_segment[column].shift(windows_back)

    # ---- volume lags relative to the origin t0 = T - h
    table["vol_t0"] = lag("traffic_volume", horizon)
    table["vol_t1"] = lag("traffic_volume", horizon + 1)
    table["vol_t2"] = lag("traffic_volume", horizon + 2)
    table["vol_same_window_yday"] = lag("traffic_volume", 48)
    table["vol_same_window_lastweek"] = lag("traffic_volume", 336)
    table["vol_diff_t0_t1"] = table["vol_t0"] - table["vol_t1"]

    # ---- rolling stats: shift FIRST (to t0), then roll over the previous k windows
    shifted_volume = lag("traffic_volume", horizon)
    shifted_speed = lag("avg_speed", horizon)
    for window_size in feature_config["rolling_windows"]:
        table[f"vol_roll_mean_{window_size}"] = shifted_volume.groupby(clean["road_id"]).transform(
            lambda values: values.rolling(window_size, min_periods=1).mean())
        table[f"vol_roll_std_{window_size}"] = shifted_volume.groupby(clean["road_id"]).transform(
            lambda values: values.rolling(window_size, min_periods=2).std())
        table[f"speed_roll_mean_{window_size}"] = shifted_speed.groupby(clean["road_id"]).transform(
            lambda values: values.rolling(window_size, min_periods=1).mean())

    # ---- other measurements at t0
    table["speed_t0"] = lag("avg_speed", horizon)
    table["speed_t1"] = lag("avg_speed", horizon + 1)
    table["occupancy_t0"] = lag("occupancy", horizon)
    table["travel_time_t0"] = lag("travel_time", horizon)
    table["travel_time_t1"] = lag("travel_time", horizon + 1)
    table["signal_timing_t0"] = lag("signal_timing", horizon)
    for share in ["share_2w", "share_car", "share_lcv", "share_hcv"]:
        table[f"{share}_t0"] = lag(share, horizon)

    # ---- capacity features (V/C uses the t0 volume, never the target window's)
    table["road_capacity"] = clean["road_capacity"]
    table["road_capacity_half"] = clean["road_capacity"] / 2.0
    table["vc_t0"] = table["vol_t0"] / table["road_capacity_half"]
    table["headroom_t0"] = table["road_capacity_half"] - table["vol_t0"]
    table["congestion_t0"] = congestion_codes(derive_congestion(table["vol_t0"], clean["road_capacity"], config), order)

    # ---- accident history (past week / past 30 days up to t0; absent windows count as 0)
    accidents_known = clean["accident_count"].fillna(0)
    shifted_accidents = accidents_known.groupby(clean["road_id"]).shift(horizon)
    history_windows = feature_config["accident_history_windows"]
    table["accidents_past_week"] = shifted_accidents.groupby(clean["road_id"]).transform(
        lambda values: values.rolling(history_windows, min_periods=1).sum())
    table["accidents_past_30d"] = shifted_accidents.groupby(clean["road_id"]).transform(
        lambda values: values.rolling(48 * 30, min_periods=1).sum())

    # ---- time encodings of the TARGET window (known in advance)
    local_time = clean["timestamp"].dt.tz_localize(None)
    window_of_day = local_time.dt.hour * 2 + local_time.dt.minute // 30
    day_of_week = local_time.dt.day_of_week
    table["tod_sin"] = np.sin(2 * np.pi * window_of_day / 48)
    table["tod_cos"] = np.cos(2 * np.pi * window_of_day / 48)
    table["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    table["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    table["is_weekend"] = (day_of_week >= 5).astype(int)
    table["is_peak"] = _peak_flag(local_time.dt.strftime("%H:%M"), feature_config["peak_windows"])

    # ---- calendar of the TARGET date (published in advance)
    table["public_holiday"] = clean["public_holiday"]
    table["event_flag"] = clean["event_flag"]
    table["roadwork_flag"] = clean["roadwork_flag"]
    table["holiday_x_peak"] = table["public_holiday"] * table["is_peak"]
    table["event_x_peak"] = table["event_flag"] * table["is_peak"]
    event_dates = set(clean.loc[clean["event_flag"] == 1, "date"])
    near_event = np.zeros(len(clean), dtype=int)
    for offset_days in [-1, 1]:
        shifted_dates = clean["date"] + pd.Timedelta(days=offset_days)
        near_event = near_event | shifted_dates.isin(event_dates).to_numpy().astype(int)
    table["event_within_1day"] = near_event

    # ---- weather: latest observation at t0 (the target window's weather is not known yet)
    table["temperature_t0"] = by_segment["temperature"].shift(horizon)
    table["rainfall_t0"] = by_segment["rainfall"].shift(horizon)
    table["visibility_t0"] = by_segment["visibility"].shift(horizon)
    condition_t0 = by_segment["weather_condition"].shift(horizon)
    table["rain_flag_t0"] = ((table["rainfall_t0"] >= feature_config["rain_mm"]) | (condition_t0 == "Rain")).astype(int)
    table["low_visibility_t0"] = (table["visibility_t0"] < feature_config["low_visibility_m"]).astype(int)
    for condition in ["Cloudy", "Rain", "Fog"]:                      # Clear is the reference level
        table[f"weather_{condition.lower()}_t0"] = (condition_t0 == condition).astype(int)

    # ---- segment identity (one-hot) and location
    table["latitude"] = clean["latitude"]
    table["longitude"] = clean["longitude"]
    for segment_id in sorted(clean["road_id"].unique()):
        table[f"seg_{segment_id}"] = (clean["road_id"] == segment_id).astype(int)

    # ---- targets at T and masks (evaluate only on originally observed targets)
    table["y_volume"] = clean["traffic_volume"]
    table["y_travel_time"] = clean["travel_time"]
    table["y_congestion"] = congestion_codes(clean["congestion_level"], order)
    table["y_accident"] = (clean["accident_count"] > 0).astype(float)
    table["mask_volume"] = clean["volume_target_observed"]
    table["mask_travel_time"] = clean["row_observed"] & ~clean["travel_time_imputed"]
    table["mask_congestion"] = clean["row_observed"] & clean["congestion_source"].isin(
        ["original", "derived_observed_volume"])
    table["mask_accident"] = clean["accident_observed"]

    table["split"] = assign_split(table["timestamp"], config)

    # Drop the first week + h windows of every segment: the same-window-last-week lag is undefined there.
    has_history = table["vol_same_window_lastweek"].notna() & table["origin_timestamp"].notna()
    dropped_rows = int((~has_history).sum())
    table = table.loc[has_history].reset_index(drop=True)
    table["vol_roll_std_4"] = table["vol_roll_std_4"].fillna(0.0)
    table["vol_roll_std_8"] = table["vol_roll_std_8"].fillna(0.0)
    logger.info(f"Features h={horizon}: {table.shape}, dropped {dropped_rows:,} warm-up rows (first 7 days)")
    return table


def get_feature_columns(table):
    """Return the model input columns (everything that is not a key, target, mask or info column)."""
    excluded = set(KEY_COLUMNS + TARGET_COLUMNS + MASK_COLUMNS + INFO_COLUMNS)
    feature_columns = []
    for column in table.columns:
        if column not in excluded:
            feature_columns.append(column)
    return feature_columns


def check_no_nulls(table, feature_columns):
    """Assert there are no nulls in model inputs (PRD §9.1 final validation stage)."""
    null_counts = table[feature_columns].isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if len(columns_with_nulls) > 0:
        raise ValueError(f"Nulls in feature columns: {columns_with_nulls.to_dict()}")


def build_all_horizons(clean, config):
    """Build and save one feature table per horizon in config['horizons']."""
    tables = {}
    for horizon in config["horizons"]:
        table = build_features(clean, config, horizon)
        check_no_nulls(table, get_feature_columns(table))
        table.to_parquet(config["paths"]["processed_dir"] / f"features_h{horizon}.parquet", index=False)
        tables[horizon] = table
    return tables
