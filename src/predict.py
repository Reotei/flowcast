"""Inference with persisted models (FR-10, FR-11, FR-12) — no retraining.

predict_next_windows() takes a cleaned table, appends the next 4 empty windows per segment,
builds the same leakage-safe features and returns, per segment, the forecast for t+1 .. t+4.
"""

import joblib
import numpy as np
import pandas as pd

from src.clean import derive_congestion
from src.evaluate import apply_residual_quantiles, prior_correct
from src.features import build_features

MEASUREMENT_COLUMNS = ["traffic_volume", "vehicle_count", "avg_speed", "occupancy", "travel_time",
                       "accident_count", "signal_timing", "share_2w", "share_car", "share_lcv", "share_hcv",
                       "temperature", "rainfall", "visibility"]


def extend_with_future(clean, config, n_future=4, calendar=None):
    """Append n_future empty windows per segment after the last timestamp in `clean`."""
    clean = clean.sort_values(["road_id", "timestamp"]).reset_index(drop=True)
    last_timestamp = clean["timestamp"].max()
    future_times = [last_timestamp + pd.Timedelta(minutes=30 * step) for step in range(1, n_future + 1)]
    last_rows = clean.loc[clean["timestamp"] == last_timestamp]
    future_parts = []
    for future_time in future_times:
        part = last_rows.copy()
        part["timestamp"] = future_time
        for column in MEASUREMENT_COLUMNS:
            if column in part.columns:
                part[column] = np.nan
            if f"{column}_obs" in part.columns:
                part[f"{column}_obs"] = np.nan
        part["weather_condition"] = None
        part["congestion_level"] = None
        part["row_observed"] = False
        part["volume_target_observed"] = False
        part["accident_observed"] = False
        part["travel_time_imputed"] = True
        part["congestion_source"] = "future"
        part["date"] = future_time.tz_localize(None).normalize()
        part["time"] = future_time.strftime("%H:%M")
        for flag_column in ["public_holiday", "event_flag", "roadwork_flag"]:
            part[flag_column] = 0                               # unknown future dates: no flags
        if calendar is not None:
            match = calendar.loc[calendar["date"] == part["date"].iloc[0]]
            if len(match) == 1:
                for flag_column in ["public_holiday", "event_flag", "roadwork_flag"]:
                    part[flag_column] = int(match[flag_column].iloc[0])
        future_parts.append(part)
    extended = pd.concat([clean] + future_parts, ignore_index=True)
    return extended.sort_values(["road_id", "timestamp"]).reset_index(drop=True), future_times


def predict_next_windows(clean, config):
    """Forecast t+1..t+4 for every segment using the saved models.

    Returns:
        DataFrame: road_id, horizon, timestamp, origin_timestamp, pred_volume (+ 90% interval),
        congestion label/confidence, travel time (+ interval), accident risk and flag.
    """
    models_dir = config["paths"]["models_dir"]
    registry = joblib.load(models_dir / "registry.joblib")
    order = config["congestion_order"]
    extended, future_times = extend_with_future(clean, config, n_future=max(config["horizons"]))
    rows = []
    for horizon in config["horizons"]:
        table = build_features(extended, config, horizon)
        target_rows = table.loc[table["timestamp"] == future_times[horizon - 1]].copy()
        hours = target_rows["timestamp"].dt.hour
        if horizon == 1:
            entry = registry["targets"]["volume"]
            volume_model = joblib.load(models_dir / entry["best_file"])
            interval_model = entry["interval_model"]
            feature_columns = entry["feature_columns"]
        else:
            entry = registry["horizons"][horizon]
            volume_model = joblib.load(models_dir / entry["file"])
            interval_model = entry["interval_model"]
            feature_columns = entry["feature_columns"]
        X = target_rows[feature_columns].to_numpy(float)
        predicted_volume = volume_model.predict(X)
        lower, upper = apply_residual_quantiles(hours, predicted_volume, interval_model)
        output = pd.DataFrame({
            "road_id": target_rows["road_id"].to_numpy(),
            "horizon": horizon,
            "timestamp": target_rows["timestamp"].to_numpy(),
            "origin_timestamp": target_rows["origin_timestamp"].to_numpy(),
            "pred_volume": predicted_volume,
            "pred_volume_lower": lower,
            "pred_volume_upper": upper,
            "road_capacity": target_rows["road_capacity"].to_numpy(),
        })
        output["congestion_from_volume"] = derive_congestion(predicted_volume, output["road_capacity"], config)
        low_band = derive_congestion(lower, output["road_capacity"], config)
        high_band = derive_congestion(upper, output["road_capacity"], config)
        output["congestion_range_90"] = [low if low == high else f"{low} to {high}"
                                         for low, high in zip(low_band, high_band)]
        if horizon == 1:
            congestion_entry = registry["targets"]["congestion"]
            congestion_model = joblib.load(models_dir / congestion_entry["best_file"])
            probabilities = congestion_model.predict_proba(target_rows[congestion_entry["feature_columns"]].to_numpy(float))
            output["pred_congestion_classifier"] = [order[code] for code in probabilities.argmax(axis=1)]
            output["pred_congestion_confidence"] = probabilities.max(axis=1)
            if congestion_entry.get("route") == "volume_bands":
                output["pred_congestion"] = output["congestion_from_volume"]
            else:
                output["pred_congestion"] = output["pred_congestion_classifier"]
            travel_entry = registry["targets"]["travel_time"]
            travel_model = joblib.load(models_dir / travel_entry["best_file"])
            travel_time = travel_model.predict(target_rows[travel_entry["feature_columns"]].to_numpy(float))
            travel_lower, travel_upper = apply_residual_quantiles(hours, travel_time, travel_entry["interval_model"])
            output["pred_travel_time"] = travel_time
            output["pred_travel_time_lower"] = travel_lower
            output["pred_travel_time_upper"] = travel_upper
            accident_entry = registry["targets"]["accident"]
            accident_model = joblib.load(models_dir / accident_entry["best_file"])
            risk = accident_model.predict_proba(target_rows[accident_entry["feature_columns"]].to_numpy(float))[:, 1]
            if accident_entry.get("positive_prior") is not None:
                risk = prior_correct(risk, accident_entry["positive_prior"])
            output["accident_risk"] = risk
            output["accident_flag"] = (risk >= accident_entry["threshold"]).astype(int)
        rows.append(output)
    return pd.concat(rows, ignore_index=True)
