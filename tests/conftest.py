"""Shared pytest fixtures: a tiny raw dataset with known defects and an isolated config."""

import json

import numpy as np
import pandas as pd
import pytest

from src.utils import load_config

N_SEGMENTS = 3
N_DAYS = 14


@pytest.fixture
def tiny_config(tmp_path):
    """Config pointing every output folder into tmp_path, with a 14-day grid."""
    config = load_config()
    for key in ["interim_dir", "processed_dir", "models_dir", "reports_dir", "figures_dir"]:
        config["paths"][key] = tmp_path / key
        config["paths"][key].mkdir(parents=True, exist_ok=True)
    config["grid"]["start_date"] = "2025-01-01"
    config["grid"]["end_date"] = "2025-01-14"
    config["split"]["train_end"] = "2025-01-10"
    config["split"]["val_end"] = "2025-01-12"
    return config


@pytest.fixture
def tiny_raw_dir(tmp_path):
    """Write tiny traffic/weather/calendar CSVs with every documented defect type."""
    rng = np.random.default_rng(0)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    times = pd.date_range("2025-01-01", periods=N_DAYS * 48, freq="30min")
    rows = []
    for segment_number in range(1, N_SEGMENTS + 1):
        capacity = 1500 + 100 * segment_number
        for timestamp in times:
            volume = float(int(200 + 300 * np.sin(np.pi * (timestamp.hour + timestamp.minute / 60) / 24) + rng.integers(0, 50)))
            ratio = volume / (capacity / 2)
            label = "Free-flow" if ratio < 0.5 else "Moderate" if ratio < 0.8 else "Heavy" if ratio < 1 else "Severe"
            rows.append({
                "road_id": f"NL-{segment_number:03d}", "road_name": f"Road {segment_number}",
                "latitude": 28.6, "longitude": 77.2,
                "weather_station_id": "WS-NORTH" if segment_number < 3 else "WS-SOUTH",
                "date": timestamp.strftime("%Y-%m-%d"), "time": timestamp.strftime("%H:%M"),
                "traffic_volume": volume, "vehicle_count": int(volume),
                "vehicle_type_dist": json.dumps({"2W": 0.4, "Car": 0.4, "LCV": 0.1, "HCV": 0.1}),
                "avg_speed": 40.0, "occupancy": 20.0, "congestion_level": label, "travel_time": 3.0,
                "accident_count": int(rng.random() < 0.05), "signal_timing": 40, "road_capacity": capacity,
            })
    traffic = pd.DataFrame(rows)
    traffic.loc[10, "traffic_volume"] = -5.0                      # negative volume
    traffic.loc[11, "avg_speed"] = 250.0                          # impossible speed
    traffic.loc[12, "occupancy"] = 130.0                          # impossible occupancy
    traffic.loc[20, "traffic_volume"] = np.nan                    # null volume (repairable from count)
    traffic.loc[30, "congestion_level"] = np.nan                  # blank label
    traffic = traffic.drop(index=list(range(100, 106)))           # 6 absent windows (one long gap)
    traffic = traffic.drop(index=[200])                           # 1 absent window (short gap)
    traffic = pd.concat([traffic, traffic.iloc[[0, 1, 2]]])       # 3 exact duplicates
    traffic = traffic.sample(frac=1.0, random_state=0)            # unsorted, like the real file
    traffic.to_csv(raw_dir / "traffic_sensor_log.csv", index=False)

    hours = pd.date_range("2025-01-01", periods=N_DAYS * 24, freq="h")
    labels = ["Clear", "clear", "CLEAR", "Cloudy", "Overcast", "rain ", "rainy", "RAIN", "foggy", "FOG"]
    weather_rows = []
    for station in ["WS-NORTH", "WS-SOUTH"]:
        for position, hour in enumerate(hours):
            weather_rows.append({"station_id": station, "date": hour.strftime("%d/%m/%Y"),
                                 "time": hour.strftime("%H:00"), "weather_condition": labels[position % len(labels)],
                                 "temperature": 15.0, "rainfall": 0.0, "visibility": 10000.0})
    weather = pd.DataFrame(weather_rows)
    weather.loc[5, "temperature"] = np.nan
    weather.to_csv(raw_dir / "weather_observations.csv", index=False)

    dates = pd.date_range("2025-01-01", periods=N_DAYS, freq="D")
    calendar = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "public_holiday": 0, "holiday_name": "",
                             "event_flag": 0, "event_name": "", "roadwork_flag": 0})
    calendar.loc[0, ["public_holiday", "holiday_name"]] = [1, "New Year's Day"]
    calendar.to_csv(raw_dir / "calendar_events.csv", index=False)
    return raw_dir


@pytest.fixture
def tiny_clean(tiny_raw_dir, tiny_config):
    """Cleaned tiny dataset (runs M1 + M2)."""
    from src.clean import run_cleaning

    clean, dq_log = run_cleaning(tiny_raw_dir, tiny_config)
    return clean
