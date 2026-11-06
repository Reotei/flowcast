"""M1 — Ingestion & validation (FR-01, FR-02).

Loads the three raw tables, checks the schema, parses the two different date formats into
one tz-aware timestamp, coerces types and range-checks values. Rows that cannot be parsed are
quarantined (written to data/interim/quarantine.csv with a reason), never silently dropped.
Fault VALUES (negative volume, speed > 200, occupancy > 100) are only flagged here; clean.py
turns them into NaN and imputes them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DataQualityLog, get_logger

TRAFFIC_COLUMNS = [
    "road_id", "road_name", "latitude", "longitude", "weather_station_id", "date", "time",
    "traffic_volume", "vehicle_count", "vehicle_type_dist", "avg_speed", "occupancy",
    "congestion_level", "travel_time", "accident_count", "signal_timing", "road_capacity",
]
TRAFFIC_NUMERIC = [
    "latitude", "longitude", "traffic_volume", "vehicle_count", "avg_speed", "occupancy",
    "travel_time", "accident_count", "signal_timing", "road_capacity",
]
WEATHER_COLUMNS = ["station_id", "date", "time", "weather_condition", "temperature", "rainfall", "visibility"]
WEATHER_NUMERIC = ["temperature", "rainfall", "visibility"]
CALENDAR_COLUMNS = ["date", "public_holiday", "holiday_name", "event_flag", "event_name", "roadwork_flag"]
KNOWN_STATIONS = ["WS-NORTH", "WS-CENTRAL", "WS-SOUTH"]


class SchemaError(ValueError):
    """Raised when a raw file is missing required columns."""


def validate_schema(frame, required_columns, table_name):
    """Raise SchemaError if any required column is missing from a raw table."""
    missing_columns = []
    for column in required_columns:
        if column not in frame.columns:
            missing_columns.append(column)
    if len(missing_columns) > 0:
        raise SchemaError(f"{table_name}: missing columns {missing_columns}")


def parse_timestamp(date_strings, time_strings, date_format, timezone):
    """Combine date and time strings into a tz-aware timestamp.

    Args:
        date_strings: Series of dates, e.g. '2025-02-26' or '26/02/2025'.
        time_strings: Series of times 'HH:MM'.
        date_format: '%Y-%m-%d' for traffic/calendar, '%d/%m/%Y' for weather.
        timezone: IANA zone name, e.g. 'Asia/Kolkata'.

    Returns:
        Series of tz-aware timestamps; unparseable entries become NaT.
    """
    combined = date_strings.astype(str).str.strip() + " " + time_strings.astype(str).str.strip()
    # An explicit format means a wrong-format date becomes NaT instead of being guessed.
    parsed = pd.to_datetime(combined, format=f"{date_format} %H:%M", errors="coerce")
    return parsed.dt.tz_localize(timezone)


def _quarantine(frame, mask, reason, table_name):
    """Return the rows selected by mask with a reason column, for the quarantine file."""
    quarantined = frame.loc[mask].copy()
    quarantined.insert(0, "quarantine_reason", reason)
    quarantined.insert(0, "source_table", table_name)
    return quarantined


def _coerce_numeric(frame, columns, table_name, dq_log):
    """Convert columns to numbers; log values that were present but not numeric."""
    for column in columns:
        was_present = frame[column].notna()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        became_nan = was_present & frame[column].isna()
        if became_nan.sum() > 0:
            dq_log.add("M1 ingest", f"{table_name}.{column} non-numeric", became_nan.sum(), "set to NaN")
    return frame


def load_calendar(path, config, dq_log):
    """Load calendar_events.csv (one row per date) and parse its date."""
    calendar = pd.read_csv(path, dtype={"holiday_name": str, "event_name": str})
    validate_schema(calendar, CALENDAR_COLUMNS, "calendar_events")
    calendar["date"] = pd.to_datetime(calendar["date"], format="%Y-%m-%d", errors="coerce")
    bad_dates = calendar["date"].isna()
    quarantined = _quarantine(calendar, bad_dates, "unparseable date", "calendar_events")
    calendar = calendar.loc[~bad_dates].copy()
    for flag_column in ["public_holiday", "event_flag", "roadwork_flag"]:
        calendar[flag_column] = pd.to_numeric(calendar[flag_column], errors="coerce").fillna(0).astype(int)
    duplicated_dates = calendar["date"].duplicated()
    quarantined = pd.concat([quarantined, _quarantine(calendar, duplicated_dates, "duplicate date", "calendar_events")])
    calendar = calendar.loc[~duplicated_dates].reset_index(drop=True)
    dq_log.add("M1 ingest", "calendar rows loaded", len(calendar), "kept",
               f"{int(bad_dates.sum())} bad dates, {int(duplicated_dates.sum())} duplicate dates quarantined")
    return calendar, quarantined


def load_weather(path, config, dq_log):
    """Load weather_observations.csv, parse DD/MM/YYYY dates and check value ranges."""
    weather = pd.read_csv(path, dtype={"station_id": str, "date": str, "time": str, "weather_condition": str})
    validate_schema(weather, WEATHER_COLUMNS, "weather_observations")
    weather["timestamp"] = parse_timestamp(weather["date"], weather["time"], "%d/%m/%Y", config["timezone"])
    weather = _coerce_numeric(weather, WEATHER_NUMERIC, "weather", dq_log)

    bad_timestamp = weather["timestamp"].isna()
    unknown_station = ~weather["station_id"].isin(KNOWN_STATIONS)
    reject = bad_timestamp | unknown_station
    quarantined = _quarantine(weather, bad_timestamp, "unparseable date/time", "weather_observations")
    quarantined = pd.concat([quarantined, _quarantine(weather, unknown_station & ~bad_timestamp,
                                                      "unknown station_id", "weather_observations")])
    weather = weather.loc[~reject].copy()

    # Range checks: physically impossible weather values become NaN (then imputed in clean.py).
    impossible_rain = weather["rainfall"] < 0
    impossible_visibility = (weather["visibility"] < 0) | (weather["visibility"] > 50000)
    weather.loc[impossible_rain, "rainfall"] = np.nan
    weather.loc[impossible_visibility, "visibility"] = np.nan
    dq_log.add("M1 ingest", "weather rows loaded", len(weather), "kept",
               f"{int(reject.sum())} quarantined; {int(impossible_rain.sum())} negative rainfall, "
               f"{int(impossible_visibility.sum())} impossible visibility set to NaN")
    return weather.reset_index(drop=True), quarantined


def load_traffic(path, config, dq_log):
    """Load traffic_sensor_log.csv, parse YYYY-MM-DD dates and flag rule-breaking values."""
    traffic = pd.read_csv(path, dtype={"road_id": str, "road_name": str, "weather_station_id": str,
                                       "date": str, "time": str, "congestion_level": str,
                                       "vehicle_type_dist": str})
    validate_schema(traffic, TRAFFIC_COLUMNS, "traffic_sensor_log")
    dq_log.add("M1 ingest", "traffic rows read", len(traffic), "loaded")
    traffic["timestamp"] = parse_timestamp(traffic["date"], traffic["time"], "%Y-%m-%d", config["timezone"])
    traffic = _coerce_numeric(traffic, TRAFFIC_NUMERIC, "traffic", dq_log)

    # Rows that can never be placed on the segment x 30-min grid are quarantined.
    bad_timestamp = traffic["timestamp"].isna()
    off_grid = traffic["timestamp"].dt.minute.isin([0, 30]) == False  # noqa: E712 (explicit for readability)
    off_grid = off_grid & ~bad_timestamp
    bad_road = traffic["road_id"].str.match(r"^NL-\d{3}$", na=False) == False  # noqa: E712
    start = pd.Timestamp(config["grid"]["start_date"], tz=config["timezone"])
    end = pd.Timestamp(config["grid"]["end_date"], tz=config["timezone"]) + pd.Timedelta(days=1)
    out_of_range = ~bad_timestamp & ((traffic["timestamp"] < start) | (traffic["timestamp"] >= end))

    quarantined_parts = [
        _quarantine(traffic, bad_timestamp, "unparseable date/time", "traffic_sensor_log"),
        _quarantine(traffic, off_grid, "time not on 30-min grid", "traffic_sensor_log"),
        _quarantine(traffic, bad_road & ~bad_timestamp, "invalid road_id", "traffic_sensor_log"),
        _quarantine(traffic, out_of_range, "date outside study period", "traffic_sensor_log"),
    ]
    reject = bad_timestamp | off_grid | bad_road | out_of_range
    for part in quarantined_parts:
        if len(part) > 0:
            dq_log.add("M1 ingest", f"traffic {part['quarantine_reason'].iloc[0]}", len(part), "quarantined")
    traffic = traffic.loc[~reject].copy()

    # Flag (do not yet change) the physically impossible values.
    rules = config["outlier_rules"]
    traffic["flag_negative_volume"] = traffic["traffic_volume"] < rules["traffic_volume_min"]
    traffic["flag_speed_over_max"] = traffic["avg_speed"] > rules["avg_speed_max"]
    traffic["flag_occupancy_over_max"] = traffic["occupancy"] > rules["occupancy_max"]
    return traffic.reset_index(drop=True), pd.concat(quarantined_parts)


def load_all(raw_dir, config, dq_log=None):
    """Load and validate all three raw tables (FR-01, FR-02).

    Load order follows the Data Dictionary: calendar, then weather, then traffic.

    Returns:
        (traffic, weather, calendar, quarantine, dq_log)
    """
    logger = get_logger()
    if dq_log is None:
        dq_log = DataQualityLog()
    raw_dir = Path(raw_dir)
    file_names = config["files"]
    for key in ["traffic", "weather", "calendar"]:
        file_path = raw_dir / file_names[key]
        if not file_path.exists():
            raise FileNotFoundError(f"Missing raw file: {file_path}")

    calendar, calendar_quarantine = load_calendar(raw_dir / file_names["calendar"], config, dq_log)
    weather, weather_quarantine = load_weather(raw_dir / file_names["weather"], config, dq_log)
    traffic, traffic_quarantine = load_traffic(raw_dir / file_names["traffic"], config, dq_log)
    quarantine = pd.concat([calendar_quarantine, weather_quarantine, traffic_quarantine], ignore_index=True)
    logger.info(f"Loaded traffic {traffic.shape}, weather {weather.shape}, calendar {calendar.shape}")
    return traffic, weather, calendar, quarantine, dq_log
