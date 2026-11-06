"""M2 — Cleaning & wrangling (FR-03, FR-04, FR-06).

Ordered stages (PRD §9.1 plus the project data rules):
    1. dedup on road_id + date + time            (before anything else)
    2. rule-based outliers -> NaN, z-score flags  (logged + quarantined copy)
    3. parse vehicle_type_dist JSON -> 4 shares
    4. reindex every segment to the full 30-min grid (25 x 151 x 48 = 181,200)
    5. repair volume from vehicle_count, then impute
       short gaps (<= 2 windows) -> time interpolation within the segment
       long gaps                  -> segment x time-window median (TRAIN dates only)
    6. derive blank congestion_level from V/C
    7. harmonise weather labels, impute weather gaps
    8. merge weather (station + floored hour) and calendar (date)

For every time-varying column a '<column>_obs' copy keeps the value BEFORE imputation.
features.py builds lags from those with a causal (past-only) fill, so an interpolated value
can never carry information from the target window into a lag feature.
"""

import json

import numpy as np
import pandas as pd

from src.ingest import load_all
from src.utils import DataQualityLog, assign_split, ensure_dirs, get_logger

SHARE_COLUMNS = {"2W": "share_2w", "Car": "share_car", "LCV": "share_lcv", "HCV": "share_hcv"}
IMPUTED_COLUMNS = ["traffic_volume", "avg_speed", "occupancy", "travel_time", "signal_timing",
                   "share_2w", "share_car", "share_lcv", "share_hcv"]
STATIC_SEGMENT_COLUMNS = ["road_name", "latitude", "longitude", "weather_station_id", "road_capacity"]


# ----------------------------------------------------------------------------- 1. dedup
def deduplicate_traffic(traffic, dq_log):
    """Drop key duplicates (road_id + timestamp), keeping the most complete record.

    Returns:
        (deduplicated traffic, quarantined duplicate rows)
    """
    exact_duplicates = traffic.duplicated(subset=[c for c in traffic.columns if not c.startswith("flag_")])
    dq_log.add("M2 dedup", "exact duplicate rows", exact_duplicates.sum(), "removed (quarantined)")

    traffic = traffic.copy()
    traffic["_non_null_count"] = traffic.notna().sum(axis=1)
    # Sort so that, within each key, the row with the most non-null values comes first.
    traffic = traffic.sort_values(["road_id", "timestamp", "_non_null_count"], ascending=[True, True, False])
    key_duplicate = traffic.duplicated(subset=["road_id", "timestamp"], keep="first")
    quarantined = traffic.loc[key_duplicate].drop(columns="_non_null_count").copy()
    quarantined.insert(0, "quarantine_reason", "duplicate road_id+timestamp")
    quarantined.insert(0, "source_table", "traffic_sensor_log")
    dq_log.add("M2 dedup", "key duplicates (road_id+date+time)", key_duplicate.sum(),
               "removed (quarantined)", "kept the most complete record per key")
    deduplicated = traffic.loc[~key_duplicate].drop(columns="_non_null_count").reset_index(drop=True)
    return deduplicated, quarantined


# ----------------------------------------------------------------------------- 2. outliers
def apply_outlier_rules(traffic, dq_log):
    """Set physically impossible values to NaN (flags were added in ingest.py).

    Returns:
        (traffic with faults set to NaN, quarantined copies of the faulty rows)
    """
    traffic = traffic.copy()
    rule_to_column = {
        "flag_negative_volume": ("traffic_volume", "traffic_volume < 0"),
        "flag_speed_over_max": ("avg_speed", "avg_speed > 200 km/h"),
        "flag_occupancy_over_max": ("occupancy", "occupancy > 100 %"),
    }
    quarantined_parts = []
    for flag_column, (value_column, rule_text) in rule_to_column.items():
        fault_mask = traffic[flag_column].fillna(False).astype(bool)
        part = traffic.loc[fault_mask].copy()
        part.insert(0, "quarantine_reason", f"sensor fault: {rule_text} (value set to NaN, row kept)")
        part.insert(0, "source_table", "traffic_sensor_log")
        quarantined_parts.append(part)
        traffic.loc[fault_mask, value_column] = np.nan
        dq_log.add("M2 outliers", rule_text, fault_mask.sum(), "set to NaN, then imputed")
    return traffic, pd.concat(quarantined_parts)


def flag_statistical_outliers(traffic, config, dq_log):
    """Flag |z| > threshold within each segment x time-of-day group (flag only, values kept).

    z = (x - mean) / std. Peaks are real traffic, so extreme-but-possible values are not changed;
    the flag is logged so the data-quality report shows how many exist.
    """
    threshold = config["outlier_rules"]["zscore_flag_threshold"]
    traffic = traffic.copy()
    time_of_day = traffic["timestamp"].dt.strftime("%H:%M")
    for column in ["traffic_volume", "avg_speed", "occupancy"]:
        group_mean = traffic.groupby([traffic["road_id"], time_of_day])[column].transform("mean")
        group_std = traffic.groupby([traffic["road_id"], time_of_day])[column].transform("std")
        z_score = (traffic[column] - group_mean) / group_std
        flag = z_score.abs() > threshold
        traffic[f"zflag_{column}"] = flag
        dq_log.add("M2 outliers", f"{column} |z| > {threshold} (segment x window)", flag.sum(), "flagged only")
    return traffic


# ----------------------------------------------------------------------------- 3. JSON
def parse_vehicle_mix(traffic, dq_log):
    """Parse the vehicle_type_dist JSON string into four share columns."""
    traffic = traffic.copy()
    parsed_rows = []
    bad_json_count = 0
    for raw_value in traffic["vehicle_type_dist"]:
        shares = {new_name: np.nan for new_name in SHARE_COLUMNS.values()}
        if isinstance(raw_value, str):
            try:
                decoded = json.loads(raw_value)
                for json_key, new_name in SHARE_COLUMNS.items():
                    shares[new_name] = float(decoded.get(json_key, np.nan))
            except (json.JSONDecodeError, TypeError, ValueError):
                bad_json_count += 1
        else:
            bad_json_count += 1
        parsed_rows.append(shares)
    share_frame = pd.DataFrame(parsed_rows, index=traffic.index)
    traffic = pd.concat([traffic.drop(columns="vehicle_type_dist"), share_frame], axis=1)
    dq_log.add("M2 JSON", "vehicle_type_dist unparseable", bad_json_count, "shares set to NaN, then imputed")
    return traffic


# ----------------------------------------------------------------------------- 4. grid
def reindex_to_full_grid(traffic, config, dq_log):
    """Reindex every segment to the complete 30-min grid so t-1 always means the previous window."""
    timezone = config["timezone"]
    full_times = pd.date_range(config["grid"]["start_date"],
                               pd.Timestamp(config["grid"]["end_date"]) + pd.Timedelta(hours=23, minutes=30),
                               freq=f"{config['grid']['window_minutes']}min", tz=timezone)
    segment_ids = sorted(traffic["road_id"].unique())
    full_index = pd.MultiIndex.from_product([segment_ids, full_times], names=["road_id", "timestamp"])

    traffic = traffic.copy()
    traffic["row_observed"] = True
    grid = traffic.set_index(["road_id", "timestamp"]).reindex(full_index).reset_index()
    grid["row_observed"] = grid["row_observed"].fillna(False).astype(bool)
    for column in grid.columns:
        if column.startswith("flag_") or column.startswith("zflag_"):
            grid[column] = grid[column].fillna(False).astype(bool)   # absent rows have no faults

    # Static segment attributes are constant per segment, so copy them into the new rows.
    static_lookup = traffic.groupby("road_id")[STATIC_SEGMENT_COLUMNS].agg(lambda values: values.mode().iloc[0])
    for column in STATIC_SEGMENT_COLUMNS:
        grid[column] = grid["road_id"].map(static_lookup[column])

    expected_rows = len(segment_ids) * len(full_times)
    missing_windows = int((~grid["row_observed"]).sum())
    dq_log.add("M2 grid", "absent windows added by reindex", missing_windows, "row added, values imputed",
               f"{len(segment_ids)} segments x {len(full_times)} windows = {expected_rows:,} rows")
    return grid


# ----------------------------------------------------------------------------- 5. imputation
def gap_run_length(is_missing):
    """For each missing value, return the length of the consecutive-NaN run it belongs to.

    Args:
        is_missing: boolean Series in time order for ONE segment.

    Returns:
        integer Series (0 where the value is present).
    """
    run_id = (is_missing != is_missing.shift()).cumsum()
    run_size = is_missing.groupby(run_id).transform("size")
    return run_size.where(is_missing, 0)


def repair_volume_from_count(grid, dq_log):
    """Fill missing/faulty traffic_volume from vehicle_count when the row exists.

    vehicle_count equals traffic_volume in ~99.9% of rows, so it is a near-exact repair. These
    repaired rows are still NOT counted as observed targets at evaluation time.
    """
    grid = grid.copy()
    both_present = grid["traffic_volume"].notna() & grid["vehicle_count"].notna()
    equal_share = (grid.loc[both_present, "traffic_volume"] == grid.loc[both_present, "vehicle_count"]).mean()
    dq_log.add("M2 impute", "vehicle_count == traffic_volume", both_present.sum(), "checked",
               f"equal in {equal_share:.2%} of rows -> vehicle_count is the target, never a same-window feature")

    grid["volume_source"] = np.where(grid["traffic_volume"].notna(), "observed", "missing")
    can_repair = grid["traffic_volume"].isna() & grid["vehicle_count"].notna()
    grid.loc[can_repair, "traffic_volume"] = grid.loc[can_repair, "vehicle_count"]
    grid.loc[can_repair, "volume_source"] = "repaired_from_count"
    dq_log.add("M2 impute", "traffic_volume repaired from vehicle_count", can_repair.sum(), "filled")
    return grid


def impute_series_columns(grid, columns, config, dq_log):
    """Impute each column: short gaps by interpolation, long gaps by train-window medians.

    The medians are computed per segment x time-of-day using TRAIN dates only, so no
    validation/test information is used to fill anything.
    """
    grid = grid.sort_values(["road_id", "timestamp"]).reset_index(drop=True)
    max_short_gap = config["imputation"]["short_gap_max_windows"]
    is_train = grid["split"] == "train"
    time_of_day = grid["timestamp"].dt.strftime("%H:%M")

    for column in columns:
        grid[f"{column}_obs"] = grid[column]                      # keep the pre-imputation value
        is_missing = grid[column].isna()
        run_length = grid.groupby("road_id")[column].transform(lambda values: gap_run_length(values.isna()))

        # Linear interpolation on a regular 30-min grid is the same as time interpolation.
        interpolated = grid.groupby("road_id")[column].transform(
            lambda values: values.interpolate(method="linear", limit_area="inside"))
        use_interpolation = is_missing & (run_length <= max_short_gap) & interpolated.notna()
        grid.loc[use_interpolation, column] = interpolated[use_interpolation]

        train_median = grid.loc[is_train].groupby([grid["road_id"], time_of_day])[f"{column}_obs"].median()
        median_lookup_keys = pd.MultiIndex.from_arrays([grid["road_id"], time_of_day])
        median_values = pd.Series(train_median.reindex(median_lookup_keys).to_numpy(), index=grid.index)
        use_median = grid[column].isna()
        grid.loc[use_median, column] = median_values[use_median]

        grid[f"{column}_imputed"] = is_missing
        dq_log.add("M2 impute", f"{column} short gaps (<= {max_short_gap} windows)", use_interpolation.sum(),
                   "time interpolation within segment")
        dq_log.add("M2 impute", f"{column} long gaps / edges", use_median.sum(),
                   "segment x time-window median (train dates only)")
        if column == "traffic_volume":
            grid.loc[use_interpolation, "volume_source"] = "interpolated"
            grid.loc[use_median, "volume_source"] = "median"
    return grid


# ----------------------------------------------------------------------------- 6. congestion
def derive_congestion(volume, capacity, config):
    """Band V/C = volume / (capacity / 2) into the four documented congestion levels.

    Bands: < 0.50 Free-flow, 0.50-0.79 Moderate, 0.80-0.99 Heavy, >= 1.00 Severe.
    """
    vc_ratio = np.asarray(volume, dtype=float) / (np.asarray(capacity, dtype=float) / 2.0)
    labels = np.empty(len(vc_ratio), dtype=object)
    for row_number in range(len(vc_ratio)):
        ratio = vc_ratio[row_number]
        if np.isnan(ratio):
            labels[row_number] = None
            continue
        for band in config["congestion_bands"]:
            if ratio < float(band["upper"]):
                labels[row_number] = band["label"]
                break
    return labels


def fill_congestion(grid, config, dq_log):
    """Keep original labels; derive the blank ones from V/C and record where each came from."""
    grid = grid.copy()
    derived = derive_congestion(grid["traffic_volume"], grid["road_capacity"], config)
    grid["congestion_derived"] = derived

    has_label = grid["congestion_level"].notna()
    has_observed_volume = grid["volume_source"] == "observed"
    comparable = has_label & has_observed_volume
    agreement = (grid.loc[comparable, "congestion_level"] == grid.loc[comparable, "congestion_derived"]).mean()
    dq_log.add("M2 congestion", "V/C rule reproduces existing labels", comparable.sum(), "checked",
               f"agreement {agreement:.2%} -> congestion is a function of same-window volume (leakage risk)")

    grid["congestion_source"] = "original"
    blank = ~has_label
    grid.loc[blank & has_observed_volume, "congestion_source"] = "derived_observed_volume"
    grid.loc[blank & ~has_observed_volume, "congestion_source"] = "derived_imputed_volume"
    grid.loc[blank, "congestion_level"] = grid.loc[blank, "congestion_derived"]
    dq_log.add("M2 congestion", "blank congestion_level (observed volume)", (blank & has_observed_volume).sum(),
               "derived from V/C")
    dq_log.add("M2 congestion", "blank congestion_level (imputed volume)", (blank & ~has_observed_volume).sum(),
               "derived from V/C, excluded from evaluation")
    return grid.drop(columns="congestion_derived")


# ----------------------------------------------------------------------------- 7. weather
def harmonise_weather(weather, config, dq_log):
    """Normalise labels to Clear / Cloudy / Rain / Fog, complete the hourly grid, fill gaps."""
    weather = weather.copy()
    label_map = config["weather_label_map"]
    normalised = weather["weather_condition"].astype(str).str.strip().str.lower()
    raw_label_count = weather["weather_condition"].nunique()
    overcast_count = int((normalised == "overcast").sum())
    weather["weather_condition"] = normalised.map(label_map)
    unknown = weather["weather_condition"].isna()
    dq_log.add("M2 weather", f"labels normalised ({raw_label_count} raw spellings)", len(weather),
               "strip + lowercase + map to Clear/Cloudy/Rain/Fog")
    dq_log.add("M2 weather", "'Overcast' (not in documented vocabulary)", overcast_count, "mapped to Cloudy")
    dq_log.add("M2 weather", "unrecognised weather labels", unknown.sum(), "forward-filled from previous hour")

    stations = sorted(weather["station_id"].unique())
    hours = pd.date_range(config["grid"]["start_date"],
                          pd.Timestamp(config["grid"]["end_date"]) + pd.Timedelta(hours=23),
                          freq="h", tz=config["timezone"])
    full_index = pd.MultiIndex.from_product([stations, hours], names=["station_id", "timestamp"])
    duplicate_hours = weather.duplicated(["station_id", "timestamp"])
    weather = weather.loc[~duplicate_hours].set_index(["station_id", "timestamp"])
    missing_hours = len(full_index) - len(weather.index.intersection(full_index))
    weather = weather.reindex(full_index).reset_index()
    dq_log.add("M2 weather", "duplicate station-hours", duplicate_hours.sum(), "removed")
    dq_log.add("M2 weather", "missing station-hours", missing_hours, "row added, forward-filled",
               f"{len(stations)} stations x {len(hours)} hours = {len(full_index):,}")

    weather = weather.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    for column in ["weather_condition", "temperature", "rainfall", "visibility"]:
        missing_count = int(weather[column].isna().sum())
        # Forward fill uses only the past; backward fill only covers the very first hours.
        weather[column] = weather.groupby("station_id")[column].transform(lambda values: values.ffill().bfill())
        dq_log.add("M2 weather", f"{column} missing", missing_count, "forward-filled within station")
    return weather[["station_id", "timestamp", "weather_condition", "temperature", "rainfall", "visibility"]]


# ----------------------------------------------------------------------------- 8. merge
def merge_sources(grid, weather, calendar, dq_log):
    """Join hourly weather on (station, floored hour) and calendar on date."""
    rows_before = len(grid)
    grid = grid.copy()
    grid["weather_hour"] = grid["timestamp"].dt.floor("h")
    weather_renamed = weather.rename(columns={"station_id": "weather_station_id", "timestamp": "weather_hour"})
    merged = grid.merge(weather_renamed, on=["weather_station_id", "weather_hour"], how="left")

    merged["date"] = merged["timestamp"].dt.tz_localize(None).dt.normalize()
    merged = merged.merge(calendar, on="date", how="left")
    for flag_column in ["public_holiday", "event_flag", "roadwork_flag"]:
        merged[flag_column] = merged[flag_column].fillna(0).astype(int)

    assert len(merged) == rows_before, "merge changed the row count"
    missing_weather = int(merged["weather_condition"].isna().sum())
    dq_log.add("M2 merge", "rows after weather + calendar join", len(merged), "kept",
               f"{missing_weather} rows without weather")
    merged["time"] = merged["timestamp"].dt.strftime("%H:%M")
    return merged.drop(columns=["weather_hour"])


# ----------------------------------------------------------------------------- driver
def run_cleaning(raw_dir, config):
    """Run M1 + M2 end to end and write the clean table, quarantine file and DQ log.

    Returns:
        (clean DataFrame, DataQualityLog)
    """
    logger = get_logger()
    ensure_dirs(config)
    dq_log = DataQualityLog()
    traffic, weather, calendar, quarantine, dq_log = load_all(raw_dir, config, dq_log)

    traffic, duplicate_rows = deduplicate_traffic(traffic, dq_log)
    traffic, fault_rows = apply_outlier_rules(traffic, dq_log)
    traffic = flag_statistical_outliers(traffic, config, dq_log)
    traffic = parse_vehicle_mix(traffic, dq_log)

    grid = reindex_to_full_grid(traffic, config, dq_log)
    grid["split"] = assign_split(grid["timestamp"], config)
    grid = repair_volume_from_count(grid, dq_log)
    grid = impute_series_columns(grid, IMPUTED_COLUMNS, config, dq_log)
    grid["volume_target_observed"] = grid["volume_source"] == "observed"
    grid["accident_observed"] = grid["row_observed"] & grid["accident_count"].notna()
    grid = fill_congestion(grid, config, dq_log)

    weather_clean = harmonise_weather(weather, config, dq_log)
    clean = merge_sources(grid, weather_clean, calendar, dq_log)

    quarantine = pd.concat([quarantine, duplicate_rows, fault_rows], ignore_index=True)
    interim_dir = config["paths"]["interim_dir"]
    processed_dir = config["paths"]["processed_dir"]
    quarantine.astype(str).to_csv(interim_dir / "quarantine.csv", index=False)
    weather_clean.to_parquet(interim_dir / "weather_clean.parquet", index=False)
    clean.to_parquet(processed_dir / "flowcast_clean.parquet", index=False)
    dq_log.add("M2 output", "quarantine file rows", len(quarantine), "written to data/interim/quarantine.csv")
    dq_log.save(config["paths"]["reports_dir"] / "data_quality_log.json")
    logger.info(f"Clean table: {clean.shape} -> {processed_dir / 'flowcast_clean.parquet'}")
    return clean, dq_log
