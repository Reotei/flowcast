"""Tests for M1 ingest and M2 cleaning."""

import numpy as np
import pandas as pd
import pytest

from src.clean import (deduplicate_traffic, derive_congestion, gap_run_length, harmonise_weather)
from src.ingest import SchemaError, parse_timestamp, validate_schema
from src.utils import DataQualityLog, load_config


def test_both_date_formats_parse_to_the_same_instant():
    traffic_style = parse_timestamp(pd.Series(["2025-02-26"]), pd.Series(["07:30"]), "%Y-%m-%d", "Asia/Kolkata")
    weather_style = parse_timestamp(pd.Series(["26/02/2025"]), pd.Series(["07:30"]), "%d/%m/%Y", "Asia/Kolkata")
    assert traffic_style.iloc[0] == weather_style.iloc[0]
    assert str(traffic_style.dt.tz) == "Asia/Kolkata"


def test_wrong_date_format_becomes_nat_not_a_guess():
    parsed = parse_timestamp(pd.Series(["26/02/2025"]), pd.Series(["07:30"]), "%Y-%m-%d", "Asia/Kolkata")
    assert parsed.isna().all()


def test_missing_column_raises_schema_error():
    with pytest.raises(SchemaError):
        validate_schema(pd.DataFrame({"road_id": []}), ["road_id", "date"], "traffic")


def test_dedup_keeps_the_most_complete_record():
    timestamp = pd.Timestamp("2025-01-01 00:00", tz="Asia/Kolkata")
    traffic = pd.DataFrame({"road_id": ["NL-001", "NL-001"], "timestamp": [timestamp, timestamp],
                            "traffic_volume": [np.nan, 300.0], "avg_speed": [40.0, 40.0]})
    deduplicated, quarantined = deduplicate_traffic(traffic, DataQualityLog())
    assert len(deduplicated) == 1
    assert deduplicated["traffic_volume"].iloc[0] == 300.0
    assert len(quarantined) == 1


@pytest.mark.parametrize("ratio, expected", [(0.49, "Free-flow"), (0.50, "Moderate"), (0.79, "Moderate"),
                                             (0.80, "Heavy"), (0.99, "Heavy"), (1.00, "Severe"), (1.7, "Severe")])
def test_congestion_bands_match_the_data_dictionary(ratio, expected):
    config = load_config()
    capacity = 2000
    label = derive_congestion([ratio * capacity / 2], [capacity], config)[0]
    assert label == expected


def test_gap_run_length_measures_consecutive_missing_windows():
    is_missing = pd.Series([False, True, False, True, True, True, False])
    assert gap_run_length(is_missing).tolist() == [0, 1, 0, 3, 3, 3, 0]


def test_weather_labels_map_to_controlled_vocabulary():
    config = load_config()
    config["grid"]["start_date"], config["grid"]["end_date"] = "2025-01-01", "2025-01-01"
    hours = pd.date_range("2025-01-01", periods=7, freq="h", tz="Asia/Kolkata")
    weather = pd.DataFrame({"station_id": "WS-NORTH", "timestamp": hours,
                            "weather_condition": ["rain ", "rainy", "RAIN", "Overcast", "FOG", "foggy", "CLEAR"],
                            "temperature": 10.0, "rainfall": 0.0, "visibility": 10000.0})
    harmonised = harmonise_weather(weather, config, DataQualityLog())
    assert harmonised["weather_condition"].iloc[:7].tolist() == ["Rain", "Rain", "Rain", "Cloudy", "Fog", "Fog", "Clear"]
    assert set(harmonised["weather_condition"].unique()) <= {"Clear", "Cloudy", "Rain", "Fog"}


def test_cleaning_produces_complete_grid_without_gaps(tiny_clean):
    assert len(tiny_clean) == 3 * 14 * 48
    assert tiny_clean.duplicated(["road_id", "timestamp"]).sum() == 0
    assert tiny_clean["traffic_volume"].isna().sum() == 0
    assert tiny_clean["congestion_level"].isna().sum() == 0
    assert tiny_clean["weather_condition"].isna().sum() == 0
    assert int((~tiny_clean["row_observed"]).sum()) == 7          # the 7 windows removed in the fixture
    assert tiny_clean["traffic_volume"].min() >= 0
    assert tiny_clean["avg_speed"].max() <= 200
    assert tiny_clean["occupancy"].max() <= 100


def test_faulty_and_imputed_targets_are_not_marked_observed(tiny_clean):
    imputed = tiny_clean.loc[tiny_clean["volume_source"].isin(["interpolated", "median", "repaired_from_count"])]
    assert len(imputed) > 0
    assert not imputed["volume_target_observed"].any()


def test_quarantine_file_records_duplicates_and_faults(tiny_clean, tiny_config):
    quarantine = pd.read_csv(tiny_config["paths"]["interim_dir"] / "quarantine.csv")
    reasons = quarantine["quarantine_reason"].str.cat(sep=" | ")
    assert "duplicate" in reasons
    assert "traffic_volume < 0" in reasons
    assert "avg_speed > 200" in reasons
    assert "occupancy > 100" in reasons
