"""Leakage tests for M3: features at target window T must not change when T's own data changes."""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, get_feature_columns
from src.ml_models import time_series_folds
from src.utils import assign_split

SAME_WINDOW_COLUMNS = ["traffic_volume", "traffic_volume_obs", "vehicle_count", "avg_speed", "avg_speed_obs",
                       "occupancy", "occupancy_obs", "travel_time", "travel_time_obs", "congestion_level",
                       "accident_count", "signal_timing", "signal_timing_obs"]


def test_t0_lag_points_to_the_previous_window(tiny_clean, tiny_config):
    table = build_features(tiny_clean, tiny_config, horizon=1)
    clean = tiny_clean.set_index(["road_id", "timestamp"])
    sample = table.loc[table["road_id"] == "NL-002"].iloc[50]
    previous = sample["timestamp"] - pd.Timedelta(minutes=30)
    assert sample["origin_timestamp"] == previous
    assert sample["vol_t0"] == pytest.approx(clean.loc[("NL-002", previous), "traffic_volume_obs"])


@pytest.mark.parametrize("horizon", [1, 2, 4])
def test_features_do_not_see_the_target_window_or_later(tiny_clean, tiny_config, horizon):
    """Perturb every measurement from a cut time T onwards; features of windows T .. T+h-1 must not move.

    A forecast for window T' made at origin T'-h may only use data up to T'-h, so for T' < T+h it must be
    blind to anything at or after T.
    """
    baseline = build_features(tiny_clean, tiny_config, horizon)
    feature_columns = get_feature_columns(baseline)
    cut_time = baseline.loc[baseline["split"] == "test", "timestamp"].min() + pd.Timedelta(hours=5)
    perturbed = tiny_clean.copy()
    at_or_after_cut = perturbed["timestamp"] >= cut_time
    for column in SAME_WINDOW_COLUMNS + ["temperature", "rainfall", "visibility", "weather_condition"]:
        if perturbed[column].dtype.kind in "fi":
            perturbed.loc[at_or_after_cut, column] = perturbed.loc[at_or_after_cut, column] * 3 + 999
        else:
            perturbed.loc[at_or_after_cut, column] = "Severe" if column == "congestion_level" else "Fog"
    changed = build_features(perturbed, tiny_config, horizon)
    checked_times = [cut_time + pd.Timedelta(minutes=30 * step) for step in range(horizon)]
    rows = baseline["timestamp"].isin(checked_times)
    assert rows.sum() == 3 * horizon
    pd.testing.assert_frame_equal(baseline.loc[rows, feature_columns].reset_index(drop=True),
                                  changed.loc[rows, feature_columns].reset_index(drop=True))
    # ...while the targets themselves DID change (the perturbation was real).
    assert not np.allclose(baseline.loc[rows, "y_volume"], changed.loc[rows, "y_volume"])


def test_interpolation_cannot_leak_through_a_gap(tiny_clean, tiny_config):
    """If window t0 is missing, its lag must not be interpolated from the target window T."""
    clean = tiny_clean.copy()
    segment_rows = clean.loc[clean["road_id"] == "NL-001"].sort_values("timestamp")
    t0_index, target_index = segment_rows.index[400], segment_rows.index[401]
    clean.loc[t0_index, ["traffic_volume_obs"]] = np.nan          # t0 missing in the raw feed
    baseline = build_features(clean, tiny_config, 1)
    clean.loc[target_index, ["traffic_volume", "traffic_volume_obs"]] = 50000.0
    changed = build_features(clean, tiny_config, 1)
    target_time = clean.loc[target_index, "timestamp"]
    row_before = baseline.loc[(baseline["road_id"] == "NL-001") & (baseline["timestamp"] == target_time)]
    row_after = changed.loc[(changed["road_id"] == "NL-001") & (changed["timestamp"] == target_time)]
    assert row_before["vol_t0"].iloc[0] == row_after["vol_t0"].iloc[0]


def test_no_forbidden_same_window_feature_names(tiny_clean, tiny_config):
    feature_columns = get_feature_columns(build_features(tiny_clean, tiny_config, 1))
    for forbidden in ["vehicle_count", "traffic_volume", "avg_speed", "occupancy", "congestion_level",
                      "travel_time", "y_volume", "accident_count"]:
        assert forbidden not in feature_columns


def test_split_is_chronological(tiny_clean, tiny_config):
    split = assign_split(tiny_clean["timestamp"], tiny_config)
    last_train = tiny_clean.loc[split == "train", "timestamp"].max()
    first_val = tiny_clean.loc[split == "val", "timestamp"].min()
    last_val = tiny_clean.loc[split == "val", "timestamp"].max()
    first_test = tiny_clean.loc[split == "test", "timestamp"].min()
    assert last_train < first_val <= last_val < first_test


def test_time_series_folds_never_validate_on_the_past(tiny_clean):
    timestamps = tiny_clean.sort_values("timestamp")["timestamp"].reset_index(drop=True)
    for train_rows, val_rows in time_series_folds(timestamps, 3):
        assert timestamps.iloc[train_rows].max() < timestamps.iloc[val_rows].min()
