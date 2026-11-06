"""Read-only access to the pipeline outputs. Nothing here trains or changes data."""

import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

CONFIG = load_config()
PROCESSED = CONFIG["paths"]["processed_dir"]
REPORTS = CONFIG["paths"]["reports_dir"]
STATIC = Path(__file__).resolve().parent / "static"
ZONES = {"WS-NORTH": "North", "WS-CENTRAL": "Central", "WS-SOUTH": "South"}
REQUIRED = [PROCESSED / "flowcast_clean.parquet", PROCESSED / "predictions_h1.parquet",
            PROCESSED / "predictions_horizons.parquet", PROCESSED / "live_predictions.parquet",
            REPORTS / "scoreboard.csv"]


def ready():
    """True when every output the dashboard reads exists."""
    return all(path.exists() for path in REQUIRED)


def version():
    """Changes whenever the pipeline finishes a run, so the cache below refreshes by itself."""
    info = REPORTS / "run_info.json"
    return info.stat().st_mtime if info.exists() else 0.0


def _local(frame):
    """Show timestamps as local wall-clock time (IST) without the offset."""
    for column in ["timestamp", "origin_timestamp"]:
        if column in frame.columns and getattr(frame[column].dt, "tz", None) is not None:
            frame[column] = frame[column].dt.tz_localize(None)
    return frame


@st.cache_data(show_spinner="Loading the latest forecasts", max_entries=2)
def load(run_version):
    """All dashboard inputs in one dictionary. `run_version` only keys the cache."""
    out = {
        "clean": _local(pd.read_parquet(PROCESSED / "flowcast_clean.parquet")),
        "pred": _local(pd.read_parquet(PROCESSED / "predictions_h1.parquet")),
        "pred_h": _local(pd.read_parquet(PROCESSED / "predictions_horizons.parquet")),
        "live": _local(pd.read_parquet(PROCESSED / "live_predictions.parquet")),
        "board": pd.read_csv(REPORTS / "scoreboard.csv"),
        "importance": pd.read_csv(PROCESSED / "feature_importance.csv"),
        "weather": pd.read_csv(PROCESSED / "weather_effect.csv"),
        "dq": pd.DataFrame(json.loads((REPORTS / "data_quality_log.json").read_text())),
    }
    history = REPORTS / "dl_training_history.csv"
    out["dl_history"] = pd.read_csv(history) if history.exists() else pd.DataFrame()
    if "selected" in out["board"]:
        out["board"]["selected"] = out["board"]["selected"].fillna(0).astype(int)

    clean = out["clean"]
    segments = clean.groupby("road_id")[["road_name", "weather_station_id", "road_capacity", "latitude", "longitude"]].first()
    segments["zone"] = segments["weather_station_id"].map(ZONES)
    segments["zone_rank"] = segments["zone"].map({"North": 0, "Central": 1, "South": 2})
    segments["label"] = segments.index + " " + segments["road_name"]
    out["segments"] = segments.sort_values("zone_rank", kind="stable")

    # "Usual" = median of recorded values for the same road, weekday and half-hour.
    recorded = clean[clean["row_observed"]].assign(dow=lambda f: f["timestamp"].dt.day_of_week)
    out["usual_volume"] = recorded.groupby(["road_id", "dow", "time"])["traffic_volume_obs"].median()
    out["usual_travel"] = recorded.groupby(["road_id", "dow", "time"])["travel_time_obs"].median()
    out["base_risk"] = float((recorded["accident_count"] > 0).mean())
    out["corridor_total"] = clean.groupby("timestamp")["traffic_volume"].sum()
    return out


def usual(out, series_name, when):
    """Usual value per road for the weekday and half-hour of `when` (a Series indexed by road_id)."""
    table = out[series_name]
    key = (when.day_of_week, when.strftime("%H:%M"))
    return table.xs(key, level=["dow", "time"])


@st.cache_resource
def wards():
    """Delhi ward outlines (DataMeet, CC BY-SA 2.5 IN) drawn under the map so it works offline."""
    return json.loads((STATIC / "delhi_wards.geojson").read_text())


@st.cache_resource(ttl=3600)
def street_tiles():
    """True when the CARTO street basemap is reachable. FLOWCAST_BASEMAP=streets|offline overrides."""
    forced = os.environ.get("FLOWCAST_BASEMAP", "").lower()
    if forced in ("streets", "offline"):
        return forced == "streets"
    try:
        urllib.request.urlopen("https://basemaps.cartocdn.com/gl/positron-gl-style/style.json", timeout=2)
        return True
    except OSError:
        return False
