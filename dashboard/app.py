"""FlowCast dashboard (PRD M7, FR-13 to FR-18).

    streamlit run dashboard/app.py

Reads the outputs written by `python -m src.pipeline`; it never trains on page load.
Pages: Now (map and what needs attention), Today (the day replayed), Roads, Accuracy, Data.
Styles and components live in dashboard/ui.py, data access in dashboard/store.py.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FlowCast", page_icon=str(Path(__file__).parent / "static" / "mark.svg"), layout="wide")

import store as d  # noqa: E402  (dashboard/ is on the path when Streamlit runs this file)
import ui  # noqa: E402

HORIZONS = {1: "30 min", 2: "1 hour", 3: "90 min", 4: "2 hours"}
SLOTS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
LOCK = d.REPORTS / "pipeline.lock"
PAGES = {}


def chosen(widget_value, default):
    """Segmented controls return None when the active option is clicked again."""
    return default if widget_value is None else widget_value


def data_through(out):
    """'Sensor data to 31 May 2025, 23:30' for page titles."""
    return f"Sensor data to {out['clean']['timestamp'].max():%d %b %Y, %H:%M}"


def pick_road(out, key, label="Road"):
    """Road picker that remembers the road chosen on the map or on another page."""
    labels = out["segments"]["label"]
    st.session_state.setdefault("road", labels.index[0])
    st.session_state[key] = st.session_state["road"]          # every picker shows the same road

    def remember():
        st.session_state["road"] = st.session_state[key]

    return st.selectbox(label, list(labels.index), format_func=labels.get, key=key, on_change=remember)


# ====================================================================== Now
def road_rows(out, frame, when, level_of, risk=None, flag=None, travel=None):
    """One dict per road, in corridor order, with the comparison against a usual week."""
    usual = d.usual(out, "usual_volume", when)
    frame = frame.set_index("road_id")
    rows = []
    for road_id, segment in out["segments"].iterrows():
        row = frame.loc[road_id]
        usual_volume = usual.get(road_id)
        vs_usual = None
        if usual_volume is not None and usual_volume > 0:
            vs_usual = row["pred_volume"] / usual_volume - 1
        rows.append({
            "zone": segment["zone"], "road_id": road_id, "name": segment["road_name"], "level": level_of(row),
            "volume": row["pred_volume"], "lower": row["pred_volume_lower"], "upper": row["pred_volume_upper"],
            "usual": usual_volume, "vs_usual": vs_usual, "vc": row["pred_volume"] / (segment["road_capacity"] / 2),
            "risk": None if risk is None else risk[road_id],
            "flag": False if flag is None else bool(flag[road_id]),
            "travel": None if travel is None else travel[road_id],
        })
    return rows


def needs_attention(rows, limit=6):
    """Roads an operator should look at, most urgent first."""
    c = ui.colors()
    items = []
    for row in rows:
        busier = row["vs_usual"] is not None and row["vs_usual"] >= 0.3 and row["vc"] >= 0.5
        if row["flag"]:
            items.append((0, -row["risk"], {"colour": c["alert"], "name": row["name"], "value": f"{row['risk']:.1%}",
                                            "reason": f"Patrol suggested. {row['level']}, {row['volume']:,.0f} vehicles"}))
        elif row["level"] in ("Severe", "Heavy"):
            rank = 1 if row["level"] == "Severe" else 2
            usual_text = "" if row["vs_usual"] is None else f", {row['vs_usual']:+.0%} on usual"
            items.append((rank, -row["vc"], {"colour": c["levels"][row["level"]], "name": row["name"],
                                             "value": f"{row['vc']:.0%}",
                                             "reason": f"{row['level']}, capacity used{usual_text}"}))
        elif busier:
            items.append((3, -row["vs_usual"], {"colour": c["levels"]["Moderate"], "name": row["name"],
                                                "value": f"{row['vs_usual']:+.0%}",
                                                "reason": f"Busier than usual for this time, {row['volume']:,.0f} vehicles"}))
    items.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in items[:limit]], max(len(items) - limit, 0)


def headline(rows, when):
    """Plain sentence for the status band."""
    slow = [r for r in rows if r["level"] in ("Heavy", "Severe")]
    flagged = [r for r in rows if r["flag"]]
    if not slow and not flagged:
        return "Traffic is moving freely", f"All {len(rows)} roads are expected to flow at {when:%H:%M}."
    if slow:
        zones = pd.Series([r["zone"] for r in slow]).value_counts()
        severe = sum(r["level"] == "Severe" for r in slow)
        text = f"{severe} Severe and {len(slow) - severe} Heavy, most on the {zones.index[0]} stretch."
        title = f"{len(slow)} road{'s' if len(slow) > 1 else ''} slow at {when:%H:%M}"
    else:
        text = f"Every road is flowing at {when:%H:%M}."
        title = f"{len(flagged)} patrol{'s' if len(flagged) > 1 else ''} suggested"
    if flagged:
        worst = max(flagged, key=lambda r: r["risk"])
        text += f" Patrol suggested on {len(flagged)}, highest risk on {ui.esc(worst['name'])}."
    return title, text


def selected_road(out, rows, when, horizon):
    """Compact detail for one road, next to the map."""
    c = ui.colors()
    road_id = pick_road(out, "now_road")
    row = next(r for r in rows if r["road_id"] == road_id)
    usual_text = "" if row["vs_usual"] is None else f"{row['vs_usual']:+.0%} on usual"
    items = [{"v": f"{row['volume']:,.0f}", "l": f"vehicles, {row['level']}",
              "d": f"likely {row['lower']:,.0f} to {row['upper']:,.0f}. {usual_text}"}]
    if row["travel"] is not None:
        items.append({"v": f"{row['travel']:.1f}", "unit": "min", "l": "to drive this road"})
        items.append({"v": f"{row['risk']:.1%}", "l": "accident risk", "up": row["flag"],
                      "d": "patrol suggested" if row["flag"] else "below the patrol line"})
    else:
        items.append({"v": f"{row['vc']:.0%}", "l": "of capacity used"})
    ui.numbers(items)
    recorded = out["clean"][(out["clean"]["road_id"] == road_id) & (out["clean"]["timestamp"] <= when)]
    recorded = recorded.set_index("timestamp")["traffic_volume"].tail(16)
    fig = ui.figure()
    fig.add_trace(go.Scatter(x=recorded.index, y=recorded.values, name="Recorded", line={"color": c["recorded"], "width": 3}))
    fig.add_trace(go.Scatter(x=[when], y=[row["volume"]], name=f"Forecast, {HORIZONS[horizon]} ahead", mode="markers",
                             marker={"color": c["accent"], "size": 9},
                             error_y={"type": "data", "symmetric": False, "color": c["accent"], "thickness": 1.2,
                                      "array": [row["upper"] - row["volume"]], "arrayminus": [row["volume"] - row["lower"]]}))
    ui.show(ui.style(fig, height=170, legend=False).update_xaxes(tickformat="%H:%M"))
    st.page_link(PAGES["roads"], label="Full history for this road", icon=":material/timeline:")


def page_now():
    """FR-13: where traffic will be slow and where patrols are needed, on a map."""
    out = d.load(d.version())
    band = st.container()
    left, right = st.columns([3, 2])
    with left:
        mode = chosen(st.segmented_control("Show", ["Forecast", "Look back"], default="Forecast", key="now_mode",
                                           label_visibility="collapsed"), "Forecast")
    if mode == "Forecast":
        live = out["live"]
        with right:
            horizon = chosen(st.segmented_control("Ahead", list(HORIZONS), default=1, format_func=HORIZONS.get,
                                                  key="now_h", label_visibility="collapsed"), 1)
        frame = live[live["horizon"] == horizon]
        nxt = live[live["horizon"] == 1].set_index("road_id")
        when = frame["timestamp"].iloc[0]
        if horizon == 1:     # travel time and accident risk are forecast for the next window only
            rows = road_rows(out, frame, when, lambda r: r["pred_congestion"], nxt["accident_risk"],
                             nxt["accident_flag"], nxt["pred_travel_time"])
        else:
            rows = road_rows(out, frame, when, lambda r: r["congestion_from_volume"])
        band_label, extra = "Forecast for", ""
    else:
        pred = out["pred"]
        a, b = right.columns(2)
        day = a.date_input("Day", value=pred["timestamp"].max().date() - pd.Timedelta(days=3), label_visibility="collapsed",
                           min_value=pred["timestamp"].min().date(), max_value=pred["timestamp"].max().date())
        slot = b.select_slider("Window", SLOTS, value="18:00", label_visibility="collapsed")
        when, horizon = pd.Timestamp(f"{day} {slot}"), 1
        frame = pred[pred["timestamp"] == when]
        if frame.empty:
            with band:
                ui.band("No forecast for that window", "Choose a day from 11 April to 31 May.", "Looking back at", f"{when:%H:%M}")
            return
        idx = frame.set_index("road_id")
        rows = road_rows(out, frame, when, lambda r: ui.LEVELS[int(r["pred_congestion"])], idx["accident_risk"],
                         idx["accident_flag"], idx["pred_travel_time"])   # saved forecasts store levels as codes 0-3
        observed = frame[frame["mask_congestion"]]
        volume = frame[frame["mask_volume"]]
        right_levels = int((observed["pred_congestion"] == observed["y_congestion"]).sum())
        error = ((volume["pred_volume"] - volume["y_volume"]).abs() / volume["y_volume"]).mean()
        band_label = "Looking back at"
        extra = (f" The forecast got <b>{right_levels} of {len(observed)}</b> levels right and was "
                 f"<b>{error:.0%}</b> off the recorded count on average.")

    title, detail = headline(rows, when)
    with band:
        ui.band(title, detail + extra, band_label, f"{when:%H:%M}, {when:%a %d %b}")

    total = sum(r["volume"] for r in rows)
    usual_total = sum(r["usual"] for r in rows if r["usual"] is not None)
    change = total / usual_total - 1 if usual_total else 0
    slow = sum(r["level"] in ("Heavy", "Severe") for r in rows)
    figures = [{"v": f"{total:,.0f}", "l": "vehicles on the corridor",
                "d": f"{change:+.0%} on a usual {when:%A} at {when:%H:%M}", "up": change > 0.1},
               {"v": slow, "unit": f"of {len(rows)}", "l": "roads slow",
                "d": f"{sum(r['level'] == 'Severe' for r in rows)} Severe"}]
    if rows[0]["travel"] is not None:
        usual_travel = d.usual(out, "usual_travel", when).sum()
        drive = sum(r["travel"] for r in rows)
        figures.append({"v": f"{drive:.0f}", "unit": "min", "l": "to drive all 25 roads",
                        "d": f"{drive - usual_travel:+.0f} min on usual", "up": drive - usual_travel > 3})
    else:
        figures.append({"v": f"{max(r['vc'] for r in rows):.0%}", "l": "capacity used, busiest road"})

    st.write("")
    map_col, side = st.columns([8, 5], gap="large")
    with map_col:
        event = ui.show(ui.corridor_map(out["segments"], rows, d.wards(), d.street_tiles()), key="now_map",
                        on_select="rerun", selection_mode="points")
        clicked = [p.get("customdata") for p in event.selection.points if p.get("customdata")] if event else []
        if clicked and clicked[0] != st.session_state.get("map_click"):
            st.session_state["map_click"] = st.session_state["road"] = clicked[0]
        ui.html_block(ui.level_key())
        ui.map_note(d.street_tiles())
    with side:
        ui.numbers(figures)
        items, more = needs_attention(rows)
        ui.section("Needs attention", f"{more} more below" if more else None)
        ui.attention(items, "Nothing needs attention. Every road is within its usual range.")
        ui.section("Selected road", "click a dot on the map")
        selected_road(out, rows, when, horizon)

    ui.section("Every road", "North to South. The tick marks full capacity, where Severe starts.")
    ui.road_list(rows)


# ====================================================================== Today
def page_today():
    """FR-14: the whole day on the map, with the corridor total against a usual day."""
    out = d.load(d.version())
    clean, segments, c = out["clean"], out["segments"], ui.colors()
    first, last = clean["date"].min().date(), clean["date"].max().date()
    head = st.container()
    pick, _ = st.columns([1, 3])
    day = pd.Timestamp(pick.date_input("Day", value=last, min_value=first, max_value=last, key="today_day"))
    rows = clean[clean["date"] == day]
    levels = rows.pivot(index="time", columns="road_id", values="congestion_level").reindex(SLOTS).ffill().bfill()
    volumes = rows.pivot(index="time", columns="road_id", values="traffic_volume").reindex(SLOTS).fillna(0)

    totals = out["corridor_total"]
    frame = totals.to_frame("total")
    frame["dow"], frame["time"] = frame.index.day_of_week, frame.index.strftime("%H:%M")
    usual = frame.groupby(["dow", "time"])["total"].median().xs(day.day_of_week, level="dow").reindex(SLOTS)
    today = totals[(totals.index >= day) & (totals.index < day + pd.Timedelta(days=1))]
    peak = today.idxmax()
    slow_windows = rows["congestion_level"].isin(["Heavy", "Severe"]).groupby(rows["time"]).sum().reindex(SLOTS).fillna(0)
    if slow_windows.max() > 0:
        worst_slot = slow_windows.idxmax()
        slow_text = f"Most roads slow at <b>{worst_slot}</b>, {int(slow_windows.max())} of {len(segments)}. "
    else:
        worst_slot = peak.strftime("%H:%M")
        slow_text = "No road was Heavy or Severe all day. "
    with head:
        ui.title(f"{day:%A %d %B}",
                 f"Busiest half-hour <b>{peak:%H:%M}</b> with {today.max():,.0f} vehicles, "
                 f"{today.max() / usual[peak.strftime('%H:%M')] - 1:+.0%} on a usual {day:%A}. "
                 + slow_text + "Press play to watch the day, or drag the slider.", data_through(out))

    left, right = st.columns([8, 5], gap="large")
    with left:
        start = SLOTS.index(worst_slot)
        times = list(levels.index)
        ui.show(ui.day_map(segments, times, [levels.loc[t].to_dict() for t in times],
                           [volumes.loc[t].to_dict() for t in times], d.wards(), d.street_tiles(), start))
        ui.html_block(ui.level_key(patrol=False))
        ui.map_note(d.street_tiles())
    with right:
        ui.section("Whole corridor", "vehicles per 30 min")
        forecast = out["pred"][out["pred"]["timestamp"].dt.normalize() == day].groupby("timestamp")["pred_volume"].sum()
        fig = ui.figure()
        fig.add_trace(go.Scatter(x=[day + pd.Timedelta(minutes=30 * i) for i in range(48)], y=usual.values,
                                 name=f"Usual {day:%A}", line={"color": c["recorded"], "width": 1.4, "dash": "dot"}))
        fig.add_trace(go.Scatter(x=today.index, y=today.values, name="Recorded", line={"color": c["ink"], "width": 2}))
        if len(forecast):
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast, 30 min ahead",
                                     line={"color": c["accent"], "width": 1.4}))
        ui.show(ui.style(fig, height=250).update_xaxes(tickformat="%H:%M"))
        ui.section("Slow roads through the day", "Heavy or Severe")
        fig = ui.figure()
        fig.add_trace(go.Bar(x=[day + pd.Timedelta(minutes=30 * i) for i in range(48)],
                             y=slow_windows.values, marker_color=c["levels"]["Heavy"],
                             hovertemplate="%{x|%H:%M}: %{y} roads<extra></extra>"))
        ui.style(fig, height=170, legend=False, hover="closest").update_xaxes(tickformat="%H:%M")
        ui.show(fig.update_yaxes(range=[0, max(slow_windows.max(), 5) * 1.1], title={"text": "roads"}))

    ui.section("Every road, every half-hour", "recorded congestion")
    day_grid(out, day)


def day_grid(out, day):
    """Recorded congestion for every road and window of one day."""
    clean = out["clean"]
    grid = clean[clean["date"] == day].pivot(index="road_id", columns="time", values="congestion_level")
    grid = grid.reindex(out["segments"].index)
    codes = grid.apply(lambda column: column.map({level: i for i, level in enumerate(ui.LEVELS)}))
    names = [out["segments"].loc[r, "road_name"] for r in grid.index]
    fig = ui.figure()
    fig.add_trace(go.Heatmap(z=codes.values, x=grid.columns, y=names, text=grid.values, showscale=False,
                             colorscale=ui.level_scale(), zmin=-0.5, zmax=3.5, xgap=1, ygap=1,
                             hovertemplate="%{y} at %{x}: %{text}<extra></extra>"))
    ui.style(fig, height=520, legend=False, hover="closest")
    fig.update_xaxes(tickvals=[f"{h:02d}:00" for h in range(0, 24, 3)], ticks="")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont={"size": 11})
    ui.show(fig)


# ====================================================================== Roads
def page_roads():
    """PRD 14.2 and FR-14: one road in detail, every road compared, trends, weather and patterns."""
    out = d.load(d.version())
    ui.title("Roads", "Pick a road for its forecast and history, or compare all 25.", data_through(out))
    one, compare, trends, weather, patterns = st.tabs(["One road", "Compare roads", "Trends", "Weather", "Busy times"])
    with one:
        road_detail(out)
    with compare:
        road_table(out)
    with trends:
        road_trends(out)
    with weather:
        weather_effect(out)
    with patterns:
        busy_times(out)


def road_detail(out):
    """One road: the next two hours with ranges, the last 12 hours, and any past forecast."""
    live, c = out["live"], ui.colors()
    left, right = st.columns([2, 2])
    with left:
        road_id = pick_road(out, "roads_road")
    with right:
        horizon = chosen(st.segmented_control("Ahead", list(HORIZONS), default=1, format_func=HORIZONS.get,
                                              key="roads_h"), 1)
    row = live[(live["road_id"] == road_id) & (live["horizon"] == horizon)].iloc[0]
    usual = d.usual(out, "usual_volume", row["timestamp"]).get(road_id)
    usual_text = f", {row['pred_volume'] / usual - 1:+.0%} on usual" if usual else ""
    ui.html_block(f'<p class="fc-lede">Forecast for <b>{row["timestamp"]:%H:%M}</b> on {row["timestamp"]:%d %B}, '
                  f'from readings up to {row["origin_timestamp"]:%H:%M}.</p>')
    level = row["pred_congestion"] if horizon == 1 else row["congestion_from_volume"]
    items = [{"v": f"{row['pred_volume']:,.0f}", "l": "vehicles",
              "d": f"likely {row['pred_volume_lower']:,.0f} to {row['pred_volume_upper']:,.0f}{usual_text}"},
             {"v": level, "l": "congestion", "d": f"could be {row['congestion_range_90']}"}]
    if horizon == 1:
        items.append({"v": f"{row['pred_travel_time']:.1f}", "unit": "min", "l": "travel time",
                      "d": f"likely {row['pred_travel_time_lower']:.1f} to {row['pred_travel_time_upper']:.1f} min"})
    ui.numbers(items)
    if horizon == 1:
        ui.numbers([{"v": f"{row['accident_risk']:.1%}", "l": "accident risk", "up": bool(row["accident_flag"]),
                     "d": "patrol suggested" if row["accident_flag"] else "below the patrol line"},
                    {"v": f"{row['pred_congestion_confidence']:.0%}", "l": "sure of the congestion level"}])

    recorded = out["clean"][out["clean"]["road_id"] == road_id].set_index("timestamp")["traffic_volume"].tail(24)
    ahead = live[live["road_id"] == road_id].sort_values("timestamp")
    ui.section("Last 12 hours and the next 2", "vehicles per 30 min")
    fig = ui.figure()
    fig.add_trace(go.Scatter(x=recorded.index, y=recorded.values, name="Recorded", line={"color": c["recorded"], "width": 3}))
    fig.add_trace(go.Scatter(x=ahead["timestamp"], y=ahead["pred_volume"], name="Forecast", mode="lines+markers",
                             line={"color": c["accent"], "width": 1.8},
                             error_y={"type": "data", "symmetric": False, "color": c["accent"], "thickness": 1, "width": 3,
                                      "array": ahead["pred_volume_upper"] - ahead["pred_volume"],
                                      "arrayminus": ahead["pred_volume"] - ahead["pred_volume_lower"]}))
    ui.show(ui.style(fig, height=260).update_xaxes(tickformat="%H:%M"))

    ui.section("Check a past forecast", "11 April to 31 May")
    history = out["pred_h"][(out["pred_h"]["road_id"] == road_id) & (out["pred_h"]["horizon"] == horizon)].sort_values("timestamp")
    when = st.select_slider("Window", list(history["timestamp"]), value=history["timestamp"].iloc[-1],
                            format_func=lambda t: pd.Timestamp(t).strftime("%d %b %H:%M"))
    past = history[history["timestamp"] == when].iloc[0]
    actual = f"{past['y_volume']:,.0f}" if past["mask_volume"] else "no reading (sensor offline)"
    st.markdown(f"Forecast **{past['pred_volume']:,.0f}** vehicles, likely {past['pred_volume_lower']:,.0f} to "
                f"{past['pred_volume_upper']:,.0f}. Recorded **{actual}**.")


def road_table(out):
    """FR-14: every road compared on every measure, sortable."""
    obs = out["clean"][out["clean"]["row_observed"]]
    table = obs.groupby("road_id").agg(
        vehicles=("traffic_volume_obs", "mean"), speed=("avg_speed_obs", "mean"),
        travel=("travel_time_obs", "mean"), p95=("travel_time_obs", lambda v: v.quantile(0.95)),
        severe=("congestion_level", lambda v: (v == "Severe").mean() * 100),
        accidents=("accident_count", lambda v: (v > 0).mean() * 100))
    table["buffer"] = (table["p95"] - table["travel"]) / table["travel"] * 100   # extra time to allow on a bad day
    table.insert(0, "segment", out["segments"]["label"])
    table = table.reindex(out["segments"].index).drop(columns="p95")
    st.caption("January to May, recorded windows only. Click a column to sort.")
    bar = st.column_config.ProgressColumn
    st.dataframe(table, hide_index=True, width="stretch", row_height=30, height=30 * (len(table) + 1) + 3, column_config={
        "segment": st.column_config.TextColumn("Road", width="medium"),
        "vehicles": bar("Vehicles per 30 min", format="%.0f", min_value=0, max_value=float(table["vehicles"].max())),
        "speed": bar("Speed, km/h", format="%.1f", min_value=0, max_value=float(table["speed"].max())),
        "travel": st.column_config.NumberColumn("Travel, min", format="%.2f"),
        "severe": bar("Severe, % of windows", format="%.1f%%", min_value=0, max_value=float(table["severe"].max())),
        "accidents": bar("Accidents, % of windows", format="%.2f%%", min_value=0, max_value=float(table["accidents"].max())),
        "buffer": bar("Extra time to allow, %", format="%.0f%%", min_value=0, max_value=float(table["buffer"].max()),
                      help="95th-percentile travel time compared with the mean. Lower is more reliable."),
    })


def road_trends(out):
    """FR-14: volume and speed over time, and the weekday profile."""
    clean, labels, c = out["clean"], out["segments"]["label"], ui.colors()
    left, right = st.columns([3, 1])
    default = [st.session_state.get("road", labels.index[0])]
    chosen_ids = left.multiselect("Roads, up to three", list(labels.index), default=default, max_selections=3,
                                  format_func=labels.get, key="trend_roads")
    daily = chosen(right.segmented_control("Resolution", ["Daily", "30 min"], default="Daily", key="t_res"), "Daily") == "Daily"
    if not chosen_ids:
        st.info("Choose at least one road.")
        return
    cols = st.columns(2, gap="large")
    for column, (measure, name, unit) in zip(cols, [("traffic_volume", "Vehicles", "per 30 min"), ("avg_speed", "Speed", "km/h")]):
        with column:
            ui.section(name, unit + (", daily mean" if daily else ", last 14 days"))
            fig = ui.figure()
            for i, road_id in enumerate(chosen_ids):
                rows = clean[clean["road_id"] == road_id]
                series = (rows.groupby("date")[measure].mean() if daily else
                          rows[rows["timestamp"] > rows["timestamp"].max() - pd.Timedelta(days=14)].set_index("timestamp")[measure])
                fig.add_trace(go.Scatter(x=series.index, y=series.values, name=labels[road_id], line={"color": c["series"][i], "width": 1.6}))
            ui.show(ui.style(fig, height=260))
    ui.section("A typical weekday", "mean vehicles per window, Monday to Friday")
    weekday = clean[clean["timestamp"].dt.day_of_week < 5]
    fig = ui.figure()
    for i, road_id in enumerate(chosen_ids):
        profile = weekday[weekday["road_id"] == road_id].groupby("time")["traffic_volume"].mean()
        fig.add_trace(go.Scatter(x=profile.index, y=profile.values, name=labels[road_id], line={"color": c["series"][i], "width": 1.6}))
    ui.show(ui.style(fig, height=240).update_xaxes(tickvals=[f"{h:02d}:00" for h in range(0, 24, 3)]))


def weather_effect(out):
    """FR-16: weather against speed, volume, accidents and forecast error."""
    effect, c = out["weather"], ui.colors()
    pred = out["pred"][out["pred"]["mask_volume"]].merge(out["clean"][["road_id", "timestamp", "weather_condition"]])
    pred["error"] = (pred["pred_volume"] - pred["y_volume"]).abs() / pred["y_volume"] * 100
    error = pred.groupby("weather_condition")["error"].agg(["mean", "size"]).reindex(effect["weather_condition"]).dropna()
    clear = effect.set_index("weather_condition")
    if {"Clear", "Rain"} <= set(clear.index):
        drop = 1 - clear.loc["Rain", "mean_speed_kmh"] / clear.loc["Clear", "mean_speed_kmh"]
        st.caption(f"Rain cuts average speed by {drop:.0%} against clear weather. Recorded windows, grouped by the weather at their station.")
    cols = st.columns(4, gap="medium")
    panels = [("Speed, km/h", effect["weather_condition"], effect["mean_speed_kmh"], "{:.1f}", None),
              ("Vehicles per 30 min", effect["weather_condition"], effect["mean_volume"], "{:,.0f}", None),
              ("Accidents, % of windows", effect["weather_condition"], effect["accident_rate_pct"], "{:.2f}", None),
              ("Forecast error, %", error.index, error["mean"], "{:.1f}", c["series"][1])]
    for column, (name, x, y, fmt, colour) in zip(cols, panels):
        with column:
            ui.section(name)
            ui.bars(list(x), list(y), fmt, colour=colour)
    ui.section("Visibility and speed", "6,000 recorded windows")
    sample = out["clean"][out["clean"]["row_observed"] & out["clean"]["avg_speed_obs"].notna()].sample(6000, random_state=42)
    fig = ui.figure()
    fig.add_trace(go.Scattergl(x=sample["visibility"], y=sample["avg_speed_obs"], mode="markers",
                               marker={"color": c["accent"], "size": 5, "opacity": 0.25},
                               hovertemplate="%{x:,.0f} m, %{y:.1f} km/h<extra></extra>"))
    ui.show(ui.style(fig, height=260, x="Visibility, m", y="Speed, km/h", legend=False, hover="closest"))


def busy_times(out):
    """FR-14: share of windows Heavy or Severe over a date range, every road and half-hour."""
    clean = out["clean"]
    first, last = clean["date"].min().date(), clean["date"].max().date()
    dates = st.date_input("Dates", value=(last - pd.Timedelta(days=13), last), min_value=first, max_value=last, key="busy_d")
    start, end = (dates if len(dates) == 2 else (dates[0], dates[0]))
    span = clean[(clean["date"].dt.date >= start) & (clean["date"].dt.date <= end)]
    share = span.assign(busy=span["congestion_level"].isin(["Heavy", "Severe"])).pivot_table(
        index="road_id", columns="time", values="busy", aggfunc="mean").reindex(out["segments"].index) * 100
    names = [out["segments"].loc[r, "road_name"] for r in share.index]
    st.caption(f"How often each road was Heavy or Severe, {start:%d %b} to {end:%d %b}.")
    fig = ui.figure()
    fig.add_trace(go.Heatmap(z=share.values, x=share.columns, y=names, colorscale=ui.ramp(), xgap=1, ygap=1,
                             colorbar={"thickness": 8, "len": 0.5, "outlinewidth": 0, "title": {"text": "%"}},
                             hovertemplate="%{y} at %{x}: %{z:.0f}%<extra></extra>"))
    ui.style(fig, height=560, legend=False, hover="closest")
    fig.update_xaxes(tickvals=[f"{h:02d}:00" for h in range(0, 24, 2)], ticks="")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont={"size": 11})
    ui.show(fig)


# ====================================================================== Accuracy
TARGETS = {"volume": ("Volume", "rmse", ["rmse", "mae", "mape", "r2"]),
           "travel_time": ("Travel time", "rmse", ["rmse", "mae", "mape", "r2"]),
           "congestion": ("Congestion", "macro_f1", ["macro_f1", "precision_macro", "recall_macro", "accuracy"]),
           "accident": ("Accident risk", "roc_auc", ["roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"])}


def page_accuracy():
    """FR-15 and FR-11: how well the forecasts did on weeks the models never saw."""
    out = d.load(d.version())
    board, goal, pred = out["board"], d.CONFIG["success_targets"], out["pred"]
    test = board[(board["split"] == "test") & (board["horizon"] == 1)]
    pick = {t: test[(test["target"] == t) & (test["selected"] == 1)].iloc[0] for t in TARGETS
            if ((test["target"] == t) & (test["selected"] == 1)).any()}
    criteria = []
    if "volume" in pick:
        criteria.append(("Volume error (MAPE) at most 12%", f"{pick['volume']['mape']:.2f}% with {pick['volume']['model']}",
                         pick["volume"]["mape"] <= goal["volume_mape_max"]))
    if "congestion" in pick:
        criteria.append(("Congestion macro-F1 at least 0.80", f"{pick['congestion']['macro_f1']:.3f} with {pick['congestion']['model']}",
                         pick["congestion"]["macro_f1"] >= goal["congestion_macro_f1_min"]))
    if "accident" in pick:
        criteria.append(("Accident ROC-AUC at least 0.75", f"{pick['accident']['roc_auc']:.3f}, PR-AUC {pick['accident']['pr_auc']:.3f}, "
                         f"with {pick['accident']['model']}", pick["accident"]["roc_auc"] >= goal["accident_roc_auc_min"]))
    deep = board[(board["split"] == "val") & (board["target"] == "volume") & (board["family"] == "deep") & (board["horizon"] == 1)]
    if len(deep) and "volume" in pick:
        name = deep.sort_values("rmse").iloc[0]["model"]
        deep_rmse = test[(test["target"] == "volume") & (test["model"] == name)]["rmse"].iloc[0]
        criteria.append(("Sequence model beats the best classical RMSE",
                         f"{name} {deep_rmse:.2f} against {pick['volume']['model']} {pick['volume']['rmse']:.2f}",
                         deep_rmse < pick["volume"]["rmse"]))

    scored = pred[(pred["split"] == "test") & pred["mask_volume"]]
    inside = ((scored["y_volume"] >= scored["pred_volume_lower"]) & (scored["y_volume"] <= scored["pred_volume_upper"])).mean()
    levels = pred[(pred["split"] == "test") & pred["mask_congestion"]]
    level_right = (levels["pred_congestion"] == levels["y_congestion"]).mean()
    ui.title("How accurate is this?",
             f"Scored on 6 to 31 May, weeks the models never saw while training. "
             f"<b>{sum(ok for *_, ok in criteria)} of {len(criteria)}</b> targets met.", data_through(out))
    figures = []
    if "volume" in pick:
        figures.append({"v": f"{pick['volume']['mape']:.1f}%", "l": "typical miss on the vehicle count", "d": "target: 12% or less"})
    figures += [{"v": f"{level_right:.0%}", "l": "congestion levels right", "d": "next 30 minutes"},
                {"v": f"{inside:.0%}", "l": "of counts inside the likely range", "d": "the range is built to hold 90%"}]
    ui.numbers(figures)

    targets, models, forecasts, confidence, inputs = st.tabs(
        ["Targets", "Compare models", "Forecast vs recorded", "Likely ranges", "What the models use"])
    with targets:
        ui.section("Targets from the requirements")
        ui.checks(criteria)
        gd = d.CONFIG["paths"]["figures_dir"] / "gd_cost_curve.png"
        if gd.exists():
            with st.expander("Linear regression by gradient descent, written in NumPy"):
                st.image(str(gd), width=520)
    with models:
        model_scores(out, board, test)
    with forecasts:
        forecast_lines(out)
    with confidence:
        likely_ranges(out, scored)
    with inputs:
        importance(out)


def model_scores(out, board, test):
    """Test metrics per model and target, confusion matrix, horizons and training curves."""
    c = ui.colors()
    target = chosen(st.segmented_control("Target", list(TARGETS), default="volume", format_func=lambda t: TARGETS[t][0],
                                         key="p_t"), "volume")
    name, metric, columns = TARGETS[target]
    rows = test[test["target"] == target].sort_values(metric, ascending=metric != "rmse")
    left, right = st.columns([5, 7], gap="large")
    with left:
        ui.section(f"{name}, test {metric.upper()}", "lower is better" if metric == "rmse" else "higher is better")
        colours = [c["accent"] if f == "classical" else c["series"][1] if f == "deep" else c["recorded"] for f in rows["family"]]
        names = [("★ " if s else "") + m for m, s in zip(rows["model"], rows["selected"])]
        ui.bars(names[::-1], list(rows[metric])[::-1], "{:.3f}", height=34 * len(rows) + 40, colour=colours[::-1], horizontal=True)
        st.caption("★ model in use. Grey bars are simple baselines, orange bars sequence models.")
    with right:
        ui.section("All metrics")
        st.dataframe(rows[["model", "family"] + [m for m in columns if m in rows] + ["n"]].round(4), hide_index=True,
                     width="stretch")
        if target == "congestion":
            names = list(rows["model"])
            in_use = [m for m, s in zip(rows["model"], rows["selected"]) if s]
            start = names.index(in_use[0]) if in_use else 0     # open on the model in use
            model = st.selectbox("Confusion matrix for", names, index=start)
            matrix = np.array(json.loads(rows[rows["model"] == model]["confusion_matrix"].iloc[0]))
            fig = ui.figure()
            fig.add_trace(go.Heatmap(z=matrix, x=ui.LEVELS, y=ui.LEVELS, colorscale=ui.ramp(), showscale=False, text=matrix,
                                     texttemplate="%{text:,}", xgap=2, ygap=2,
                                     hovertemplate="Actual %{y}, forecast %{x}: %{z:,}<extra></extra>"))
            ui.show(ui.style(fig, height=300, x="Forecast", y="Actual", legend=False, hover="closest")
                    .update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)"))

    left, right = st.columns(2, gap="large")
    horizon_rows = board[(board["split"] == "test") & (board["target"] == "volume")
                         & board["model"].isin(["XGBoost", "Baseline: last observed (t0)"])]
    with left:
        ui.section("Volume RMSE by how far ahead")
        fig = ui.figure()
        for colour, model in [(c["accent"], "XGBoost"), (c["recorded"], "Baseline: last observed (t0)")]:
            r = horizon_rows[horizon_rows["model"] == model].sort_values("horizon")
            fig.add_trace(go.Scatter(x=[HORIZONS[h] for h in r["horizon"]], y=r["rmse"], name=model, mode="lines+markers",
                                     line={"color": colour, "width": 1.8}))
        ui.show(ui.style(fig, height=240, y="RMSE, vehicles"))
    with right:
        ui.section("Sequence model training", "loss per epoch, solid train, dotted validation")
        fig = ui.figure()
        for i, (cell, r) in enumerate(out["dl_history"].groupby("cell", sort=False) if len(out["dl_history"]) else []):
            for split, dash in [("train_loss", "solid"), ("val_loss", "dot")]:
                fig.add_trace(go.Scatter(x=r["epoch"], y=r[split], name=f"{cell.upper()} {split[:-5]}",
                                         line={"color": c["series"][i], "dash": dash, "width": 1.6}))
        ui.show(ui.style(fig, height=240, y="MSE, scaled"))


def road_and_days(out, key):
    """Road and day-range controls for the accuracy tabs."""
    pred = out["pred"]
    last = pred["timestamp"].max().date()
    left, right = st.columns([2, 2])
    with left:
        road_id = pick_road(out, f"{key}_s")
    days = right.date_input("Days", value=(last - pd.Timedelta(days=2), last), min_value=pred["timestamp"].min().date(),
                            max_value=last, key=f"{key}_d")
    start, end = (days if len(days) == 2 else (days[0], days[0]))
    window = pred[(pred["road_id"] == road_id) & (pred["timestamp"].dt.date >= start) & (pred["timestamp"].dt.date <= end)]
    return road_id, start, end, window


def forecast_lines(out):
    """FR-15: forecasts drawn over what the sensors recorded."""
    c = ui.colors()
    road_id, start, end, window = road_and_days(out, "f")
    names = [col.replace("pred_volume__", "") for col in out["pred"] if col.startswith("pred_volume__")]
    models = st.multiselect("Models, up to three", names, max_selections=3,
                            default=[n for n in ["XGBoost", "GRU", "Baseline: last observed (t0)"] if n in names])
    left, right = st.columns(2, gap="large")
    with left:
        ui.section("30 minutes ahead, by model")
        fig = ui.figure()
        fig.add_trace(go.Scatter(x=window["timestamp"], y=window["y_volume"].where(window["mask_volume"]), name="Recorded",
                                 line={"color": c["recorded"], "width": 3}))
        for i, name in enumerate(models):
            fig.add_trace(go.Scatter(x=window["timestamp"], y=window[f"pred_volume__{name}"], name=name,
                                     line={"color": c["series"][i], "width": 1.4}))
        ui.show(ui.style(fig, height=300, y="Vehicles per 30 min"))
    with right:
        ui.section("Further ahead, best model for each")
        ph = out["pred_h"]
        ph = ph[(ph["road_id"] == road_id) & (ph["timestamp"].dt.date >= start) & (ph["timestamp"].dt.date <= end)]
        fig = ui.figure()
        base = ph[ph["horizon"] == 1]
        fig.add_trace(go.Scatter(x=base["timestamp"], y=base["y_volume"].where(base["mask_volume"]), name="Recorded",
                                 line={"color": c["recorded"], "width": 3}))
        for i, (h, r) in enumerate(ph.groupby("horizon")):
            fig.add_trace(go.Scatter(x=r["timestamp"], y=r["pred_volume"], name=HORIZONS[h], line={"color": c["series"][i], "width": 1.4}))
        ui.show(ui.style(fig, height=300, y="Vehicles per 30 min"))


def likely_ranges(out, scored):
    """FR-11: how often the 90% ranges contain the recorded value."""
    c = ui.colors()
    road_id, start, end, window = road_and_days(out, "c")
    st.caption(f"Typical range width {(scored['pred_volume_upper'] - scored['pred_volume_lower']).median():.0f} vehicles. "
               f"Typical confidence in the congestion level {scored['pred_congestion_confidence'].median():.0%}.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section("Forecast with its likely range", "90% range")
        rgb = tuple(int(c["accent"][i:i + 2], 16) for i in (1, 3, 5))
        fig = ui.figure()
        fig.add_trace(go.Scatter(x=window["timestamp"], y=window["pred_volume_upper"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=window["timestamp"], y=window["pred_volume_lower"], fill="tonexty", line={"width": 0},
                                 fillcolor=f"rgba{rgb + (0.16,)}", name="Likely range"))
        fig.add_trace(go.Scatter(x=window["timestamp"], y=window["pred_volume"], name="Forecast", line={"color": c["accent"], "width": 1.6}))
        fig.add_trace(go.Scatter(x=window["timestamp"], y=window["y_volume"].where(window["mask_volume"]), name="Recorded",
                                 mode="markers", marker={"color": c["ink"], "size": 4}))
        ui.show(ui.style(fig, height=300, y="Vehicles per 30 min"))
    with right:
        ui.section("Congestion: share right, by confidence")
        bins = pd.cut(scored["pred_congestion_confidence"], [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        calibration = (scored["pred_congestion"] == scored["y_congestion"]).groupby(bins, observed=True).mean() * 100
        ui.bars([f"{b.left:.0%}–{b.right:.0%}" for b in calibration.index], list(calibration), "{:.0f}%", height=300,
                y_title="% right")


def importance(out):
    """FR-15: what the tree models rely on."""
    imp = out["importance"]
    left, right = st.columns(2)
    targets = [t for t in TARGETS if t in set(imp["target"])]
    target = chosen(left.segmented_control("Target", targets, default=targets[0], format_func=lambda t: TARGETS[t][0], key="i_t"), targets[0])
    model = chosen(right.segmented_control("Model", sorted(imp["model"].unique()), default="XGBoost", key="i_m"), "XGBoost")
    rows = imp[(imp["target"] == target) & (imp["model"] == model)].nlargest(20, "importance").iloc[::-1]
    ui.section(f"Top 20 inputs for {TARGETS[target][0].lower()}, {model}", "names ending in _t0 are the latest readings")
    ui.bars(list(rows["feature"]), list(rows["importance"] * 100), "{:.1f}%", height=560, horizontal=True)


# ====================================================================== Data
def page_data():
    """FR-17, FR-18 and PRD 14.2: upload new files, export a report, check data quality, retrain."""
    out = d.load(d.version())
    ui.title("Data", "Bring in new sensor files, download a summary, or see what the last load fixed.", data_through(out))
    upload, report, quality = st.tabs(["Upload new data", "Download a report", "Data checks"])
    with upload:
        upload_files()
    with report:
        export_report(out)
    with quality:
        data_quality(out)


def upload_files():
    """FR-17: check new files and forecast with the saved models."""
    st.caption("Check a new sensor log and weather file, then forecast the next two hours with the models in use. "
               "Nothing is retrained.")
    form, _ = st.columns([3, 2])
    with form, st.form("upload", border=False):
        traffic = st.file_uploader("Traffic sensor log", type="csv", help="Same columns as traffic_sensor_log.csv")
        weather = st.file_uploader("Weather observations", type="csv", help="Same columns as weather_observations.csv")
        calendar = st.file_uploader("Calendar events (optional)", type="csv",
                                    help="Leave empty to use the calendar already loaded.")
        submitted = st.form_submit_button("Check files and forecast", type="primary")
    if not submitted:
        return
    if traffic is None or weather is None:
        st.error("Add both the traffic sensor log and the weather observations.")
        return
    from src.clean import run_cleaning
    from src.predict import predict_next_windows

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "raw").mkdir()
        (work / "raw" / "traffic_sensor_log.csv").write_bytes(traffic.getvalue())
        (work / "raw" / "weather_observations.csv").write_bytes(weather.getvalue())
        (work / "raw" / "calendar_events.csv").write_bytes(
            calendar.getvalue() if calendar else (d.CONFIG["paths"]["raw_dir"] / "calendar_events.csv").read_bytes())
        config = d.load_config()
        for key in ["interim_dir", "processed_dir", "reports_dir", "figures_dir"]:
            config["paths"][key] = work / key
        try:
            with st.spinner("Checking and cleaning the files"):
                clean, log = run_cleaning(work / "raw", config)
            with st.spinner("Forecasting the next four windows"):
                forecast = predict_next_windows(clean, config)
        except Exception as error:  # shown to the operator, the files stay on their machine
            st.error(f"These files could not be read: {error}. Check the column names against the data dictionary.")
            return
    forecast["timestamp"] = forecast["timestamp"].dt.tz_localize(None)
    st.success(f"{len(clean):,} road-windows checked. Forecast ready.")
    ui.section("Forecast, next four windows")
    st.dataframe(forecast.drop(columns=["origin_timestamp", "road_capacity"]).round(3), hide_index=True, width="stretch")
    st.download_button("Download forecast as CSV", forecast.to_csv(index=False), "flowcast_forecast.csv", "text/csv")
    ui.section("Cleaning log")
    st.dataframe(pd.DataFrame(log.entries), hide_index=True, width="stretch")


def export_report(out):
    """FR-18: summary of any period, downloadable."""
    clean, pred, labels = out["clean"], out["pred"], out["segments"]["label"]
    first, last = clean["date"].min().date(), clean["date"].max().date()
    dates = st.date_input("Period", value=(last - pd.Timedelta(days=6), last), min_value=first, max_value=last, key="rep_d")
    start, end = (dates if len(dates) == 2 else (dates[0], dates[0]))
    span = clean[(clean["date"].dt.date >= start) & (clean["date"].dt.date <= end) & clean["row_observed"]]
    by = span.groupby("road_id").agg(vehicles=("traffic_volume_obs", "mean"), speed=("avg_speed_obs", "mean"),
                                     severe=("congestion_level", lambda v: (v == "Severe").mean() * 100),
                                     accidents=("accident_count", "sum")).reindex(out["segments"].index)
    by.insert(0, "segment", labels)
    busiest, worst = by["vehicles"].idxmax(), by["severe"].idxmax()
    ui.html_block(f'<p class="fc-lede">{start:%d %B} to {end:%d %B}. Busiest: <b>{ui.esc(labels[busiest])}</b>. '
                  f'Most time in Severe: <b>{ui.esc(labels[worst])}</b>.</p>')
    ui.numbers([{"v": f"{span['traffic_volume_obs'].mean():,.0f}", "l": "vehicles per 30 min, mean"},
                {"v": f"{span['avg_speed_obs'].mean():.1f}", "unit": "km/h", "l": "mean speed",
                 "d": f"{(span['congestion_level'] == 'Severe').mean():.1%} of windows Severe"},
                {"v": int(span["accident_count"].sum()), "l": "accidents recorded"}])
    window = pred[(pred["timestamp"].dt.date >= start) & (pred["timestamp"].dt.date <= end) & pred["mask_volume"]]
    lines = [f"# FlowCast: {start:%d %b %Y} to {end:%d %b %Y}", "",
             f"- Recorded segment-windows: {len(span):,}",
             f"- Mean vehicles per 30 min: {span['traffic_volume_obs'].mean():.0f}",
             f"- Mean speed: {span['avg_speed_obs'].mean():.1f} km/h",
             f"- Severe windows: {(span['congestion_level'] == 'Severe').mean():.1%}",
             f"- Accidents recorded: {int(span['accident_count'].sum())}",
             f"- Busiest segment: {labels[busiest]}", f"- Most time in Severe: {labels[worst]}"]
    if len(window):
        lines.append(f"- Volume forecast error (MAPE): {((window['pred_volume'] - window['y_volume']).abs() / window['y_volume']).mean():.2%}")
    a, b, _ = st.columns([1, 1, 3])
    a.download_button("Download summary", "\n".join(lines), f"flowcast_{start}_{end}.md", "text/markdown", type="primary")
    b.download_button("Download table as CSV", by.to_csv(), f"flowcast_{start}_{end}.csv", "text/csv")
    ui.section("By road")
    st.dataframe(by.round(2), hide_index=True, width="stretch", row_height=30, height=30 * (len(by) + 1) + 3)


def training_status():
    """PID of a running pipeline, or None (stale lock files are removed)."""
    if not LOCK.exists():
        return None
    try:
        pid = int(LOCK.read_text() or 0)
        os.kill(pid, 0)
        return pid
    except (ProcessLookupError, ValueError, PermissionError):
        LOCK.unlink(missing_ok=True)
        return None


def data_quality(out):
    """Data-quality log, rows set aside, and a background retraining run (PRD 14.2)."""
    st.caption(f"{len(out['dq'])} checks ran on the last load. Rows that failed a check are kept in "
               "data/interim/quarantine.csv, never dropped.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section("Cleaning log")
        st.dataframe(out["dq"], hide_index=True, width="stretch", height=520)
    with right:
        quarantine = d.CONFIG["paths"]["interim_dir"] / "quarantine.csv"
        if quarantine.exists():
            ui.section("Rows set aside, by reason")
            counts = pd.read_csv(quarantine, usecols=["quarantine_reason"])["quarantine_reason"].value_counts()
            st.dataframe(counts.rename("rows"), width="stretch")
        ui.section("Retrain the models", "about 30 minutes, runs in the background")
        log = d.REPORTS / "pipeline_run.log"
        if training_status():
            lines = log.read_text().strip().splitlines() if log.exists() else []
            st.info(f"Training is running. Latest step: {lines[-1][-160:] if lines else 'starting'}")
            st.button("Refresh status")
            return
        raw = st.text_input("Raw data folder", str(d.CONFIG["paths"]["raw_dir"]),
                            help="Must contain traffic_sensor_log.csv, weather_observations.csv and calendar_events.csv.")
        if st.button("Start training"):
            missing = [name for name in d.CONFIG["files"].values() if not (Path(raw) / name).exists()]
            if missing:
                st.error(f"Missing in that folder: {', '.join(missing)}.")
                return
            with open(log, "w", encoding="utf-8") as handle:
                process = subprocess.Popen([sys.executable, "-m", "src.pipeline", "--data-dir", raw], cwd=d.ROOT,
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
            LOCK.write_text(str(process.pid))
            st.rerun()


# ====================================================================== app shell
def main():
    """Five pages along the top. Each has its own address, so any page can be bookmarked."""
    ui.setup()
    if not d.ready():
        ui.title("FlowCast", "No forecasts yet. Put the three raw files in data/raw and run "
                 "<b>python -m src.pipeline --data-dir data/raw</b>, then reload this page.")
        return
    page = st.Page
    PAGES.update({
        "now": page(page_now, title="Now", icon=":material/near_me:", default=True),
        "today": page(page_today, title="Today", icon=":material/schedule:", url_path="today"),
        "roads": page(page_roads, title="Roads", icon=":material/route:", url_path="roads"),
        "accuracy": page(page_accuracy, title="Accuracy", icon=":material/target:", url_path="accuracy"),
        "data": page(page_data, title="Data", icon=":material/database:", url_path="data"),
    })
    st.navigation(list(PAGES.values()), position="top").run()


main()
