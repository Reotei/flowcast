"""M4 — EDA & data-quality report (FR-06, PRD §18).

Writes figures to reports/figures/, small analytics tables to data/processed/ (read by the
dashboard) and reports/data_quality.md with every cleaning rule and its count.
"""

import numpy as np
import pandas as pd

from src.utils import DataQualityLog, get_logger
from src.viz_style import (CONGESTION_COLORS, DIVERGING_NEGATIVE, DIVERGING_POSITIVE, SERIES,
                           TEXT_SECONDARY, apply_matplotlib_style)


def _save(figure, path):
    """Save and close a matplotlib figure."""
    figure.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(figure)


def daily_profile_figure(clean, figures_dir):
    """Mean observed volume by time of day, weekday vs weekend (two series, direct labels)."""
    plt = apply_matplotlib_style()
    observed = clean.loc[clean["volume_target_observed"]].copy()
    observed["is_weekend"] = observed["timestamp"].dt.day_of_week >= 5
    profile = observed.groupby(["is_weekend", "time"])["traffic_volume"].mean().unstack(0)
    figure, axis = plt.subplots(figsize=(9, 4))
    for column_number, (is_weekend, label) in enumerate([(False, "Weekday"), (True, "Weekend")]):
        values = profile[is_weekend]
        axis.plot(range(len(values)), values.to_numpy(), color=SERIES[column_number], label=label)
    tick_positions = list(range(0, 48, 6))
    axis.set_xticks(tick_positions)
    axis.set_xticklabels([profile.index[position] for position in tick_positions])
    axis.set_title("Average traffic volume by time of day")
    axis.set_ylabel("Vehicles per 30 min")
    axis.legend(loc="upper left")
    _save(figure, figures_dir / "eda_daily_profile.png")
    return profile


def congestion_balance_figure(clean, config, figures_dir):
    """Class counts for the congestion target (shows the imbalance)."""
    plt = apply_matplotlib_style()
    order = config["congestion_order"]
    usable = clean.loc[clean["congestion_source"] != "derived_imputed_volume"]
    counts = usable["congestion_level"].value_counts().reindex(order).fillna(0)
    figure, axis = plt.subplots(figsize=(7, 3.6))
    bars = axis.bar(order, counts.to_numpy(), color=[CONGESTION_COLORS[label] for label in order], width=0.6)
    for bar, count in zip(bars, counts.to_numpy()):
        axis.annotate(f"{int(count):,} ({count / counts.sum():.1%})", (bar.get_x() + bar.get_width() / 2, count),
                      xytext=(0, 3), textcoords="offset points", ha="center", color=TEXT_SECONDARY, fontsize=9)
    axis.set_title("Congestion level class balance")
    axis.set_ylabel("Windows")
    _save(figure, figures_dir / "eda_congestion_balance.png")
    return counts


def accident_rate_figure(clean, figures_dir):
    """Accident rate (% of windows with >= 1 accident) by hour of day."""
    plt = apply_matplotlib_style()
    known = clean.loc[clean["accident_observed"]].copy()
    known["hour"] = known["timestamp"].dt.hour
    rate = (known["accident_count"] > 0).groupby(known["hour"]).mean() * 100
    figure, axis = plt.subplots(figsize=(9, 3.6))
    axis.bar(rate.index, rate.to_numpy(), color=SERIES[0], width=0.7)
    axis.set_title("Accident rate by hour of day")
    axis.set_xlabel("Hour")
    axis.set_ylabel("% of windows with an accident")
    _save(figure, figures_dir / "eda_accident_rate_by_hour.png")
    return rate


def weather_effect_table(clean):
    """Mean speed, volume and accident rate by weather condition (also used by the dashboard)."""
    observed = clean.loc[clean["row_observed"]]
    rows = []
    for condition in ["Clear", "Cloudy", "Rain", "Fog"]:
        subset = observed.loc[observed["weather_condition"] == condition]
        if len(subset) == 0:
            continue
        rows.append({
            "weather_condition": condition,
            "windows": len(subset),
            "mean_speed_kmh": subset["avg_speed_obs"].mean(),
            "mean_volume": subset["traffic_volume_obs"].mean(),
            "accident_rate_pct": (subset["accident_count"] > 0).mean() * 100,
        })
    return pd.DataFrame(rows)


def weather_figure(weather_table, figures_dir):
    """Bar chart of mean speed by weather condition."""
    plt = apply_matplotlib_style()
    figure, axis = plt.subplots(figsize=(7, 3.6))
    axis.bar(weather_table["weather_condition"], weather_table["mean_speed_kmh"], color=SERIES[0], width=0.6)
    for x_position, speed in enumerate(weather_table["mean_speed_kmh"]):
        axis.annotate(f"{speed:.1f}", (x_position, speed), xytext=(0, 3), textcoords="offset points",
                      ha="center", color=TEXT_SECONDARY, fontsize=9)
    axis.set_title("Mean speed by weather condition")
    axis.set_ylabel("km/h")
    _save(figure, figures_dir / "eda_weather_speed.png")


def correlation_figure(features_h1, figures_dir, top_n=15):
    """Pearson correlation of each feature with next-window volume (train rows only).

    r = cov(x, y) / (std(x) * std(y)); bars coloured by sign (diverging poles).
    """
    plt = apply_matplotlib_style()
    train = features_h1.loc[(features_h1["split"] == "train") & features_h1["mask_volume"]]
    candidate_columns = []
    for column in train.columns:
        if column.startswith("seg_") or column.startswith("y_") or column.startswith("mask_"):
            continue
        if train[column].dtype.kind in "fiub" and column not in ["horizon"]:
            candidate_columns.append(column)
    correlations = {}
    for column in candidate_columns:
        if train[column].std() > 0:
            correlations[column] = np.corrcoef(train[column].astype(float), train["y_volume"].astype(float))[0, 1]
    correlation_series = pd.Series(correlations).dropna()
    top = correlation_series.reindex(correlation_series.abs().sort_values(ascending=False).index).head(top_n)
    top = top.iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = [DIVERGING_POSITIVE if value >= 0 else DIVERGING_NEGATIVE for value in top.to_numpy()]
    axis.barh(top.index, top.to_numpy(), color=colors, height=0.6)
    axis.axvline(0, color="#c3c2b7", linewidth=1)
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.set_title("Top feature correlations with next-window volume (train)")
    axis.set_xlabel("Pearson r")
    _save(figure, figures_dir / "eda_feature_correlation.png")
    return correlation_series.sort_values(key=lambda values: values.abs(), ascending=False)


def missingness_figure(clean, figures_dir):
    """Absent windows per segment (where the sensor sent nothing)."""
    plt = apply_matplotlib_style()
    absent = (~clean["row_observed"]).groupby(clean["road_id"]).sum()
    figure, axis = plt.subplots(figsize=(10, 3.6))
    axis.bar(absent.index, absent.to_numpy(), color=SERIES[0], width=0.7)
    axis.set_title("Absent 30-min windows per segment (added by grid reindex)")
    axis.set_ylabel("Windows")
    axis.tick_params(axis="x", rotation=90)
    _save(figure, figures_dir / "eda_missing_windows.png")
    return absent


def summary_statistics(clean):
    """Descriptive statistics (mean, median, std) of the main measurements, observed rows only."""
    rows = []
    for column in ["traffic_volume", "avg_speed", "occupancy", "travel_time"]:
        values = clean[f"{column}_obs"].dropna()
        rows.append({"column": column, "count": len(values), "mean": values.mean(), "median": values.median(),
                     "std": values.std(), "min": values.min(), "max": values.max()})
    return pd.DataFrame(rows)


def write_data_quality_report(clean, dq_log, config, stats, weather_table, correlations, data_label):
    """Write reports/data_quality.md."""
    reports_dir = config["paths"]["reports_dir"]
    lines = [
        "# FlowCast — Data-Quality Report",
        "",
        f"**Data source:** {data_label}",
        "",
        "Generated by `src/eda.py` from the cleaning log. Every rule that touched the data is listed with",
        "its count. Bad rows are quarantined in `data/interim/quarantine.csv`; nothing is dropped silently.",
        "",
        "## 1. Cleaning log",
        "",
        dq_log.to_markdown_table(),
        "",
        "## 2. Final table",
        "",
        f"- Rows: {len(clean):,} (= {clean['road_id'].nunique()} segments x "
        f"{clean['timestamp'].nunique():,} windows)",
        f"- Rows that existed in the raw file: {int(clean['row_observed'].sum()):,}; added by reindex: "
        f"{int((~clean['row_observed']).sum()):,}",
        f"- Volume source: " + ", ".join(f"{key} {value:,}" for key, value in clean["volume_source"].value_counts().items()),
        f"- Congestion source: " + ", ".join(f"{key} {value:,}" for key, value in clean["congestion_source"].value_counts().items()),
        f"- Split (by date): " + ", ".join(f"{key} {value:,}" for key, value in clean["split"].value_counts().items()),
        "",
        "## 3. Decisions to defend in review",
        "",
        "- **Dedup first.** Duplicates are removed on road_id + date + time before any other step, keeping the most complete record.",
        "- **Full grid before lags.** Every segment is reindexed to 48 windows x 151 days, so a lag of 1 always means the previous 30 minutes.",
        "- **Faults -> NaN -> impute.** Negative volume, speed > 200 km/h and occupancy > 100 % are physically impossible and treated as missing.",
        "- **Repair from vehicle_count.** vehicle_count equals traffic_volume in ~99.9 % of rows, so missing/faulty volume is repaired from it. Repaired rows are still excluded from test metrics.",
        "- **Imputation.** Gaps of <= 2 windows use time interpolation within the segment; longer gaps use the segment x time-window median computed on training dates only.",
        "- **Interpolation vs leakage (PRD conflict).** PRD §9.1 asks for time interpolation, but an interpolated value at t uses the value at t+1. Lag features are therefore built from a causal fill (forward fill <= 2 windows, then the train median). The interpolated column is used only for EDA and the dashboard.",
        "- **'Overcast'** is not in the documented vocabulary and is mapped to Cloudy.",
        "- **Congestion is derived from volume.** The V/C rule reproduces the existing labels almost exactly, so the same window's volume/V/C/occupancy/speed are never model inputs for that window.",
        "",
        "## 4. Descriptive statistics (observed values)",
        "",
        stats.round(2).to_markdown(index=False),
        "",
        "## 5. Weather vs traffic",
        "",
        weather_table.round(2).to_markdown(index=False),
        "",
        "## 6. Strongest feature correlations with next-window volume (train)",
        "",
        correlations.head(10).round(3).to_frame("pearson_r").to_markdown(),
        "",
        "## 7. Figures",
        "",
        "![Daily profile](figures/eda_daily_profile.png)",
        "![Congestion balance](figures/eda_congestion_balance.png)",
        "![Accident rate by hour](figures/eda_accident_rate_by_hour.png)",
        "![Weather vs speed](figures/eda_weather_speed.png)",
        "![Feature correlation](figures/eda_feature_correlation.png)",
        "![Missing windows](figures/eda_missing_windows.png)",
        "",
    ]
    (reports_dir / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")


def run_eda(clean, features_h1, dq_log, config, data_label):
    """Produce every EDA figure, analytics table and the data-quality report."""
    logger = get_logger()
    figures_dir = config["paths"]["figures_dir"]
    processed_dir = config["paths"]["processed_dir"]
    daily_profile_figure(clean, figures_dir)
    congestion_balance_figure(clean, config, figures_dir)
    accident_rate_figure(clean, figures_dir)
    weather_table = weather_effect_table(clean)
    weather_figure(weather_table, figures_dir)
    correlations = correlation_figure(features_h1, figures_dir)
    missingness_figure(clean, figures_dir)
    stats = summary_statistics(clean)
    weather_table.to_csv(processed_dir / "weather_effect.csv", index=False)
    correlations.to_frame("pearson_r").to_csv(processed_dir / "feature_correlation.csv")
    write_data_quality_report(clean, dq_log, config, stats, weather_table, correlations, data_label)
    logger.info("EDA figures and reports/data_quality.md written")
