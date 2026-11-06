"""Shared helpers: config loading, seeding, logging and the data-quality log."""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path=None):
    """Load config.yaml and turn every path entry into an absolute Path.

    Args:
        config_path: optional path to a YAML file; defaults to <project>/config.yaml.

    Returns:
        dict with the configuration.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    for key, relative_path in config["paths"].items():
        absolute_path = Path(relative_path)
        if not absolute_path.is_absolute():
            absolute_path = PROJECT_ROOT / absolute_path
        config["paths"][key] = absolute_path
    return config


def ensure_dirs(config):
    """Create every output directory listed in config['paths']."""
    for path in config["paths"].values():
        Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    """Seed Python, NumPy and (if installed) PyTorch so reruns are deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_logger(name="flowcast"):
    """Return a console logger with a simple, readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def assign_split(timestamps, config):
    """Label each timestamp 'train', 'val' or 'test' by calendar date (no shuffling).

    Args:
        timestamps: tz-aware Series of window start times.
        config: loaded configuration (uses split.train_end and split.val_end).

    Returns:
        Series of strings aligned with timestamps.
    """
    local_dates = timestamps.dt.tz_localize(None).dt.normalize()
    train_end = np.datetime64(config["split"]["train_end"])
    val_end = np.datetime64(config["split"]["val_end"])
    split = np.where(local_dates <= train_end, "train", np.where(local_dates <= val_end, "val", "test"))
    return pd.Series(split, index=timestamps.index)


class DataQualityLog:
    """Collects one entry per cleaning rule so the data-quality report can be generated.

    Each entry records the pipeline stage, the rule, how many rows/values it touched
    and what was done with them. Nothing is changed silently (PRD §5 Reliability).
    """

    def __init__(self):
        self.entries = []

    def add(self, stage, rule, count, action, note=""):
        """Record one rule application."""
        entry = {
            "stage": stage,
            "rule": rule,
            "count": int(count),
            "action": action,
            "note": note,
        }
        self.entries.append(entry)
        get_logger().info(f"[DQ] {stage} | {rule}: {int(count):,} -> {action}")

    def to_markdown_table(self):
        """Return the log as a Markdown table string."""
        lines = ["| Stage | Rule | Count | Action | Note |", "| --- | --- | ---: | --- | --- |"]
        for entry in self.entries:
            lines.append(
                f"| {entry['stage']} | {entry['rule']} | {entry['count']:,} | {entry['action']} | {entry['note']} |"
            )
        return "\n".join(lines)

    def save(self, path):
        """Write the log as JSON (read later by the EDA report and the dashboard)."""
        with open(path, "w", encoding="utf-8") as output_file:
            json.dump(self.entries, output_file, indent=2)

    @classmethod
    def load(cls, path):
        """Read a saved log back from JSON."""
        log = cls()
        with open(path, "r", encoding="utf-8") as input_file:
            log.entries = json.load(input_file)
        return log
