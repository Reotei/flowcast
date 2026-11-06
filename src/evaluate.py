"""Metrics, the single scoreboard, threshold choice, intervals and model cards (PRD §10.2, §13)."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.utils import get_logger


# ----------------------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred):
    """RMSE, MAE, MAPE (%) and R^2.

    RMSE = sqrt(mean((y - y_hat)^2))      MAE = mean(|y - y_hat|)
    MAPE = 100 * mean(|y - y_hat| / y)     R^2 = 1 - SS_res / SS_tot
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(errors[nonzero]) / np.abs(y_true[nonzero])) * 100)
    total_sum_squares = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - np.sum(errors ** 2) / total_sum_squares) if total_sum_squares > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "n": int(len(y_true))}


def multiclass_metrics(y_true, y_pred, n_classes=4):
    """Macro-F1 (headline), macro precision/recall, accuracy and the confusion matrix."""
    labels = list(range(n_classes))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def binary_metrics(y_true, scores, threshold):
    """ROC-AUC (headline), PR-AUC, and precision/recall/F1 at the chosen threshold."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)) if has_both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, scores)) if has_both_classes else float("nan"),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "threshold": float(threshold),
        "positive_rate": float(y_true.mean()),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
    }


def choose_threshold(y_val, scores_val):
    """Pick the threshold that maximises F1 on the VALIDATION set (never on test)."""
    candidate_thresholds = np.unique(np.quantile(scores_val, np.linspace(0.50, 0.999, 300)))
    best_threshold = float(candidate_thresholds[0])
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        predicted = (scores_val >= threshold).astype(int)
        current_f1 = f1_score(y_val, predicted, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)
    return best_threshold


def prior_correct(weighted_probability, positive_prior):
    """Turn a probability from a class-weighted model back into a real-world probability.

    Balanced class weights make the model behave as if both classes were equally common
    (prior 0.5). Bayes' rule with the true prior pi restores the scale:
        odds_true = odds_weighted * pi / (1 - pi),   p = odds_true / (1 + odds_true)
    The transform is monotonic, so ROC-AUC and PR-AUC do not change; only the displayed
    probability (and therefore the threshold value) does.
    """
    probability = np.clip(np.asarray(weighted_probability, dtype=float), 1e-9, 1 - 1e-9)
    odds = probability / (1 - probability) * positive_prior / (1 - positive_prior)
    return odds / (1 + odds)


# ----------------------------------------------------------------------------- intervals
def fit_residual_quantiles(hours, y_true, y_pred, coverage=0.90):
    """Split-conformal style intervals: per hour of day, the `coverage` quantile of |residual|.

    Fitted on VALIDATION residuals; the interval for a new prediction is y_hat +/- q[hour].
    """
    frame = pd.DataFrame({"hour": hours, "abs_residual": np.abs(np.asarray(y_true) - np.asarray(y_pred))})
    quantiles = frame.groupby("hour")["abs_residual"].quantile(coverage)
    overall = float(frame["abs_residual"].quantile(coverage))
    return {"by_hour": {int(hour): float(value) for hour, value in quantiles.items()}, "overall": overall,
            "coverage": coverage}


def apply_residual_quantiles(hours, y_pred, quantile_model, lower_bound=0.0):
    """Return (lower, upper) interval arrays from fit_residual_quantiles output."""
    half_widths = []
    for hour in np.asarray(hours):
        half_widths.append(quantile_model["by_hour"].get(int(hour), quantile_model["overall"]))
    half_widths = np.asarray(half_widths)
    lower = np.maximum(np.asarray(y_pred) - half_widths, lower_bound)
    upper = np.asarray(y_pred) + half_widths
    return lower, upper


# ----------------------------------------------------------------------------- scoreboard
class Scoreboard:
    """One table for every (model, target, split) result — PRD §11.3 single scoreboard."""

    HEADLINE = {"volume": "rmse", "travel_time": "rmse", "congestion": "macro_f1", "accident": "roc_auc"}

    def __init__(self):
        self.rows = []

    def add(self, model_name, target, split, metrics, horizon=1, family="classical"):
        """Record a metrics dict (non-scalar entries such as confusion matrices are kept separately)."""
        row = {"model": model_name, "target": target, "split": split, "horizon": horizon, "family": family,
               "selected": 0}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                row[key] = float(value)
        if "confusion_matrix" in metrics:
            row["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
        self.rows.append(row)

    def mark_selected(self, target, model_name, horizon=1):
        """Flag the model chosen on VALIDATION for a target (its val and test rows)."""
        for row in self.rows:
            if row["target"] == target and row["horizon"] == horizon:
                row["selected"] = 1 if row["model"] == model_name else 0

    def to_frame(self):
        """Scoreboard as a DataFrame."""
        return pd.DataFrame(self.rows)

    def save(self, path):
        """Write scoreboard.csv (merging with any rows already on disk from other modules)."""
        frame = self.to_frame()
        try:
            existing = pd.read_csv(path)
            keys = ["model", "target", "split", "horizon"]
            existing = existing.merge(frame[keys], on=keys, how="left", indicator=True)
            existing = existing.loc[existing["_merge"] == "left_only"].drop(columns="_merge")
            frame = pd.concat([existing, frame], ignore_index=True)
        except FileNotFoundError:
            pass
        frame.to_csv(path, index=False)
        return frame


def leakage_alarm(target, metrics, config):
    """Warn loudly when a score is too good to be true (likely leakage)."""
    logger = get_logger()
    limit = config["success_targets"]["leakage_alarm_macro_f1"]
    if target == "congestion" and metrics.get("macro_f1", 0) > limit:
        logger.warning(f"LEAKAGE ALARM: congestion macro-F1 {metrics['macro_f1']:.3f} > {limit}. "
                       "Check that no same-window volume / V/C / occupancy / speed feature is used.")
        return True
    if target == "volume" and metrics.get("mape", 100) < 2.0:
        logger.warning(f"LEAKAGE ALARM: volume MAPE {metrics['mape']:.2f}% is suspiciously low.")
        return True
    return False


# ----------------------------------------------------------------------------- model cards
def write_model_card(path, model_name, target, horizon, config, feature_columns, hyperparameters,
                     val_metrics, test_metrics, notes=""):
    """Write a JSON + Markdown model card (PRD §13.4) next to the saved model."""
    card = {
        "model": model_name,
        "target": target,
        "horizon_windows": horizon,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_window": f"{config['grid']['start_date']} (+7 days warm-up) to {config['split']['train_end']}",
        "validation_window": f"day after {config['split']['train_end']} to {config['split']['val_end']}",
        "test_window": f"day after {config['split']['val_end']} to {config['grid']['end_date']}",
        "seed": config["seed"],
        "n_features": len(feature_columns),
        "features": list(feature_columns),
        "hyperparameters": {key: (value if isinstance(value, (int, float, str, bool, type(None))) else str(value))
                            for key, value in hyperparameters.items()},
        "validation_metrics": {key: value for key, value in val_metrics.items() if key != "confusion_matrix"},
        "test_metrics": {key: value for key, value in test_metrics.items() if key != "confusion_matrix"},
        "notes": notes,
    }
    with open(f"{path}.json", "w", encoding="utf-8") as card_file:
        json.dump(card, card_file, indent=2, default=str)

    lines = [f"# Model card — {model_name} ({target}, h={horizon})", ""]
    for key in ["training_window", "validation_window", "test_window", "seed", "n_features", "created_utc"]:
        lines.append(f"- **{key}**: {card[key]}")
    lines += ["", "## Hyperparameters", ""]
    for key, value in card["hyperparameters"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Metrics", "", "| metric | validation | test |", "| --- | ---: | ---: |"]
    for key, value in card["test_metrics"].items():
        val_value = card["validation_metrics"].get(key, float("nan"))
        lines.append(f"| {key} | {val_value:.4f} | {value:.4f} |")
    lines += ["", "## Features", "", ", ".join(feature_columns), "", f"Notes: {notes}", ""]
    with open(f"{path}.md", "w", encoding="utf-8") as card_file:
        card_file.write("\n".join(lines))
