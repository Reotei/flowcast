"""M6 — Deep-learning engine: GRU / LSTM sequence forecaster built and trained from scratch (FR-09).

Sample = (segment, target window T). Input = the previous N windows (T-N .. T-1) of scaled
per-step features for that segment, plus calendar/time features of T (known in advance) and two
seasonal lags (same window yesterday / last week). Output = next-window volume.
No pre-trained weights. Same test rows and metrics as the classical models.

Scaling (fit on TRAIN rows only):
  z-score  z = (x - mean) / std   for unbounded features (volume, speed, temperature)
  min-max  x' = (x - min) / (max - min)  for bounded features (occupancy, visibility)
"""

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.clean import derive_congestion
from src.evaluate import multiclass_metrics, regression_metrics
from src.features import causal_series, congestion_codes
from src.utils import get_logger, set_seed

STEP_FEATURES = ["volume_z", "speed_z", "occupancy_mm", "vc", "tod_sin", "tod_cos", "dow_sin", "dow_cos",
                 "is_weekend", "rain_flag", "visibility_mm", "temperature_z", "public_holiday", "event_flag",
                 "roadwork_flag"]
TARGET_FEATURES = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "public_holiday", "event_flag",
                   "roadwork_flag", "is_peak", "vol_same_window_yday_z", "vol_same_window_lastweek_z"]


class SequenceForecaster(nn.Module):
    """Recurrent encoder + dense head.

    GRU/LSTM layers read the sequence; the final hidden state h_N is concatenated with the
    target-window features and a learned segment embedding, then a dense layer maps it to the
    forecast:  y_hat = W2 * ReLU(W1 * [h_N, x_T, e_seg] + b1) + b2.
    """

    def __init__(self, n_step_features, n_target_features, n_segments, cell, hidden_size, num_layers, dropout):
        super().__init__()
        recurrent_class = nn.GRU if cell == "gru" else nn.LSTM
        self.recurrent = recurrent_class(input_size=n_step_features, hidden_size=hidden_size,
                                         num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
                                         batch_first=True)
        self.segment_embedding = nn.Embedding(n_segments, 4)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + n_target_features + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, sequence, target_features, segment_index):
        """Return one scaled volume forecast per sample."""
        outputs, _ = self.recurrent(sequence)
        last_hidden = outputs[:, -1, :]                     # hidden state after the last step (t0)
        combined = torch.cat([last_hidden, target_features, self.segment_embedding(segment_index)], dim=1)
        return self.head(combined).squeeze(1)


def build_step_arrays(clean, config):
    """Per-segment arrays of scaled step features: shape (n_segments, n_windows, n_features)."""
    clean = clean.sort_values(["road_id", "timestamp"]).reset_index(drop=True)
    short_gap = config["imputation"]["short_gap_max_windows"]
    is_train = (clean["split"] == "train").to_numpy()

    volume = causal_series(clean, "traffic_volume", short_gap).to_numpy(float)
    speed = causal_series(clean, "avg_speed", short_gap).to_numpy(float)
    occupancy = causal_series(clean, "occupancy", short_gap).to_numpy(float)
    temperature = clean["temperature"].to_numpy(float)
    visibility = clean["visibility"].to_numpy(float)

    scalers = {
        "volume": (volume[is_train].mean(), volume[is_train].std()),
        "speed": (speed[is_train].mean(), speed[is_train].std()),
        "temperature": (temperature[is_train].mean(), temperature[is_train].std()),
        "occupancy": (occupancy[is_train].min(), occupancy[is_train].max()),
        "visibility": (visibility[is_train].min(), visibility[is_train].max()),
    }
    local_time = clean["timestamp"].dt.tz_localize(None)
    window_of_day = (local_time.dt.hour * 2 + local_time.dt.minute // 30).to_numpy()
    day_of_week = local_time.dt.day_of_week.to_numpy()
    step = pd.DataFrame({
        "volume_z": (volume - scalers["volume"][0]) / scalers["volume"][1],
        "speed_z": (speed - scalers["speed"][0]) / scalers["speed"][1],
        "occupancy_mm": (occupancy - scalers["occupancy"][0]) / (scalers["occupancy"][1] - scalers["occupancy"][0]),
        "vc": volume / (clean["road_capacity"].to_numpy(float) / 2),
        "tod_sin": np.sin(2 * np.pi * window_of_day / 48),
        "tod_cos": np.cos(2 * np.pi * window_of_day / 48),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
        "is_weekend": (day_of_week >= 5).astype(float),
        "rain_flag": (clean["weather_condition"] == "Rain").to_numpy(float),
        "visibility_mm": (visibility - scalers["visibility"][0]) / (scalers["visibility"][1] - scalers["visibility"][0]),
        "temperature_z": (temperature - scalers["temperature"][0]) / scalers["temperature"][1],
        "public_holiday": clean["public_holiday"].to_numpy(float),
        "event_flag": clean["event_flag"].to_numpy(float),
        "roadwork_flag": clean["roadwork_flag"].to_numpy(float),
    })
    segments = sorted(clean["road_id"].unique())
    n_windows = len(clean) // len(segments)
    step_array = step[STEP_FEATURES].to_numpy(np.float32).reshape(len(segments), n_windows, len(STEP_FEATURES))
    first_timestamp = clean["timestamp"].iloc[0]
    return step_array, scalers, segments, first_timestamp


def build_samples(table, segments, first_timestamp, scalers, seq_len):
    """Turn feature-table rows (h=1) into index arrays + target-window features for the model."""
    segment_lookup = {segment_id: position for position, segment_id in enumerate(segments)}
    segment_index = table["road_id"].map(segment_lookup).to_numpy(np.int64)
    window_index = ((table["timestamp"] - first_timestamp) / pd.Timedelta(minutes=30)).to_numpy().astype(np.int64)
    volume_mean, volume_std = scalers["volume"]
    target_frame = table[TARGET_FEATURES[:-2]].astype(float).copy()
    target_frame["vol_same_window_yday_z"] = (table["vol_same_window_yday"] - volume_mean) / volume_std
    target_frame["vol_same_window_lastweek_z"] = (table["vol_same_window_lastweek"] - volume_mean) / volume_std
    valid = window_index >= seq_len
    return {
        "segment": segment_index[valid],
        "window": window_index[valid],
        "target_features": target_frame[TARGET_FEATURES].to_numpy(np.float32)[valid],
        "y": ((table["y_volume"].to_numpy(float) - volume_mean) / volume_std).astype(np.float32)[valid],
        "rows": table.loc[valid],
    }


def gather_sequences(step_tensor, segment_index, window_index, seq_len):
    """Collect step features for windows T-seq_len .. T-1 (all strictly before the target)."""
    offsets = torch.arange(-seq_len, 0)
    time_positions = window_index.unsqueeze(1) + offsets.unsqueeze(0)
    return step_tensor[segment_index.unsqueeze(1), time_positions]


def predict_scaled(model, step_tensor, samples, seq_len, batch_size=2048):
    """Model predictions (scaled units) for a sample dict, in eval mode."""
    model.eval()
    outputs = []
    segment_all = torch.from_numpy(samples["segment"])
    window_all = torch.from_numpy(samples["window"])
    target_all = torch.from_numpy(samples["target_features"])
    with torch.no_grad():
        for start in range(0, len(segment_all), batch_size):
            stop = start + batch_size
            sequences = gather_sequences(step_tensor, segment_all[start:stop], window_all[start:stop], seq_len)
            outputs.append(model(sequences, target_all[start:stop], segment_all[start:stop]).numpy())
    return np.concatenate(outputs)


def train_one(cell, step_tensor, train_samples, val_samples, n_segments, dl_config, seed):
    """Train one GRU/LSTM with Adam, LR-on-plateau, dropout and early stopping (best weights kept)."""
    logger = get_logger()
    set_seed(seed)
    seq_len = dl_config["seq_len"]
    model = SequenceForecaster(len(STEP_FEATURES), len(TARGET_FEATURES), n_segments, cell,
                               dl_config["hidden_size"], dl_config["num_layers"], dl_config["dropout"])
    optimiser = torch.optim.Adam(model.parameters(), lr=dl_config["learning_rate"],
                                 weight_decay=dl_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min",
                                                           factor=dl_config["lr_plateau_factor"],
                                                           patience=dl_config["lr_plateau_patience"])
    loss_function = nn.MSELoss()                              # J = mean((y_hat - y)^2)
    generator = torch.Generator().manual_seed(seed)

    segment_train = torch.from_numpy(train_samples["segment"])
    window_train = torch.from_numpy(train_samples["window"])
    target_train = torch.from_numpy(train_samples["target_features"])
    y_train = torch.from_numpy(train_samples["y"])
    y_val = val_samples["y"]
    history = []
    best_val_loss, best_state, epochs_without_improvement = np.inf, None, 0
    for epoch in range(1, dl_config["max_epochs"] + 1):
        start = time.time()
        model.train()
        # Shuffling mini-batch ORDER is fine: each sample's sequence is still built in time order.
        permutation = torch.randperm(len(y_train), generator=generator)
        running_loss, batches = 0.0, 0
        for batch_start in range(0, len(permutation), dl_config["batch_size"]):
            batch = permutation[batch_start:batch_start + dl_config["batch_size"]]
            sequences = gather_sequences(step_tensor, segment_train[batch], window_train[batch], seq_len)
            optimiser.zero_grad()
            loss = loss_function(model(sequences, target_train[batch], segment_train[batch]), y_train[batch])
            loss.backward()                                    # backpropagation through time
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running_loss += loss.item()
            batches += 1
        train_loss = running_loss / batches
        val_loss = float(np.mean((predict_scaled(model, step_tensor, val_samples, seq_len) - y_val) ** 2))
        scheduler.step(val_loss)
        current_lr = optimiser.param_groups[0]["lr"]
        history.append({"cell": cell, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})
        logger.info(f"    {cell} epoch {epoch:2d}: train {train_loss:.4f}  val {val_loss:.4f}  "
                    f"lr {current_lr:.5f}  ({time.time() - start:.0f}s)")
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= dl_config["early_stopping_patience"]:
                logger.info(f"    early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)                          # restore best weights
    return model, pd.DataFrame(history)


def plot_loss_curves(history, figures_dir):
    """Train vs validation loss for every cell type (one chart per cell, shared y-axis meaning)."""
    from src.viz_style import SERIES, apply_matplotlib_style

    plt = apply_matplotlib_style()
    cells = list(history["cell"].unique())
    figure, axes = plt.subplots(1, len(cells), figsize=(5 * len(cells), 3.6), squeeze=False)
    for axis, cell in zip(axes[0], cells):
        cell_history = history.loc[history["cell"] == cell]
        axis.plot(cell_history["epoch"], cell_history["train_loss"], color=SERIES[0], label="Train")
        axis.plot(cell_history["epoch"], cell_history["val_loss"], color=SERIES[1], label="Validation")
        axis.set_title(f"{cell.upper()} loss (MSE, scaled volume)")
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.savefig(figures_dir / "dl_loss_curves.png")
    plt.close(figure)


def run_deep_learning(clean, table_h1, config, fast=False):
    """Train GRU and LSTM, evaluate on the same test rows as M5, save models and predictions."""
    from src.evaluate import Scoreboard, write_model_card

    logger = get_logger()
    torch.set_num_threads(2)
    dl_config = dict(config["dl"])
    if fast:
        dl_config["max_epochs"] = 2
        dl_config["cells"] = ["gru"]
    seq_len = dl_config["seq_len"]
    models_dir = config["paths"]["models_dir"]
    processed_dir = config["paths"]["processed_dir"]

    step_array, scalers, segments, first_timestamp = build_step_arrays(clean, config)
    step_tensor = torch.from_numpy(np.ascontiguousarray(step_array).copy())
    samples = {}
    for split in ["train", "val", "test"]:
        rows = table_h1.loc[(table_h1["split"] == split) & table_h1["mask_volume"]].sort_values(["timestamp", "road_id"])
        if fast:
            rows = rows.loc[rows["road_id"].isin(segments[:8])]
        samples[split] = build_samples(rows, segments, first_timestamp, scalers, seq_len)
    eval_rows = table_h1.loc[table_h1["split"].isin(["val", "test"])].sort_values(["timestamp", "road_id"])
    eval_samples = build_samples(eval_rows, segments, first_timestamp, scalers, seq_len)
    logger.info(f"M6 | sequences train {len(samples['train']['y']):,}, val {len(samples['val']['y']):,}, "
                f"test {len(samples['test']['y']):,}; step features {len(STEP_FEATURES)}, seq_len {seq_len}")

    volume_mean, volume_std = scalers["volume"]
    scoreboard = Scoreboard()
    histories, results = [], {}
    for cell in dl_config["cells"]:
        logger.info(f"M6 | training {cell.upper()}")
        model, history = train_one(cell, step_tensor, samples["train"], samples["val"], len(segments), dl_config,
                                   config["seed"])
        histories.append(history)
        metrics = {}
        for split in ["val", "test"]:
            predicted = predict_scaled(model, step_tensor, samples[split], seq_len) * volume_std + volume_mean
            actual = samples[split]["rows"]["y_volume"].to_numpy(float)
            metrics[split] = regression_metrics(actual, predicted)
            scoreboard.add(cell.upper(), "volume", split, metrics[split], family="deep")
            labels = derive_congestion(predicted, samples[split]["rows"]["road_capacity"], config)
            codes = congestion_codes(labels, config["congestion_order"]).astype(int)
            congestion_rows = samples[split]["rows"]["mask_congestion"].to_numpy()
            congestion_metrics = multiclass_metrics(samples[split]["rows"]["y_congestion"].to_numpy()[congestion_rows].astype(int),
                                                    codes[congestion_rows])
            scoreboard.add(f"{cell.upper()} volume -> V/C bands", "congestion", split, congestion_metrics, family="deep")
        torch.save(model.state_dict(), models_dir / f"volume_h1_{cell.upper()}.pt")
        hyperparameters = {key: value for key, value in dl_config.items() if key != "cells"}
        hyperparameters["cell"] = cell
        hyperparameters["epochs_trained"] = int(history["epoch"].max())
        write_model_card(models_dir / f"volume_h1_{cell.upper()}", cell.upper(), "volume", 1, config,
                         STEP_FEATURES + TARGET_FEATURES + ["segment_embedding(4)"], hyperparameters,
                         metrics["val"], metrics["test"], "trained from scratch; best epoch restored")
        results[cell] = {"model": model, "metrics": metrics}
        logger.info(f"  {cell.upper()} test RMSE {metrics['test']['rmse']:.2f}, MAPE {metrics['test']['mape']:.2f}%")

    history_frame = pd.concat(histories, ignore_index=True)
    history_frame.to_csv(config["paths"]["reports_dir"] / "dl_training_history.csv", index=False)
    plot_loss_curves(history_frame, config["paths"]["figures_dir"])

    best_cell = min(results, key=lambda cell: results[cell]["metrics"]["val"]["rmse"])
    predictions_path = processed_dir / "predictions_h1.parquet"
    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        for cell, result in results.items():
            predicted = predict_scaled(result["model"], step_tensor, eval_samples, seq_len) * volume_std + volume_mean
            keyed = pd.DataFrame({"road_id": eval_samples["rows"]["road_id"].to_numpy(),
                                  "timestamp": eval_samples["rows"]["timestamp"].to_numpy(),
                                  f"pred_volume__{cell.upper()}": predicted})
            column = f"pred_volume__{cell.upper()}"
            if column in predictions.columns:
                predictions = predictions.drop(columns=column)
            predictions = predictions.merge(keyed, on=["road_id", "timestamp"], how="left")
        predictions.to_parquet(predictions_path, index=False)

    import joblib

    joblib.dump({"scalers": scalers, "segments": segments, "first_timestamp": first_timestamp,
                 "best_cell": best_cell, "config": {k: v for k, v in dl_config.items()}},
                models_dir / "dl_meta.joblib")
    scoreboard.save(config["paths"]["reports_dir"] / "scoreboard.csv")
    return results, history_frame, best_cell
