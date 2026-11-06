"""M5 — Classical ML engine (FR-07, FR-08, FR-10, FR-11, FR-12).

For each target:
  1. naive baseline(s) so every model has something to beat;
  2. hyperparameter search with time-series CV on the TRAIN window only (explicit loop);
  3. final fit on train, model selection / early stopping / thresholds on VALIDATION;
  4. one look at the untouched TEST window, metrics only on originally observed targets.
Linear regression is first implemented with NumPy gradient descent (PRD build-it-once rule).
"""

import itertools
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from src.clean import derive_congestion
from src.evaluate import (Scoreboard, apply_residual_quantiles, binary_metrics, choose_threshold,
                          fit_residual_quantiles, leakage_alarm, multiclass_metrics, prior_correct,
                          regression_metrics, write_model_card)
from src.features import congestion_codes, get_feature_columns
from src.utils import get_logger

SEED = 42
N_JOBS = 2


# ============================================================================ build-it-once
class LinearRegressionGD:
    """Linear regression trained by batch gradient descent, written with NumPy only.

    Model:     y_hat = X w + b
    Cost:      J(w, b) = (1 / 2m) * sum_i (y_hat_i - y_i)^2
    Gradients: dJ/dw = (1/m) * X^T (y_hat - y)      dJ/db = (1/m) * sum_i (y_hat_i - y_i)
    Update:    w <- w - alpha * dJ/dw              b <- b - alpha * dJ/db

    Features and target are z-scored with TRAIN statistics so one learning rate suits all
    columns: z = (x - mean) / std.
    """

    def __init__(self, learning_rate=0.1, epochs=600):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_history = []

    def fit(self, X, y):
        """Learn w and b; stores the cost after every epoch in self.cost_history."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std == 0] = 1.0           # constant columns: avoid divide-by-zero
        self.target_mean = y.mean()
        self.target_std = y.std()
        X_scaled = (X - self.feature_mean) / self.feature_std
        y_scaled = (y - self.target_mean) / self.target_std

        n_rows, n_features = X_scaled.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []
        for epoch in range(self.epochs):
            predictions = X_scaled @ self.weights + self.bias       # matrix-vector product Xw + b
            residuals = predictions - y_scaled
            cost = (residuals @ residuals) / (2 * n_rows)
            self.cost_history.append(float(cost))
            gradient_weights = (X_scaled.T @ residuals) / n_rows
            gradient_bias = residuals.mean()
            self.weights = self.weights - self.learning_rate * gradient_weights
            self.bias = self.bias - self.learning_rate * gradient_bias
        return self

    def predict(self, X):
        """Predict in original units (undo the target z-score)."""
        X_scaled = (np.asarray(X, dtype=float) - self.feature_mean) / self.feature_std
        return (X_scaled @ self.weights + self.bias) * self.target_std + self.target_mean

    def get_params(self, deep=False):
        """Hyperparameters, for the model card."""
        return {"learning_rate": self.learning_rate, "epochs": self.epochs}


# ============================================================================ data helpers
def time_series_folds(timestamps, n_splits):
    """Expanding-window CV folds built on UNIQUE timestamps, so a fold never splits one window.

    Returns a list of (train_row_positions, validation_row_positions).
    """
    timestamps = pd.Series(timestamps).reset_index(drop=True)
    unique_times = np.sort(timestamps.unique())
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_time_positions, val_time_positions in splitter.split(unique_times):
        train_times = unique_times[train_time_positions]
        val_times = unique_times[val_time_positions]
        train_rows = np.flatnonzero(timestamps.isin(train_times).to_numpy())
        val_rows = np.flatnonzero(timestamps.isin(val_times).to_numpy())
        folds.append((train_rows, val_rows))
    return folds


def select_rows(table, split, mask_column):
    """Rows of one split whose target was originally observed, in time order."""
    rows = table.loc[(table["split"] == split) & table[mask_column]]
    return rows.sort_values(["timestamp", "road_id"])


def subsample_in_time_order(frame, max_rows, seed=SEED):
    """Random subsample (for tuning speed) that keeps the rows sorted by time."""
    if max_rows is None or len(frame) <= max_rows:
        return frame
    sampled = frame.sample(max_rows, random_state=seed)
    return sampled.sort_values(["timestamp", "road_id"])


def tune_with_time_series_cv(build_model, param_grid, X, y, timestamps, n_splits, score_function,
                             use_class_weights=False):
    """Grid search written as an explicit loop over (parameters x time-series folds).

    Args:
        build_model: function(params) -> unfitted model.
        param_grid: dict of name -> list of values.
        score_function: function(model, X_val, y_val) -> score (higher is better).
        use_class_weights: pass balanced sample weights to fit (for XGBoost classifiers).

    Returns:
        (best_params, list of result dicts)
    """
    logger = get_logger()
    folds = time_series_folds(timestamps, n_splits)
    parameter_names = list(param_grid.keys())
    results = []
    best_params, best_score = {}, -np.inf
    for values in itertools.product(*[param_grid[name] for name in parameter_names]):
        params = dict(zip(parameter_names, values))
        fold_scores = []
        for train_rows, val_rows in folds:
            model = build_model(params)
            if use_class_weights:
                weights = compute_sample_weight("balanced", y[train_rows])
                model.fit(X[train_rows], y[train_rows], sample_weight=weights)
            else:
                model.fit(X[train_rows], y[train_rows])
            fold_scores.append(score_function(model, X[val_rows], y[val_rows]))
        mean_score = float(np.mean(fold_scores))
        results.append({"params": params, "mean_score": mean_score, "fold_std": float(np.std(fold_scores))})
        if mean_score > best_score:
            best_score, best_params = mean_score, params
    if len(parameter_names) > 0:
        logger.info(f"    CV best {best_params} score={best_score:.4f}")
    return best_params, results


# ============================================================================ scorers
def negative_rmse(model, X, y):
    """Score for regression tuning (higher is better)."""
    return -float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))


def macro_f1_score(model, X, y):
    """Score for congestion tuning."""
    return multiclass_metrics(y, model.predict(X))["macro_f1"]


def positive_scores(model, X):
    """Probability (or margin for SVM) of the positive class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def roc_auc_scorer(model, X, y):
    """Score for accident tuning."""
    return binary_metrics(y, positive_scores(model, X), 0.5)["roc_auc"]


# ============================================================================ model zoo
def regression_zoo(fast):
    """(name -> (builder, grid)) for the regression targets (FR-07)."""
    trees_small = 40 if fast else 60
    return {
        "LinearRegression": (lambda p: make_pipeline(StandardScaler(), LinearRegression()), {}),
        "DecisionTree": (lambda p: DecisionTreeRegressor(random_state=SEED, **p),
                         {"max_depth": [8, 12, 16], "min_samples_leaf": [5, 20]}),
        "RandomForest": (lambda p: RandomForestRegressor(n_estimators=trees_small, max_features=0.5,
                                                         n_jobs=N_JOBS, random_state=SEED, **p),
                         {"max_depth": [12, 20], "min_samples_leaf": [2, 10]}),
        "XGBoost": (lambda p: XGBRegressor(n_estimators=300, tree_method="hist", subsample=0.8,
                                           colsample_bytree=0.8, random_state=SEED, n_jobs=N_JOBS, **p),
                    {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]}),
    }


def classification_zoo(fast, binary):
    """(name -> (builder, grid)) for the classification targets (FR-08), incl. the SVM baseline."""
    trees_small = 40 if fast else 60
    xgb_objective = {"objective": "binary:logistic"} if binary else {"objective": "multi:softprob"}
    return {
        "LogisticRegression": (lambda p: make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=400, class_weight="balanced", random_state=SEED, **p)), {"C": [0.1, 1.0]}),
        "DecisionTree": (lambda p: DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **p),
                         {"max_depth": [6, 10, 14], "min_samples_leaf": [10, 50]}),
        "RandomForest": (lambda p: RandomForestClassifier(n_estimators=trees_small, max_features=0.5,
                                                          class_weight="balanced_subsample", n_jobs=N_JOBS,
                                                          random_state=SEED, **p),
                         {"max_depth": [10, 18], "min_samples_leaf": [5, 20]}),
        "XGBoost": (lambda p: XGBClassifier(n_estimators=300, tree_method="hist", subsample=0.8,
                                            colsample_bytree=0.8, random_state=SEED, n_jobs=N_JOBS,
                                            eval_metric="logloss" if binary else "mlogloss",
                                            **xgb_objective, **p),
                    {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]}),
        "SVM (LinearSVC)": (lambda p: make_pipeline(StandardScaler(), LinearSVC(
            class_weight="balanced", random_state=SEED, max_iter=3000, **p)), {"C": [0.01, 0.1]}),
    }


def final_model_from_params(name, builder, best_params, fast, task):
    """Rebuild the tuned model with full-size settings for the final fit on all train rows."""
    model = builder(best_params)
    if name == "RandomForest":
        model.set_params(n_estimators=60 if fast else 150)
    if name == "XGBoost":
        model.set_params(n_estimators=300 if fast else 1500, early_stopping_rounds=50)
    return model


def fit_model(name, model, X_train, y_train, X_val, y_val, class_weighted):
    """Fit one model; XGBoost uses the validation set for early stopping only."""
    fit_kwargs = {}
    if class_weighted and name == "XGBoost":
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
    if name == "XGBoost":
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ============================================================================ per-target runs
class TargetRun:
    """Holds everything produced for one target so the driver can save it."""

    def __init__(self, target):
        self.target = target
        self.models = {}
        self.params = {}
        self.val_metrics = {}
        self.test_metrics = {}
        self.predictions = {}           # name -> predictions for the val+test frame
        self.best_name = None
        self.extra = {}


def run_regression(table, target, y_column, mask_column, baselines, config, scoreboard, fast):
    """Train baselines + the regression zoo for one continuous target."""
    logger = get_logger()
    feature_columns = get_feature_columns(table)
    train = select_rows(table, "train", mask_column)
    val = select_rows(table, "val", mask_column)
    test = select_rows(table, "test", mask_column)
    eval_frame = table.loc[table["split"].isin(["val", "test"])].sort_values(["timestamp", "road_id"])
    X_train, y_train = train[feature_columns].to_numpy(float), train[y_column].to_numpy(float)
    X_val, y_val = val[feature_columns].to_numpy(float), val[y_column].to_numpy(float)
    X_test, y_test = test[feature_columns].to_numpy(float), test[y_column].to_numpy(float)
    run = TargetRun(target)

    for baseline_name, baseline_column in baselines.items():
        run.val_metrics[baseline_name] = regression_metrics(y_val, val[baseline_column])
        run.test_metrics[baseline_name] = regression_metrics(y_test, test[baseline_column])
        run.predictions[baseline_name] = eval_frame[baseline_column].to_numpy(float)

    if target == "volume":
        logger.info("  LinearRegression-GD (NumPy, from scratch)")
        gd_model = LinearRegressionGD(config["ml"]["gd_learning_rate"], config["ml"]["gd_epochs"]).fit(X_train, y_train)
        run.models["LinearRegression-GD"] = gd_model
        run.params["LinearRegression-GD"] = gd_model.get_params()

    tuning_frame = subsample_in_time_order(train, 25000 if fast else 40000)
    X_tune, y_tune = tuning_frame[feature_columns].to_numpy(float), tuning_frame[y_column].to_numpy(float)
    for name, (builder, grid) in regression_zoo(fast).items():
        start = time.time()
        logger.info(f"  {name}")
        best_params, _ = tune_with_time_series_cv(builder, grid, X_tune, y_tune, tuning_frame["timestamp"],
                                                  config["cv"]["n_splits"], negative_rmse)
        model = final_model_from_params(name, builder, best_params, fast, "regression")
        model = fit_model(name, model, X_train, y_train, X_val, y_val, class_weighted=False)
        run.models[name] = model
        run.params[name] = dict(best_params)
        if name == "XGBoost":
            run.params[name]["best_iteration"] = int(model.best_iteration)
        logger.info(f"    fitted in {time.time() - start:.0f}s")

    for name, model in run.models.items():
        run.val_metrics[name] = regression_metrics(y_val, model.predict(X_val))
        run.test_metrics[name] = regression_metrics(y_test, model.predict(X_test))
        run.predictions[name] = model.predict(eval_frame[feature_columns].to_numpy(float))

    for name in run.val_metrics:
        family = "baseline" if name in baselines else "classical"
        scoreboard.add(name, target, "val", run.val_metrics[name], family=family)
        scoreboard.add(name, target, "test", run.test_metrics[name], family=family)
        leakage_alarm(target, run.test_metrics[name], config)

    trained_names = list(run.models.keys())
    run.best_name = min(trained_names, key=lambda model_name: run.val_metrics[model_name]["rmse"])
    best_val_predictions = run.models[run.best_name].predict(X_val)
    run.extra["interval_model"] = fit_residual_quantiles(val["timestamp"].dt.hour, y_val, best_val_predictions)
    lower, upper = apply_residual_quantiles(eval_frame["timestamp"].dt.hour, run.predictions[run.best_name],
                                            run.extra["interval_model"])
    test_lower, test_upper = apply_residual_quantiles(test["timestamp"].dt.hour,
                                                      run.models[run.best_name].predict(X_test),
                                                      run.extra["interval_model"])
    run.extra["test_interval_coverage"] = float(np.mean((y_test >= test_lower) & (y_test <= test_upper)))
    run.extra["eval_frame"] = eval_frame
    run.extra["interval_lower"], run.extra["interval_upper"] = lower, upper
    run.extra["feature_columns"] = feature_columns
    logger.info(f"  best {target} model on validation: {run.best_name} "
                f"(val RMSE {run.val_metrics[run.best_name]['rmse']:.2f}, "
                f"90% interval test coverage {run.extra['test_interval_coverage']:.1%})")
    return run


def run_classification(table, target, y_column, mask_column, baseline, config, scoreboard, fast, binary):
    """Train a baseline + the classifier zoo (incl. SVM) for congestion or accident risk."""
    logger = get_logger()
    feature_columns = get_feature_columns(table)
    train = select_rows(table, "train", mask_column)
    val = select_rows(table, "val", mask_column)
    test = select_rows(table, "test", mask_column)
    eval_frame = table.loc[table["split"].isin(["val", "test"])].sort_values(["timestamp", "road_id"])
    X_train, y_train = train[feature_columns].to_numpy(float), train[y_column].to_numpy(int)
    X_val, y_val = val[feature_columns].to_numpy(float), val[y_column].to_numpy(int)
    X_test, y_test = test[feature_columns].to_numpy(float), test[y_column].to_numpy(int)
    run = TargetRun(target)
    run.extra["thresholds"] = {}

    baseline_name, baseline_column = baseline
    if binary:
        threshold = choose_threshold(y_val, val[baseline_column].to_numpy(float))
        run.extra["thresholds"][baseline_name] = threshold
        run.val_metrics[baseline_name] = binary_metrics(y_val, val[baseline_column], threshold)
        run.test_metrics[baseline_name] = binary_metrics(y_test, test[baseline_column], threshold)
    else:
        run.val_metrics[baseline_name] = multiclass_metrics(y_val, val[baseline_column].astype(int))
        run.test_metrics[baseline_name] = multiclass_metrics(y_test, test[baseline_column].astype(int))
    run.predictions[baseline_name] = eval_frame[baseline_column].to_numpy(float)

    tuning_frame = subsample_in_time_order(train, 25000 if fast else 40000)
    X_tune, y_tune = tuning_frame[feature_columns].to_numpy(float), tuning_frame[y_column].to_numpy(int)
    scorer = roc_auc_scorer if binary else macro_f1_score
    for name, (builder, grid) in classification_zoo(fast, binary).items():
        start = time.time()
        logger.info(f"  {name}")
        best_params, _ = tune_with_time_series_cv(builder, grid, X_tune, y_tune, tuning_frame["timestamp"],
                                                  config["cv"]["n_splits"], scorer,
                                                  use_class_weights=(name == "XGBoost" and not binary))
        model = final_model_from_params(name, builder, best_params, fast, "classification")
        if binary and name == "XGBoost":
            positive_share = max(y_train.mean(), 1e-6)
            model.set_params(scale_pos_weight=(1 - positive_share) / positive_share)
        if name.startswith("SVM"):
            svm_rows = subsample_in_time_order(train, config["ml"]["svm_train_rows"])
            model.fit(svm_rows[feature_columns].to_numpy(float), svm_rows[y_column].to_numpy(int))
        else:
            model = fit_model(name, model, X_train, y_train, X_val, y_val, class_weighted=not binary)
        run.models[name] = model
        run.params[name] = dict(best_params)
        logger.info(f"    fitted in {time.time() - start:.0f}s")

    X_eval = eval_frame[feature_columns].to_numpy(float)
    positive_prior = float(y_train.mean()) if binary else None
    run.extra["positive_prior"] = positive_prior

    def risk_scores(model, X):
        """Positive-class score; class-weighted probabilities are prior-corrected to real probabilities."""
        scores = positive_scores(model, X)
        if hasattr(model, "predict_proba"):
            scores = prior_correct(scores, positive_prior)
        return scores

    for name, model in run.models.items():
        if binary:
            val_scores = risk_scores(model, X_val)
            threshold = choose_threshold(y_val, val_scores)
            run.extra["thresholds"][name] = threshold
            run.val_metrics[name] = binary_metrics(y_val, val_scores, threshold)
            run.test_metrics[name] = binary_metrics(y_test, risk_scores(model, X_test), threshold)
            run.predictions[name] = risk_scores(model, X_eval)
        else:
            run.val_metrics[name] = multiclass_metrics(y_val, model.predict(X_val))
            run.test_metrics[name] = multiclass_metrics(y_test, model.predict(X_test))
            run.predictions[name] = model.predict(X_eval)

    headline = "roc_auc" if binary else "macro_f1"
    for name in run.val_metrics:
        family = "baseline" if name == baseline_name else "classical"
        scoreboard.add(name, target, "val", run.val_metrics[name], family=family)
        scoreboard.add(name, target, "test", run.test_metrics[name], family=family)
        leakage_alarm(target, run.test_metrics[name], config)

    probabilistic_models = [name for name in run.models if hasattr(run.models[name], "predict_proba")]
    run.best_name = max(probabilistic_models, key=lambda model_name: run.val_metrics[model_name][headline])
    if not binary:
        run.extra["best_probabilities"] = run.models[run.best_name].predict_proba(X_eval)
    run.extra["eval_frame"] = eval_frame
    run.extra["feature_columns"] = feature_columns
    logger.info(f"  best {target} model on validation: {run.best_name} "
                f"(val {headline} {run.val_metrics[run.best_name][headline]:.3f})")
    return run


def congestion_from_volume(volume_run, table, config, scoreboard):
    """Alternative congestion route: band the best volume forecast with the V/C rule."""
    order = config["congestion_order"]
    best_model = volume_run.models[volume_run.best_name]
    feature_columns = volume_run.extra["feature_columns"]
    results = {}
    for split in ["val", "test"]:
        rows = select_rows(table, split, "mask_congestion")
        predicted_volume = best_model.predict(rows[feature_columns].to_numpy(float))
        predicted_labels = derive_congestion(predicted_volume, rows["road_capacity"], config)
        predicted_codes = congestion_codes(predicted_labels, order).astype(int)
        metrics = multiclass_metrics(rows["y_congestion"].astype(int), predicted_codes)
        scoreboard.add(f"{volume_run.best_name} volume -> V/C bands", "congestion", split, metrics)
        results[split] = metrics
    return results


def feature_importance_table(runs):
    """Tree-ensemble feature importances (RF impurity, XGBoost gain) for every target."""
    rows = []
    for target, run in runs.items():
        feature_columns = run.extra["feature_columns"]
        for name in ["RandomForest", "XGBoost"]:
            model = run.models.get(name)
            if model is None:
                continue
            importances = model.feature_importances_
            total = importances.sum() if importances.sum() > 0 else 1.0
            for feature, importance in zip(feature_columns, importances):
                rows.append({"target": target, "model": name, "feature": feature,
                             "importance": float(importance / total)})
    return pd.DataFrame(rows)


def save_gd_check(volume_run, table, config):
    """Compare GD weights with sklearn's closed-form fit and plot the cost curve (for the review)."""
    from src.viz_style import SERIES, apply_matplotlib_style

    plt = apply_matplotlib_style()
    gd_model = volume_run.models["LinearRegression-GD"]
    figure, axis = plt.subplots(figsize=(7, 3.6))
    axis.plot(gd_model.cost_history, color=SERIES[0])
    axis.set_yscale("log")
    axis.set_title("Gradient descent: cost J(w, b) per epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("J (scaled units, log)")
    figure.savefig(config["paths"]["figures_dir"] / "gd_cost_curve.png")
    plt.close(figure)
    gd_val = volume_run.val_metrics["LinearRegression-GD"]["rmse"]
    sk_val = volume_run.val_metrics["LinearRegression"]["rmse"]
    return {"gd_val_rmse": gd_val, "sklearn_val_rmse": sk_val, "final_cost": gd_model.cost_history[-1],
            "epochs": gd_model.epochs, "learning_rate": gd_model.learning_rate}


def train_volume_horizons(tables, best_params, config, scoreboard, fast):
    """Volume XGBoost for horizons 2..4 (re-using the h=1 tuned parameters), for the forecast views."""
    logger = get_logger()
    outputs = {}
    for horizon, table in tables.items():
        if horizon == 1:
            continue
        feature_columns = get_feature_columns(table)
        train = select_rows(table, "train", "mask_volume")
        val = select_rows(table, "val", "mask_volume")
        test = select_rows(table, "test", "mask_volume")
        model = XGBRegressor(n_estimators=300 if fast else 1500, early_stopping_rounds=50, tree_method="hist",
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=N_JOBS,
                             max_depth=best_params.get("max_depth", 6),
                             learning_rate=best_params.get("learning_rate", 0.1))
        model.fit(train[feature_columns].to_numpy(float), train["y_volume"].to_numpy(float),
                  eval_set=[(val[feature_columns].to_numpy(float), val["y_volume"].to_numpy(float))], verbose=False)
        val_pred = model.predict(val[feature_columns].to_numpy(float))
        val_metrics = regression_metrics(val["y_volume"], val_pred)
        test_metrics = regression_metrics(test["y_volume"], model.predict(test[feature_columns].to_numpy(float)))
        scoreboard.add("XGBoost", "volume", "val", val_metrics, horizon=horizon)
        scoreboard.add("XGBoost", "volume", "test", test_metrics, horizon=horizon)
        persistence_metrics = regression_metrics(test["y_volume"], test["vol_t0"])
        scoreboard.add("Baseline: last observed (t0)", "volume", "test", persistence_metrics, horizon=horizon,
                       family="baseline")
        interval_model = fit_residual_quantiles(val["timestamp"].dt.hour, val["y_volume"], val_pred)
        outputs[horizon] = {"model": model, "feature_columns": feature_columns, "val_metrics": val_metrics,
                            "test_metrics": test_metrics, "interval_model": interval_model}
        logger.info(f"  volume h={horizon}: test RMSE {test_metrics['rmse']:.2f}, MAPE {test_metrics['mape']:.2f}%")
    return outputs


# ============================================================================ driver
def run_classical(tables, config, fast=False):
    """Run M5 end to end; save models, model cards, scoreboard, predictions and importances."""
    logger = get_logger()
    models_dir = config["paths"]["models_dir"]
    reports_dir = config["paths"]["reports_dir"]
    processed_dir = config["paths"]["processed_dir"]
    table = tables[1]
    if fast:
        keep_segments = sorted(table["road_id"].unique())[:8]
        tables = {h: t.loc[t["road_id"].isin(keep_segments)].reset_index(drop=True) for h, t in tables.items()}
        table = tables[1]
    scoreboard = Scoreboard()
    runs = {}

    logger.info("M5 | volume (regression)")
    runs["volume"] = run_regression(table, "volume", "y_volume", "mask_volume",
                                    {"Baseline: last observed (t0)": "vol_t0",
                                     "Baseline: same window last week": "vol_same_window_lastweek"},
                                    config, scoreboard, fast)
    logger.info("M5 | travel time (regression)")
    runs["travel_time"] = run_regression(table, "travel_time", "y_travel_time", "mask_travel_time",
                                         {"Baseline: last observed (t0)": "travel_time_t0"},
                                         config, scoreboard, fast)
    logger.info("M5 | congestion (4-class)")
    runs["congestion"] = run_classification(table, "congestion", "y_congestion", "mask_congestion",
                                            ("Baseline: persistence (t0 class)", "congestion_t0"),
                                            config, scoreboard, fast, binary=False)
    banding_results = congestion_from_volume(runs["volume"], table, config, scoreboard)
    banding_name = f"{runs['volume'].best_name} volume -> V/C bands"
    classifier_val_f1 = runs["congestion"].val_metrics[runs["congestion"].best_name]["macro_f1"]
    congestion_route = "volume_bands" if banding_results["val"]["macro_f1"] > classifier_val_f1 else "classifier"
    logger.info(f"  congestion route chosen on validation: {congestion_route} "
                f"(classifier {classifier_val_f1:.3f} vs V/C bands {banding_results['val']['macro_f1']:.3f})")
    logger.info("M5 | accident risk (binary)")
    runs["accident"] = run_classification(table, "accident", "y_accident", "mask_accident",
                                          ("Baseline: past-30-day accident count", "accidents_past_30d"),
                                          config, scoreboard, fast, binary=True)
    logger.info("M5 | volume horizons 2-4")
    horizon_models = train_volume_horizons(tables, runs["volume"].params.get("XGBoost", {}), config, scoreboard, fast)
    gd_check = save_gd_check(runs["volume"], table, config)
    for target, run in runs.items():
        selected_name = run.best_name
        if target == "congestion" and congestion_route == "volume_bands":
            selected_name = banding_name
        scoreboard.mark_selected(target, selected_name)

    # ---- persist models (FR-12) + model cards (PRD §13.4)
    registry = {"targets": {}, "horizons": {}, "gd_check": gd_check}
    for target, run in runs.items():
        for name, model in run.models.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            file_stem = models_dir / f"{target}_h1_{safe_name}"
            joblib.dump(model, f"{file_stem}.joblib")
            notes = "selected on validation" if name == run.best_name else ""
            if target == "accident":
                notes += f"; operating threshold {run.extra['thresholds'][name]:.4f} chosen on validation"
            write_model_card(file_stem, name, target, 1, config, run.extra["feature_columns"], run.params[name],
                             run.val_metrics[name], run.test_metrics[name], notes)
        entry = {"best_model": run.best_name,
                 "best_file": f"{run.target}_h1_{run.best_name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib",
                 "feature_columns": run.extra["feature_columns"]}
        if "interval_model" in run.extra:
            entry["interval_model"] = run.extra["interval_model"]
            entry["test_interval_coverage"] = run.extra["test_interval_coverage"]
        if run.extra.get("thresholds"):
            entry["threshold"] = run.extra["thresholds"][run.best_name]
            entry["positive_prior"] = run.extra["positive_prior"]
        if target == "congestion":
            entry["route"] = congestion_route
        registry["targets"][target] = entry
    for horizon, output in horizon_models.items():
        file_name = f"volume_h{horizon}_XGBoost.joblib"
        joblib.dump(output["model"], models_dir / file_name)
        write_model_card(models_dir / f"volume_h{horizon}_XGBoost", "XGBoost", "volume", horizon, config,
                         output["feature_columns"], output["model"].get_params(), output["val_metrics"],
                         output["test_metrics"])
        registry["horizons"][horizon] = {"file": file_name, "feature_columns": output["feature_columns"],
                                         "interval_model": output["interval_model"]}
    joblib.dump(registry, models_dir / "registry.joblib")

    # ---- predictions on validation + test windows for the dashboard
    order = config["congestion_order"]
    eval_frame = runs["volume"].extra["eval_frame"]
    predictions = eval_frame[["road_id", "timestamp", "split", "road_capacity", "y_volume", "y_travel_time",
                              "y_congestion", "y_accident", "mask_volume", "mask_travel_time",
                              "mask_congestion", "mask_accident", "vol_t0", "vol_same_window_lastweek"]].copy()
    for name, values in runs["volume"].predictions.items():
        predictions[f"pred_volume__{name}"] = values
    predictions["pred_volume"] = runs["volume"].predictions[runs["volume"].best_name]
    predictions["pred_volume_lower"] = runs["volume"].extra["interval_lower"]
    predictions["pred_volume_upper"] = runs["volume"].extra["interval_upper"]
    predictions["pred_travel_time"] = runs["travel_time"].predictions[runs["travel_time"].best_name]
    predictions["pred_travel_time_lower"] = runs["travel_time"].extra["interval_lower"]
    predictions["pred_travel_time_upper"] = runs["travel_time"].extra["interval_upper"]
    predictions["pred_congestion"] = runs["congestion"].predictions[runs["congestion"].best_name].astype(int)
    probabilities = runs["congestion"].extra["best_probabilities"]
    for code, label in enumerate(order):
        predictions[f"proba_{label}"] = probabilities[:, code]
    predictions["pred_congestion_confidence"] = probabilities.max(axis=1)
    predictions["accident_risk"] = runs["accident"].predictions[runs["accident"].best_name]
    predictions["accident_flag"] = (predictions["accident_risk"] >= registry["targets"]["accident"]["threshold"]).astype(int)
    predictions.to_parquet(processed_dir / "predictions_h1.parquet", index=False)

    horizon_rows = []
    for horizon, output in horizon_models.items():
        horizon_table = tables[horizon]
        rows = horizon_table.loc[horizon_table["split"].isin(["val", "test"])]
        predicted = output["model"].predict(rows[output["feature_columns"]].to_numpy(float))
        lower, upper = apply_residual_quantiles(rows["timestamp"].dt.hour, predicted, output["interval_model"])
        horizon_rows.append(pd.DataFrame({"road_id": rows["road_id"].to_numpy(), "timestamp": rows["timestamp"].to_numpy(),
                                          "horizon": horizon, "y_volume": rows["y_volume"].to_numpy(),
                                          "mask_volume": rows["mask_volume"].to_numpy(), "pred_volume": predicted,
                                          "pred_volume_lower": lower, "pred_volume_upper": upper}))
    h1 = predictions[["road_id", "timestamp", "y_volume", "mask_volume", "pred_volume", "pred_volume_lower",
                      "pred_volume_upper"]].copy()
    h1.insert(2, "horizon", 1)
    pd.concat([h1] + horizon_rows, ignore_index=True).to_parquet(processed_dir / "predictions_horizons.parquet",
                                                                 index=False)

    feature_importance_table(runs).to_csv(processed_dir / "feature_importance.csv", index=False)
    scoreboard.save(reports_dir / "scoreboard.csv")
    logger.info(f"M5 done: {len(scoreboard.rows)} scoreboard rows; models in {models_dir}")
    return runs, scoreboard, registry
