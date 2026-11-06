"""Tests for the from-scratch gradient descent, the metrics and threshold selection."""

import numpy as np
import pytest

from src.evaluate import (apply_residual_quantiles, binary_metrics, choose_threshold, fit_residual_quantiles,
                          multiclass_metrics, regression_metrics)
from src.ml_models import LinearRegressionGD


def test_gradient_descent_matches_the_closed_form_solution():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(2000, 4)) * [1, 10, 100, 0.1]
    true_weights = np.array([2.0, -0.5, 0.03, 7.0])
    y = X @ true_weights + 5.0 + rng.normal(0, 0.1, 2000)
    model = LinearRegressionGD(learning_rate=0.1, epochs=2000).fit(X, y)
    X_with_bias = np.column_stack([X, np.ones(len(X))])
    closed_form = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]   # normal-equation solution
    assert np.allclose(model.predict(X), X_with_bias @ closed_form, atol=0.05)


def test_gradient_descent_cost_never_increases():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    y = X @ [1.0, 2.0, 3.0]
    history = LinearRegressionGD(learning_rate=0.1, epochs=200).fit(X, y).cost_history
    assert all(later <= earlier + 1e-12 for earlier, later in zip(history, history[1:]))


def test_regression_metrics_known_values():
    metrics = regression_metrics([100, 200], [110, 180])
    assert metrics["mae"] == pytest.approx(15.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert metrics["mape"] == pytest.approx((10 / 100 + 20 / 200) / 2 * 100)


def test_macro_f1_is_perfect_for_perfect_predictions():
    assert multiclass_metrics([0, 1, 2, 3], [0, 1, 2, 3])["macro_f1"] == pytest.approx(1.0)


def test_threshold_is_chosen_on_validation_scores():
    y_val = np.array([0] * 95 + [1] * 5)
    scores = np.concatenate([np.linspace(0, 0.5, 95), np.linspace(0.8, 0.9, 5)])
    threshold = choose_threshold(y_val, scores)
    assert 0.5 < threshold <= 0.8
    assert binary_metrics(y_val, scores, threshold)["f1"] == pytest.approx(1.0)


def test_residual_intervals_cover_about_the_requested_share():
    rng = np.random.default_rng(3)
    hours = rng.integers(0, 24, 20000)
    y_true = rng.normal(100, 10, 20000)
    y_pred = np.full(20000, 100.0)
    model = fit_residual_quantiles(hours[:10000], y_true[:10000], y_pred[:10000], coverage=0.9)
    lower, upper = apply_residual_quantiles(hours[10000:], y_pred[10000:], model)
    coverage = np.mean((y_true[10000:] >= lower) & (y_true[10000:] <= upper))
    assert 0.87 < coverage < 0.93


def test_prior_correction_restores_real_world_probabilities():
    from src.evaluate import prior_correct

    # With balanced weights a "neutral" model outputs 0.5; the true probability is the prior itself.
    assert prior_correct([0.5], 0.01)[0] == pytest.approx(0.01)
    corrected = prior_correct(np.array([0.2, 0.5, 0.9]), 0.01)
    assert np.all(np.diff(corrected) > 0)          # monotonic, so ROC-AUC is unchanged
