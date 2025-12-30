"""Unit tests for metrics calculation."""

import numpy as np

from src.ml.training.metrics import (
    calculate_all_metrics,
    calculate_directional_accuracy,
    calculate_mae,
    calculate_mape,
    calculate_r2_score,
    calculate_rmse,
)


def test_calculate_mae():
    """Test MAE calculation."""
    y_true = np.array([100, 110, 120, 130])
    y_pred = np.array([102, 108, 121, 128])

    mae = calculate_mae(y_true, y_pred)

    # MAE = (2 + 2 + 1 + 2) / 4 = 1.75
    assert abs(mae - 1.75) < 0.01


def test_calculate_rmse():
    """Test RMSE calculation."""
    y_true = np.array([100, 110, 120])
    y_pred = np.array([102, 108, 121])

    rmse = calculate_rmse(y_true, y_pred)

    # RMSE = sqrt((4 + 4 + 1) / 3) = sqrt(3) ≈ 1.732
    assert abs(rmse - 1.732) < 0.01


def test_calculate_mape():
    """Test MAPE calculation."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])

    mape = calculate_mape(y_true, y_pred)

    # MAPE = mean(|110-100|/100, |190-200|/200, |310-300|/300) * 100
    # = mean(0.1, 0.05, 0.0333) * 100 = 6.11%
    assert abs(mape - 6.11) < 0.1


def test_calculate_r2_score():
    """Test R² score calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])

    r2 = calculate_r2_score(y_true, y_pred)

    # Should be close to 1 for good predictions
    assert r2 > 0.9


def test_calculate_directional_accuracy():
    """Test directional accuracy."""
    # True: up, up, down, up
    y_true = np.array([100, 105, 110, 108, 112])

    # Pred: up, up, down, up (all correct)
    y_pred = np.array([100, 103, 107, 106, 110])

    da = calculate_directional_accuracy(y_true, y_pred)

    # All directions correct = 100%
    assert abs(da - 100.0) < 0.01


def test_calculate_all_metrics():
    """Test calculating all metrics together."""
    y_true = np.array([100, 110, 120, 115, 125])
    y_pred = np.array([102, 108, 121, 114, 126])

    metrics = calculate_all_metrics(y_true, y_pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert "R2" in metrics
    assert "Directional_Accuracy" in metrics

    # All metrics should be positive
    assert all(isinstance(v, float) for v in metrics.values())


def test_mape_with_zero_values():
    """Test MAPE handles near-zero values."""
    y_true = np.array([0.001, 100, 200])
    y_pred = np.array([0.002, 102, 198])

    # Should not raise division by zero
    mape = calculate_mape(y_true, y_pred)
    assert isinstance(mape, float)
    assert mape >= 0
