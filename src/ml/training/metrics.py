"""Metrics calculation for model evaluation.

This module provides functions to calculate common regression metrics
used in stock price prediction: MAE, RMSE, MAPE, etc.
"""

from typing import Dict

import numpy as np
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE).

    MAE measures the average magnitude of errors in predictions,
    without considering their direction.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MAE value.

    Formula:
        MAE = (1/n) * Σ|y_true - y_pred|

    Example:
        >>> y_true = np.array([100, 110, 120])
        >>> y_pred = np.array([102, 108, 121])
        >>> mae = calculate_mae(y_true, y_pred)
        >>> print(f"MAE: {mae:.2f}")
    """
    mae = mean_absolute_error(y_true, y_pred)
    return float(mae)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    RMSE is the square root of the average of squared differences
    between predictions and actual observations. It penalizes large errors more.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        RMSE value.

    Formula:
        RMSE = √[(1/n) * Σ(y_true - y_pred)²]

    Example:
        >>> y_true = np.array([100, 110, 120])
        >>> y_pred = np.array([102, 108, 121])
        >>> rmse = calculate_rmse(y_true, y_pred)
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return float(rmse)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    MAPE expresses error as a percentage of actual values.
    Useful for comparing errors across different scales.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        epsilon: Small value to avoid division by zero.

    Returns:
        MAPE value as percentage (0-100).

    Formula:
        MAPE = (100/n) * Σ|(y_true - y_pred) / y_true|

    Note:
        MAPE is undefined when y_true contains zeros.
        Use epsilon to avoid division by zero.

    Example:
        >>> y_true = np.array([100, 110, 120])
        >>> y_pred = np.array([102, 108, 121])
        >>> mape = calculate_mape(y_true, y_pred)
        >>> print(f"MAPE: {mape:.2f}%")
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return float(mape)


def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination) score.

    R² represents the proportion of variance in the target variable
    that is predictable from the features. Range: (-∞, 1].

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        R² score. 1.0 is perfect, 0.0 means model is no better than mean.

    Formula:
        R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y_true - y_pred)²
              SS_tot = Σ(y_true - mean(y_true))²
    """
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true, y_pred)
    return float(r2)


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate directional accuracy (up/down prediction).

    Measures what percentage of time the model correctly predicts
    whether the price will go up or down compared to previous value.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Directional accuracy as percentage (0-100).

    Example:
        >>> y_true = np.array([100, 102, 101, 105])
        >>> y_pred = np.array([100, 103, 100, 106])
        >>> da = calculate_directional_accuracy(y_true, y_pred)
        >>> # Checks if both went up/down compared to previous value
    """
    # Calculate direction changes
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0

    # Calculate accuracy
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)

    accuracy = (correct / total) * 100
    return float(accuracy)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary with all metrics:
            - MAE: Mean Absolute Error
            - RMSE: Root Mean Squared Error
            - MAPE: Mean Absolute Percentage Error
            - R2: R-squared score
            - Directional_Accuracy: Percentage of correct direction predictions

    Example:
        >>> y_true = np.array([100, 110, 120, 115, 125])
        >>> y_pred = np.array([102, 108, 121, 114, 126])
        >>> metrics = calculate_all_metrics(y_true, y_pred)
        >>> for metric, value in metrics.items():
        ...     print(f"{metric}: {value:.4f}")
    """
    metrics = {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "R2": calculate_r2_score(y_true, y_pred),
        "Directional_Accuracy": calculate_directional_accuracy(y_true, y_pred),
    }

    logger.debug(f"Calculated metrics: {metrics}")

    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Test") -> None:
    """Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics.
        dataset_name: Name of the dataset (e.g., "Test", "Validation").
    """
    print("\n" + "=" * 60)
    print(f"📊 {dataset_name} Set Metrics")
    print("=" * 60)

    for metric_name, value in metrics.items():
        if metric_name in ["MAPE", "Directional_Accuracy"]:
            print(f"{metric_name:25s}: {value:8.2f}%")
        else:
            print(f"{metric_name:25s}: {value:10.4f}")

    print("=" * 60)


def evaluate_prediction_quality(mape: float, directional_accuracy: float) -> str:
    """Evaluate overall prediction quality based on metrics.

    Args:
        mape: MAPE value.
        directional_accuracy: Directional accuracy value.

    Returns:
        Quality assessment string.
    """
    if mape < 5 and directional_accuracy > 60:
        return "🟢 Excellent - Model performs very well"
    elif mape < 10 and directional_accuracy > 55:
        return "🟡 Good - Model performance is acceptable"
    elif mape < 15:
        return "🟠 Fair - Model needs improvement"
    else:
        return "🔴 Poor - Model needs significant improvement"
