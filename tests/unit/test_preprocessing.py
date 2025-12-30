"""Unit tests for data preprocessing."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.ml.data.preprocessing import StockPreprocessor


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = StockPreprocessor(
        lookback_period=60, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    assert preprocessor.lookback_period == 60
    assert preprocessor.train_ratio == 0.7
    assert preprocessor.val_ratio == 0.15
    assert preprocessor.test_ratio == 0.15


def test_preprocessor_invalid_ratios():
    """Test preprocessor with invalid split ratios."""
    with pytest.raises(ValueError):
        StockPreprocessor(train_ratio=0.6, val_ratio=0.2, test_ratio=0.1)


def test_normalize():
    """Test data normalization."""
    preprocessor = StockPreprocessor()

    data = np.array([[10], [20], [30], [40], [50]])

    normalized = preprocessor.normalize(data, fit=True)

    # Should be in range [0, 1]
    assert normalized.min() >= 0
    assert normalized.max() <= 1
    assert preprocessor.is_fitted


def test_create_sequences():
    """Test sequence creation with PyTorch."""
    preprocessor = StockPreprocessor(lookback_period=3)

    data = np.array([[1], [2], [3], [4], [5]])

    X, y = preprocessor.create_sequences(data, lookback=3)

    # Now returns PyTorch tensors
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # X should have sequences of 3, y should have next values
    assert X.shape == (2, 3, 1)  # 2 sequences, lookback=3, 1 feature
    assert y.shape == (2,)

    # Check values
    assert torch.allclose(X[0], torch.FloatTensor([[1], [2], [3]]))
    assert y[0].item() == 4


def test_split_data():
    """Test train/val/test split with PyTorch."""
    preprocessor = StockPreprocessor(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    # Create PyTorch tensors directly
    X = torch.randn(100, 10, 1)
    y = torch.randn(100)

    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data(X, y)

    # Check all are tensors
    assert all(
        isinstance(t, torch.Tensor) for t in [X_train, y_train, X_val, y_val, X_test, y_test]
    )

    # Check sizes
    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20


def test_to_device():
    """Test moving tensors to device."""
    preprocessor = StockPreprocessor()

    X = torch.randn(10, 5, 1)
    y = torch.randn(10)

    # Move to CPU (default)
    X_cpu, y_cpu = preprocessor.to_device(X, y)

    assert isinstance(X_cpu, torch.Tensor)
    assert isinstance(y_cpu, torch.Tensor)
    assert X_cpu.device.type == "cpu"
    assert y_cpu.device.type == "cpu"


def test_prepare_data_pipeline():
    """Test complete preprocessing pipeline."""
    # Create dummy dataframe
    dates = pd.date_range("2020-01-01", periods=200)
    df = pd.DataFrame({"Close": np.random.randn(200).cumsum() + 100}, index=dates)

    preprocessor = StockPreprocessor(lookback_period=10)

    data = preprocessor.prepare_data(df)

    # Check all keys present
    assert "X_train" in data
    assert "y_train" in data
    assert "X_val" in data
    assert "y_val" in data
    assert "X_test" in data
    assert "y_test" in data
    assert "scaler" in data

    # Check tensors
    assert isinstance(data["X_train"], torch.Tensor)
    assert isinstance(data["y_train"], torch.Tensor)
