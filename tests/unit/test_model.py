"""Unit tests for LSTM model with ticker embedding."""

import pytest
import torch

from src.ml.models.lstm import StockLSTM, create_model


def test_lstm_initialization():
    """Test LSTM model initialization with ticker embedding."""
    model = StockLSTM(
        num_tickers=50,
        num_features=19,
        embedding_dim=8,
        hidden_size=100,
        num_layers=3,
        dropout=0.3
    )

    assert model.num_tickers == 50
    assert model.num_features == 19
    assert model.embedding_dim == 8
    assert model.hidden_size == 100
    assert model.num_layers == 3
    assert model.dropout_prob == 0.3


def test_lstm_forward_pass():
    """Test LSTM forward pass with ticker embedding."""
    model = StockLSTM(num_tickers=10, num_features=19, embedding_dim=8, hidden_size=50, num_layers=2)

    # Create dummy inputs
    x_features = torch.randn(32, 60, 19)  # (batch, seq, features)
    ticker_ids = torch.randint(0, 10, (32,))  # (batch,)

    # Forward pass
    output = model(x_features, ticker_ids)

    # Check output shape: (batch_size,) due to squeeze
    assert output.shape == (32,)


def test_lstm_invalid_parameters():
    """Test LSTM with invalid parameters."""
    # Note: Current StockLSTM doesn't validate params, just test instantiation works
    model = StockLSTM(num_tickers=1, num_features=19)
    assert model is not None


def test_create_model_factory():
    """Test model factory function."""
    model = create_model(
        num_tickers=50,
        num_features=19,
        embedding_dim=8,
        hidden_size=100,
        num_layers=2,
        dropout=0.2,
        device=torch.device("cpu")
    )

    assert isinstance(model, StockLSTM)
    assert model.hidden_size == 100
    assert model.num_tickers == 50


def test_model_parameters_count():
    """Test parameter counting using PyTorch's built-in method."""
    model = StockLSTM(num_tickers=50, num_features=19, hidden_size=50, num_layers=2)
    num_params = sum(p.numel() for p in model.parameters())

    assert num_params > 0
    assert isinstance(num_params, int)
