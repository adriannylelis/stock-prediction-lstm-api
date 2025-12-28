"""Unit tests for LSTM model."""

import pytest
import torch

from src.ml.models.lstm import StockLSTM, create_model


def test_lstm_initialization():
    """Test LSTM model initialization."""
    model = StockLSTM(input_size=1, hidden_size=50, num_layers=2, dropout=0.2)

    assert model.input_size == 1
    assert model.hidden_size == 50
    assert model.num_layers == 2
    assert model.dropout_prob == 0.2


def test_lstm_forward_pass():
    """Test LSTM forward pass."""
    model = StockLSTM(input_size=1, hidden_size=50, num_layers=2)

    # Create dummy input: (batch_size=32, sequence_length=60, features=1)
    x = torch.randn(32, 60, 1)

    # Forward pass
    output = model(x)

    # Check output shape: (batch_size, 1)
    assert output.shape == (32, 1)


def test_lstm_invalid_parameters():
    """Test LSTM with invalid parameters."""
    with pytest.raises(ValueError):
        StockLSTM(input_size=0, hidden_size=50)

    with pytest.raises(ValueError):
        StockLSTM(input_size=1, hidden_size=-10)

    with pytest.raises(ValueError):
        StockLSTM(input_size=1, hidden_size=50, dropout=1.5)


def test_create_model_factory():
    """Test model factory function."""
    model = create_model(input_size=1, hidden_size=100, device="cpu")

    assert isinstance(model, StockLSTM)
    assert model.hidden_size == 100


def test_model_summary():
    """Test model summary method."""
    model = StockLSTM(input_size=1, hidden_size=50, num_layers=2)
    summary = model.summary()

    assert "StockLSTM Model Summary" in summary
    assert "Hidden size: 50" in summary
    assert "Number of layers: 2" in summary


def test_model_parameters_count():
    """Test parameter counting."""
    model = StockLSTM(input_size=1, hidden_size=50, num_layers=2)
    num_params = model.get_num_parameters()

    assert num_params > 0
    assert isinstance(num_params, int)
