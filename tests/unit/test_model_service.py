"""
Unit Tests for Model Service

Tests the ModelService singleton without requiring actual models.
Uses mocks and fixtures to simulate model loading scenarios.

Author: MLOps Team
Created: 2025-01-07
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import pytest
import torch
import yaml

from src.api.services.model_service import ModelService


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset ModelService singleton between tests"""
    ModelService._instance = None
    yield
    ModelService._instance = None


@pytest.fixture
def temp_production_config():
    """Create temporary production config"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "model_uri": "models:/test-model/1",
            "deployed_at": "2025-01-01 00:00:00",
            "version": 1,
            "tracking_uri": "file:./data/mlflow/tracking",
        }
        yaml.dump(config, f)
        temp_path = Path(f.name)

    yield temp_path

    import time

    for _ in range(3):
        try:
            if temp_path.exists():
                temp_path.unlink()
            break
        except PermissionError:
            time.sleep(0.1)


@pytest.fixture
def temp_artifacts_dir():
    """Create temporary artifacts directory with new layout (best_model + scalers)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_path = Path(tmpdir) / "artifacts" / "models"
        scalers_path = artifacts_path / "scalers"
        scalers_path.mkdir(parents=True)

        checkpoint = {
            "num_tickers": 1,
            "num_features": 5,
            "embedding_dim": 8,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.2,
            "model_state_dict": {},
        }
        torch.save(checkpoint, artifacts_path / "best_model.pt")

        # Scalers (simple dict objects are enough for tests)
        joblib.dump({"scaler": "x"}, scalers_path / "scaler.pkl")
        joblib.dump({"scaler": "y"}, scalers_path / "y_scaler.pkl")

        # Preprocessing config with feature metadata
        prep_config = {
            "num_features": 5,
            "num_tickers": 1,
            "lookback": 60,
            "ticker_to_id": {"TEST": 0},
        }
        with open(scalers_path / "preprocessing_config.json", "w") as f:
            json.dump(prep_config, f)

        yield artifacts_path


@pytest.fixture
def temp_legacy_artifacts_dir():
    """Create temporary artifacts directory with legacy layout (model_lstm_1x16)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_path = Path(tmpdir) / "artifacts" / "models"
        artifacts_path.mkdir(parents=True)

        legacy_config = {
            "input_size": 5,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.2,
            "architecture": "LSTM",
        }
        with open(artifacts_path / "model_config.json", "w") as f:
            json.dump(legacy_config, f)

        # Legacy model/state dict
        torch.save({}, artifacts_path / "model_lstm_1x16.pt")

        # Legacy scaler
        joblib.dump({"scaler": "legacy"}, artifacts_path / "scaler_corrected.pkl")

        yield artifacts_path


def test_model_service_singleton():
    """Test ModelService implements singleton pattern"""
    with patch.object(ModelService, "_load_artifacts"):
        service1 = ModelService()
        service2 = ModelService()

    assert service1 is service2


def test_load_from_mlflow_success(temp_production_config):
    """Test successful model loading from MLflow"""
    # Skip this test - requires full MLflow setup
    pytest.skip("MLflow integration test - requires running MLflow server")


def test_load_from_local_artifacts_success(temp_artifacts_dir):
    """Test successful model loading from local artifacts"""
    mock_model = Mock()
    mock_model.load_state_dict = Mock()
    mock_model.eval = Mock()

    checkpoint = {
        "num_tickers": 1,
        "num_features": 5,
        "embedding_dim": 8,
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.2,
        "model_state_dict": {},
    }

    with patch("src.api.services.model_service.StockLSTM", return_value=mock_model):
        with patch("torch.load", return_value=checkpoint):
            with patch("joblib.load", return_value={"scaler": "x"}):
                service = ModelService.__new__(ModelService)
                service.artifacts_path = temp_artifacts_dir

                result = service._load_from_local_artifacts()

    assert result is True
    assert service.model is not None
    assert service.scaler is not None


def test_fallback_to_local_when_mlflow_fails(
    temp_production_config, temp_artifacts_dir
):
    """Test fallback to local artifacts when MLflow fails"""
    mock_model = Mock()
    mock_model.load_state_dict = Mock()
    mock_model.eval = Mock()

    checkpoint = {
        "num_tickers": 1,
        "num_features": 5,
        "embedding_dim": 8,
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.2,
        "model_state_dict": {},
    }

    with patch("mlflow.pytorch.load_model", side_effect=Exception("MLflow down")):
        with patch("src.api.services.model_service.StockLSTM", return_value=mock_model):
            with patch("torch.load", return_value=checkpoint):
                with patch("joblib.load", return_value={"scaler": "x"}):
                    service = ModelService.__new__(ModelService)
                    service._initialized = False
                    service.model = None
                    service.scaler = None
                    service.config = None
                    service.production_config_path = temp_production_config
                    service.artifacts_path = temp_artifacts_dir

                    # Now load with mocked paths
                    service._load_artifacts()
                    service._initialized = True

    # Should fall back to local
    assert service.model is not None
    assert service.scaler is not None


def test_load_from_legacy_artifacts_success(temp_legacy_artifacts_dir):
    """Ensure legacy layout still loads when best_model.pt is missing."""
    mock_model = Mock()
    mock_model.load_state_dict = Mock()
    mock_model.eval = Mock()
    mock_scaler = Mock()

    with patch("src.api.services.model_service.StockLSTM", return_value=mock_model):
        with patch("torch.load", return_value={}):
            with patch("joblib.load", return_value=mock_scaler):
                service = ModelService.__new__(ModelService)
                service.artifacts_path = temp_legacy_artifacts_dir

                result = service._load_from_local_artifacts()

    assert result is True
    assert service.model is not None
    assert service.scaler is not None


def test_reload_reloads_model():
    """Test reload() method reloads model"""
    service = ModelService.__new__(ModelService)
    service._initialized = True
    service.model = Mock()

    # Mock _load_artifacts as simple Mock (does nothing)
    mock_load = Mock()

    with patch.object(service, "_load_artifacts", mock_load):
        result = service.reload()

    assert result is True
    assert service._initialized is True
    mock_load.assert_called_once()


def test_get_model_raises_when_not_loaded():
    """Test get_model() raises error if model not loaded"""
    service = ModelService.__new__(ModelService)
    service.model = None

    with pytest.raises(RuntimeError, match="Modelo não foi carregado"):
        service.get_model()


def test_get_scaler_raises_when_not_loaded():
    """Test get_scaler() raises error if scaler not loaded"""
    service = ModelService.__new__(ModelService)
    service.scaler = None

    with pytest.raises(RuntimeError, match="Scaler não foi carregado"):
        service.get_scaler()


def test_is_ready_returns_true_when_all_loaded():
    """Test is_ready() returns True when all components loaded"""
    service = ModelService.__new__(ModelService)
    service.model = Mock()
    service.scaler = Mock()
    service.config = {"input_size": 5}

    assert service.is_ready() is True


def test_is_ready_returns_false_when_model_missing():
    """Test is_ready() returns False when model missing"""
    service = ModelService.__new__(ModelService)
    service.model = None
    service.scaler = Mock()
    service.config = {"input_size": 5}

    assert service.is_ready() is False


@pytest.mark.parametrize("missing_component", ["model", "scaler", "config"])
def test_is_ready_false_with_missing_components(missing_component):
    """Test is_ready() returns False when any component is missing"""
    service = ModelService.__new__(ModelService)
    service.model = Mock()
    service.scaler = Mock()
    service.config = {"input_size": 5}

    setattr(service, missing_component, None)

    assert service.is_ready() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
