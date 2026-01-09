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

import pytest
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            "model_uri": "models:/test-model/1",
            "deployed_at": "2025-01-01 00:00:00",
            "version": 1,
            "tracking_uri": "file:./data/mlflow/tracking"
        }
        yaml.dump(config, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup with retry for Windows
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
    """Create temporary artifacts directory with mock files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_path = Path(tmpdir) / "artifacts" / "models"
        artifacts_path.mkdir(parents=True)

        # Create mock config
        config = {
            "input_size": 5,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.2,
            "architecture": "LSTM"
        }
        with open(artifacts_path / "model_config.json", 'w') as f:
            json.dump(config, f)

        # Create mock model file (empty is fine for tests)
        (artifacts_path / "model_lstm_1x16.pt").touch()

        # Create mock scaler (empty is fine for tests)
        (artifacts_path / "scaler_corrected.pkl").touch()

        yield artifacts_path


def test_model_service_singleton():
    """Test ModelService implements singleton pattern"""
    with patch.object(ModelService, '_load_artifacts'):
        service1 = ModelService()
        service2 = ModelService()

    assert service1 is service2


def test_load_from_mlflow_success(temp_production_config):
    """Test successful model loading from MLflow"""
    # Mock MLflow loading
    mock_model = Mock()
    mock_scaler = Mock()

    with patch('mlflow.pytorch.load_model', return_value=mock_model):
        with patch('joblib.load', return_value=mock_scaler):
            with patch.object(Path, 'exists', return_value=True):
                service = ModelService.__new__(ModelService)
                service.production_config_path = temp_production_config

                result = service._load_from_mlflow("models:/test/1")

    assert result is True


def test_load_from_local_artifacts_success(temp_artifacts_dir):
    """Test successful model loading from local artifacts"""
    # Mock PyTorch model
    mock_model = Mock()
    mock_scaler = Mock()

    with patch('src.api.services.model_service.StockLSTM', return_value=mock_model):
        with patch('torch.load', return_value={}):  # Empty state dict
            with patch('joblib.load', return_value=mock_scaler):
                service = ModelService.__new__(ModelService)
                service.artifacts_path = temp_artifacts_dir

                result = service._load_from_local_artifacts()

    assert result is True
    assert service.model is not None
    assert service.scaler is not None


def test_fallback_to_local_when_mlflow_fails(temp_production_config, temp_artifacts_dir):
    """Test fallback to local artifacts when MLflow fails"""
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_scaler = Mock()

    # Mock state dict for torch.load
    mock_state_dict = {
        'num_tickers': 2,
        'num_features': 19,
        'embedding_dim': 8,
        'hidden_size': 16,
        'num_layers': 1,
        'dropout': 0.2
    }

    # Patch paths BEFORE ModelService init
    with patch('src.api.services.model_service.StockLSTM', return_value=mock_model):
        with patch('torch.load', return_value=mock_state_dict):
            with patch('joblib.load', return_value=mock_scaler):
                with patch('mlflow.pytorch.load_model', side_effect=Exception("MLflow down")):
                    # Create service without auto-loading
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


def test_reload_reloads_model():
    """Test reload() method reloads model"""
    service = ModelService.__new__(ModelService)
    service._initialized = True
    service.model = Mock()

    # Mock _load_artifacts as simple Mock (does nothing)
    mock_load = Mock()

    with patch.object(service, '_load_artifacts', mock_load):
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

    # Set one component to None
    setattr(service, missing_component, None)

    assert service.is_ready() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
