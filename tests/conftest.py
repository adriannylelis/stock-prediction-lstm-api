"""
pytest configuration and shared fixtures

This file defines pytest fixtures available to all tests.

Author: MLOps Team
Created: 2025-01-07
"""

import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def temp_mlflow_tracking():
    """Create temporary MLflow tracking directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_path = Path(tmpdir) / "data" / "mlflow" / "tracking"
        tracking_path.mkdir(parents=True)
        yield tracking_path


@pytest.fixture
def temp_mlflow_artifacts():
    """Create temporary MLflow artifacts directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_path = Path(tmpdir) / "data" / "mlflow" / "artifacts"
        artifacts_path.mkdir(parents=True)
        yield artifacts_path


@pytest.fixture
def temp_production_config():
    """Create temporary production config file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "model_uri": "models:/test-model/1",
            "deployed_at": "2025-01-01 00:00:00",
            "deployed_by": "pytest",
            "tracking_uri": "file:./data/mlflow/tracking",
            "version": 1,
            "metrics": {"r2": 0.85, "mae": 0.02, "rmse": 0.03},
        }
        yaml.dump(config, f)
        yield Path(f.name)
        Path(f.name).unlink()


@pytest.fixture
def mock_model_config():
    """Mock model configuration dict"""
    return {
        "input_size": 5,
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.2,
        "architecture": "LSTM",
        "num_tickers": 3,
    }


@pytest.fixture
def mock_training_metrics():
    """Mock training metrics"""
    return {
        "train_loss": 0.015,
        "val_loss": 0.020,
        "val_r2": 0.85,
        "val_mae": 0.018,
        "val_rmse": 0.025,
        "epochs": 50,
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
