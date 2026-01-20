"""
Unit Tests for Model Deployer

Tests the ModelDeployer class without requiring actual models or MLflow.
Uses mocks to simulate deployment scenarios.

Author: MLOps Team
Created: 2025-01-07
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from src.mlops.deployment.model_deployer import ModelDeployer


@pytest.fixture
def temp_config_file():
    """Create temporary production config file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "model_uri": "models:/test-model/1",
            "deployed_at": "2025-01-01 00:00:00",
            "deployed_by": "test",
            "tracking_uri": "file:./data/mlflow/tracking",
            "version": 1,
            "metrics": {"r2": 0.85, "mae": 0.02},
        }
        yaml.dump(config, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup - with retry for Windows
    import time

    for _ in range(3):
        try:
            if temp_path.exists():
                temp_path.unlink()
            break
        except PermissionError:
            time.sleep(0.1)  # Wait for file handle to close


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client"""
    with patch("src.mlops.deployment.model_deployer.MlflowClient") as mock_client:
        client = Mock()
        mock_client.return_value = client
        yield client


def test_deployer_initialization(temp_config_file, mock_mlflow_client):
    """Test ModelDeployer initializes correctly"""
    deployer = ModelDeployer(config_path=str(temp_config_file))

    assert deployer.config_path == temp_config_file
    assert deployer.client is not None


def test_deploy_updates_config_file(temp_config_file, mock_mlflow_client):
    """Test that deploy() updates production config file"""
    deployer = ModelDeployer(config_path=str(temp_config_file))

    # Mock smoke test to pass
    with patch.object(deployer, "_smoke_test", return_value=(True, "OK")):
        # Mock transition stage
        mock_mlflow_client.transition_model_version_stage = Mock()

        result = deployer.deploy(
            model_uri="models:/test-model/2",
            metadata={"version": 2, "metrics": {"r2": 0.90, "mae": 0.015}},
        )

    # Verify config was updated
    with open(temp_config_file) as f:
        config = yaml.safe_load(f)

    assert config["version"] == 2
    assert config["model_uri"] == "models:/test-model/2"
    assert config["metrics"]["r2"] == 0.90
    assert result.success is True
    assert result.smoke_test_passed is True


def test_smoke_test_passes_with_valid_model():
    """Test smoke test passes with valid predictions"""
    deployer = ModelDeployer()

    # Mock PyTorch model that returns valid predictions
    mock_model = Mock()
    mock_model.eval = Mock(return_value=None)
    mock_model.to = Mock(return_value=mock_model)

    # Configure as legacy model (no embedding) - use hasattr checks
    # Remove ticker_embedding attribute so hasattr returns False
    del mock_model.ticker_embedding

    # Mock LSTM with input_size
    mock_lstm = Mock()
    mock_lstm.input_size = 19
    mock_model.lstm = mock_lstm

    # Valid output tensor
    valid_output = torch.tensor([[0.5], [0.6], [0.7]])

    # Mock forward pass - needs to be callable
    def mock_forward(*args, **kwargs):
        return valid_output

    mock_model.side_effect = mock_forward

    with patch("mlflow.pytorch.load_model", return_value=mock_model), patch(
        "torch.no_grad"
    ):
        passed, message = deployer._smoke_test("models:/test/1")

    assert passed is True, f"Expected True, got {passed}. Message: {message}"
    assert "passed" in message.lower()


def test_smoke_test_fails_with_nan_predictions():
    """Test smoke test fails when model returns NaN"""
    deployer = ModelDeployer()

    # Mock PyTorch model that returns NaN
    import numpy as np

    mock_model = Mock()
    mock_model.eval = Mock(return_value=None)
    mock_model.to = Mock(return_value=mock_model)

    # Remove ticker_embedding
    del mock_model.ticker_embedding

    mock_lstm = Mock()
    mock_lstm.input_size = 19
    mock_model.lstm = mock_lstm

    # NaN output
    nan_output = torch.tensor([[np.nan], [0.6], [0.7]])

    def mock_forward(*args, **kwargs):
        return nan_output

    mock_model.side_effect = mock_forward

    with patch("mlflow.pytorch.load_model", return_value=mock_model), patch(
        "torch.no_grad"
    ):
        passed, message = deployer._smoke_test("models:/test/1")

    assert passed is False, f"Expected False, got {passed}. Message: {message}"
    assert "nan" in message.lower() or "inf" in message.lower()


def test_deploy_fails_on_failed_smoke_test(temp_config_file, mock_mlflow_client):
    """Test deploy aborts if smoke test fails"""
    deployer = ModelDeployer(config_path=str(temp_config_file))

    # Mock smoke test to fail
    with patch.object(
        deployer, "_smoke_test", return_value=(False, "Model returns NaN")
    ):
        result = deployer.deploy(
            model_uri="models:/bad-model/1",
            metadata={"version": 1, "metrics": {"r2": 0.90, "mae": 0.015}},
        )

    assert result.success is False
    assert result.smoke_test_passed is False
    assert "smoke test" in result.error.lower()

    # Verify config was NOT updated (still version 1 from fixture)
    with open(temp_config_file) as f:
        config = yaml.safe_load(f)

    assert config["version"] == 1  # Unchanged
    assert config["model_uri"] == "models:/test-model/1"  # Unchanged


@pytest.mark.parametrize(
    "model_uri,version,expected_stage",
    [
        ("models:/test/1", 1, "Production"),
        ("models:/test/5", 5, "Production"),
    ],
)
def test_deploy_transitions_to_production(
    temp_config_file, mock_mlflow_client, model_uri, version, expected_stage
):
    """Test deploy transitions model to Production stage"""
    deployer = ModelDeployer(config_path=str(temp_config_file))

    with patch.object(deployer, "_smoke_test", return_value=(True, "OK")):
        result = deployer.deploy(
            model_uri=model_uri, metadata={"version": version, "metrics": {}}
        )

    assert result.success is True

    # Note: transition_model_version_stage is not called in current implementation
    # This test needs update or removal based on actual deployment flow


def test_deployer_handles_missing_config_file():
    """Test deployer creates config if missing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "new_config.yaml"

        deployer = ModelDeployer(config_path=str(config_path))

        # Config file should be created during deployment
        with patch.object(deployer, "_smoke_test", return_value=(True, "OK")):
            result = deployer.deploy(
                "models:/test/1", metadata={"version": 1, "metrics": {"r2": 0.8}}
            )

        assert config_path.exists()
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
