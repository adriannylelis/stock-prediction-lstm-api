"""Integration tests for complete ML pipelines.

Tests end-to-end workflows including training and prediction.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.monitoring.drift_detector import DriftDetector
from src.ml.pipeline.predict_pipeline import PredictPipeline
from src.ml.pipeline.train_pipeline import TrainPipeline
from src.ml.utils.persistence import ArtifactManager, DataVersionManager


@pytest.fixture
def clean_artifacts():
    """Clean up artifacts before and after tests."""
    artifacts_dir = Path("artifacts_test")
    data_dir = Path("data_test")

    # Cleanup before
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)

    yield

    # Cleanup after
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.mark.integration
def test_complete_training_pipeline(clean_artifacts):
    """Test complete training pipeline from ingestion to model saving."""

    # Create pipeline
    pipeline = TrainPipeline(
        ticker="PETR4.SA",
        start_date="2023-01-01",
        end_date="2023-06-01",
        lookback=10,
        hidden_size=16,
        num_layers=1,
        epochs=2,  # Few epochs for speed
        batch_size=32,
        model_save_path="artifacts_test/best_model.pt",
        experiment_name=None,  # Disable MLflow for tests
    )

    # Run pipeline
    results = pipeline.run()

    # Assertions
    assert results is not None
    assert "model_path" in results
    assert "training_history" in results
    assert "test_metrics" in results
    assert "metadata" in results

    # Check model was saved
    assert Path(results["model_path"]).exists()

    # Check training history
    history = results["training_history"]
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 2  # 2 epochs

    # Check test metrics
    metrics = results["test_metrics"]
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert metrics["MAE"] > 0


@pytest.mark.integration
def test_training_and_prediction_pipeline(clean_artifacts):
    """Test complete workflow: train model, then make predictions."""

    # Step 1: Train model
    train_pipeline = TrainPipeline(
        ticker="PETR4.SA",
        start_date="2023-01-01",
        end_date="2023-06-01",
        lookback=10,
        hidden_size=16,
        epochs=2,
        model_save_path="artifacts_test/best_model.pt",
        experiment_name=None,
    )

    train_results = train_pipeline.run()
    assert Path(train_results["model_path"]).exists()

    # Step 2: Make predictions (PredictPipeline only accepts model_path, ticker, lookback)
    predict_pipeline = PredictPipeline(
        model_path=train_results["model_path"], ticker="PETR4.SA", lookback=10
    )

    predictions = predict_pipeline.predict(days_ahead=5)

    # Assertions
    assert predictions is not None
    assert isinstance(predictions, pd.DataFrame)
    assert "Date" in predictions.columns
    assert "Predicted_Close" in predictions.columns
    assert len(predictions) > 0
    assert predictions["Predicted_Close"].notna().all()


@pytest.mark.integration
def test_data_versioning_integration(clean_artifacts):
    """Test data versioning throughout pipeline."""

    manager = DataVersionManager(base_path="data_test/versioned", auto_cleanup=True, max_versions=3)

    # Create and save multiple versions
    versions = []
    for i in range(5):
        df = pd.DataFrame(
            {
                "Close": np.random.rand(100) + i,  # Different data each time
                "Volume": np.random.rand(100) * 1000,
            },
            index=pd.date_range("2023-01-01", periods=100),
        )

        version = manager.save(df, "TEST.SA", metadata={"iteration": i})
        versions.append(version)

    # Should only keep 3 versions (auto-cleanup)
    remaining_versions = manager.list_versions("TEST.SA")
    assert len(remaining_versions) <= 3

    # Load latest version
    latest_df = manager.load_latest("TEST.SA")
    assert latest_df is not None
    assert len(latest_df) == 100


@pytest.mark.integration
def test_drift_detection_integration(clean_artifacts):
    """Test drift detection with real pipeline data."""

    # Create reference data
    np.random.seed(42)
    ref_data = pd.DataFrame(
        {
            "Close": np.random.normal(30, 5, 500),
            "Volume": np.random.normal(1e6, 2e5, 500),
            "RSI": np.random.normal(50, 10, 500),
        }
    )

    # Create production data with drift
    prod_data = pd.DataFrame(
        {
            "Close": np.random.normal(35, 5, 500),  # Shifted mean
            "Volume": np.random.normal(1.2e6, 2e5, 500),  # Shifted mean
            "RSI": np.random.normal(50, 10, 500),  # No drift
        }
    )

    # Detect drift
    detector = DriftDetector(ks_threshold=0.05, psi_threshold=0.1)

    # KS test
    ks_report = detector.detect_drift(ref_data, prod_data)
    assert ks_report["has_drift"]
    assert "Close" in ks_report["drifted_features"]
    assert "Volume" in ks_report["drifted_features"]
    assert "RSI" not in ks_report["drifted_features"]

    # PSI test
    psi_report = detector.detect_drift_psi(ref_data, prod_data)
    assert psi_report["has_drift"]
    assert psi_report["summary"]["max_psi"] > 0.1


@pytest.mark.integration
def test_artifact_management_integration(clean_artifacts):
    """Test artifact manager with scalers and configs."""

    manager = ArtifactManager(
        base_path="artifacts_test",
        use_mlflow=False,  # Disable MLflow for tests
    )

    # Save and load scaler
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit([[1, 2], [3, 4]])

    scaler_path = manager.save_scaler(scaler, "test_scaler", metadata={"ticker": "TEST.SA"})
    assert scaler_path.exists()

    loaded_scaler = manager.load_scaler("test_scaler")
    assert loaded_scaler is not None

    # Save and load config
    config = {"lookback": 60, "hidden_size": 50, "learning_rate": 0.001}

    config_path = manager.save_config(config, "test_config")
    assert config_path.exists()

    loaded_config = manager.load_config("test_config")
    assert loaded_config == config


@pytest.mark.integration
def test_full_retraining_workflow(clean_artifacts):
    """Test complete retraining workflow with drift detection."""

    # Initialize managers
    detector = DriftDetector()

    # Step 1: Train initial model
    pipeline_v1 = TrainPipeline(
        ticker="PETR4.SA",
        start_date="2023-01-01",
        end_date="2023-03-01",
        lookback=10,
        epochs=2,
        model_save_path="artifacts_test/v1/best_model.pt",
        experiment_name=None,
    )

    results_v1 = pipeline_v1.run()
    assert Path(results_v1["model_path"]).exists()

    # Step 2: Simulate new data and check for drift
    # (In real scenario, this would be actual new production data)
    np.random.seed(123)
    ref_data = pd.DataFrame(
        {"Close": np.random.normal(30, 5, 200), "Volume": np.random.normal(1e6, 2e5, 200)}
    )

    new_data = pd.DataFrame(
        {
            "Close": np.random.normal(33, 5, 200),  # Slight drift
            "Volume": np.random.normal(1.1e6, 2e5, 200),  # Slight drift
        }
    )

    drift_report = detector.detect_drift(ref_data, new_data)

    # Step 3: Retrain if drift detected
    if drift_report["has_drift"]:
        pipeline_v2 = TrainPipeline(
            ticker="PETR4.SA",
            start_date="2023-02-01",
            end_date="2023-04-01",
            lookback=10,
            epochs=2,
            model_save_path="artifacts_test/v2/best_model.pt",
            experiment_name=None,
        )

        results_v2 = pipeline_v2.run()
        assert Path(results_v2["model_path"]).exists()

        # Verify new model is different file
        assert results_v2["model_path"] != results_v1["model_path"]


@pytest.mark.integration
def test_pipeline_with_small_dataset():
    """Test pipeline handles small datasets gracefully."""

    pipeline = TrainPipeline(
        ticker="PETR4.SA",
        start_date="2023-06-01",
        end_date="2023-06-30",  # Only 1 month
        lookback=5,
        hidden_size=8,
        epochs=1,
        model_save_path="artifacts_test/model_small.pt",
        experiment_name=None,
    )

    results = pipeline.run()

    # Should complete without errors
    assert results is not None
    assert "test_metrics" in results
