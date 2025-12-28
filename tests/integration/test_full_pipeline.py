"""Integration tests for complete ML pipeline."""

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
def temp_artifacts_dir(tmp_path):
    """Create temporary artifacts directory."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    yield str(artifacts_dir)
    # Cleanup
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yield str(data_dir)
    # Cleanup
    if data_dir.exists():
        shutil.rmtree(data_dir)


class TestFullPipeline:
    """Test complete end-to-end pipeline."""

    def test_train_pipeline_end_to_end(self, temp_artifacts_dir):
        """Test complete training pipeline from data ingestion to model save."""
        # Create pipeline
        pipeline = TrainPipeline(
            ticker="PETR4.SA",
            start_date="2023-01-01",
            end_date="2023-12-31",
            lookback=30,
            hidden_size=32,
            num_layers=1,
            epochs=2,  # Reduced for testing
            batch_size=32,
            model_save_path=f"{temp_artifacts_dir}/best_model.pt",
            experiment_name="test_experiment",
        )

        # Run pipeline
        results = pipeline.run()

        # Assertions
        assert "model_path" in results
        assert "training_history" in results
        assert "test_metrics" in results
        assert "metadata" in results

        # Check model was saved (TrainPipeline saves as best_model.pt)
        assert Path(results["model_path"]).exists()

        # Check metrics (keys are uppercase: MAE, RMSE, MAPE, R2)
        assert "RMSE" in results["test_metrics"]
        assert "MAE" in results["test_metrics"]
        assert "MAPE" in results["test_metrics"]
        assert "R2" in results["test_metrics"]

        # Check training history
        assert len(results["training_history"]["train_loss"]) == 2
        assert len(results["training_history"]["val_loss"]) == 2

    def test_predict_pipeline_end_to_end(self, temp_artifacts_dir):
        """Test complete prediction pipeline."""
        # First train a model
        train_pipeline = TrainPipeline(
            ticker="PETR4.SA",
            start_date="2023-01-01",
            end_date="2023-12-31",
            lookback=30,
            hidden_size=32,
            epochs=2,
            model_save_path=f"{temp_artifacts_dir}/pred_test_model.pt",
        )
        train_results = train_pipeline.run()

        # Create prediction pipeline (PredictPipeline only accepts model_path, ticker, lookback)
        predict_pipeline = PredictPipeline(
            model_path=train_results["model_path"], ticker="PETR4.SA", lookback=30
        )

        # Run predictions
        predictions_df = predict_pipeline.predict(days_ahead=5)

        # Assertions
        assert isinstance(predictions_df, pd.DataFrame)
        assert "Date" in predictions_df.columns
        assert "Predicted_Close" in predictions_df.columns
        assert len(predictions_df) > 0
        assert predictions_df["Predicted_Close"].notna().all()

    def test_data_versioning_workflow(self, temp_data_dir):
        """Test data versioning and loading workflow."""
        manager = DataVersionManager(base_path=temp_data_dir, auto_cleanup=True, max_versions=3)

        # Create and save test data
        test_data = pd.DataFrame(
            {"Close": np.random.random(100), "Volume": np.random.randint(1000, 10000, 100)}
        )

        # Save multiple versions
        versions = []
        for i in range(5):
            version = manager.save(test_data, ticker="TEST.SA", metadata={"iteration": i})
            versions.append(version)

        # Check auto-cleanup (should keep only 3)
        remaining = manager.list_versions("TEST.SA")
        assert len(remaining) <= 3

        # Load latest
        loaded_df = manager.load_latest("TEST.SA")
        assert len(loaded_df) == 100
        assert "Close" in loaded_df.columns

    def test_drift_detection_workflow(self):
        """Test drift detection workflow."""
        detector = DriftDetector(ks_threshold=0.05, psi_threshold=0.1)

        # Create reference and production data
        np.random.seed(42)
        ref_data = pd.DataFrame(
            {"feature1": np.random.normal(0, 1, 1000), "feature2": np.random.normal(5, 2, 1000)}
        )

        # Production data with drift
        prod_data = pd.DataFrame(
            {
                "feature1": np.random.normal(1, 1, 1000),  # Mean shifted
                "feature2": np.random.normal(5, 2, 1000),  # No drift
            }
        )

        # Detect drift using KS
        ks_report = detector.detect_drift(ref_data, prod_data)
        assert ks_report["has_drift"] is True
        assert "feature1" in ks_report["drifted_features"]

        # Detect drift using PSI
        psi_report = detector.detect_drift_psi(ref_data, prod_data)
        assert isinstance(psi_report["feature_psi"], dict)
        assert "feature1" in psi_report["feature_psi"]

    def test_artifact_manager_workflow(self, temp_artifacts_dir):
        """Test artifact management workflow."""
        manager = ArtifactManager(
            base_path=temp_artifacts_dir,
            use_mlflow=False,  # Disable MLflow for testing
        )

        # Test config save/load
        config = {"model_type": "LSTM", "hidden_size": 50, "learning_rate": 0.001}
        manager.save_config(config, "test_config")
        loaded_config = manager.load_config("test_config")
        assert loaded_config == config

        # Test scaler save/load
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaler.fit([[1], [2], [3], [4], [5]])

        manager.save_scaler(scaler, "test_scaler")
        loaded_scaler = manager.load_scaler("test_scaler")

        # Test scaler works
        test_val = loaded_scaler.transform([[3]])
        assert test_val[0][0] == pytest.approx(0.5, abs=0.01)


class TestPipelineIntegration:
    """Test integration between different pipeline components."""

    def test_train_and_predict_integration(self, temp_artifacts_dir):
        """Test training and prediction work together."""
        # Train
        train_pipeline = TrainPipeline(
            ticker="VALE3.SA",
            start_date="2023-01-01",
            end_date="2023-06-30",
            lookback=20,
            hidden_size=16,
            epochs=1,
            model_save_path=f"{temp_artifacts_dir}/integration_model.pt",
        )
        train_results = train_pipeline.run()

        # Verify scaler was saved
        scaler_path = Path(train_results["model_path"]).parent / "scaler.pkl"
        assert scaler_path.exists()

        # Predict using trained model (PredictPipeline only accepts model_path, ticker, lookback)
        predict_pipeline = PredictPipeline(
            model_path=train_results["model_path"], ticker="VALE3.SA", lookback=20
        )
        predictions = predict_pipeline.predict(days_ahead=3)

        # Verify predictions
        assert len(predictions) > 0
        assert predictions["Predicted_Close"].dtype in [np.float64, np.float32]

    def test_versioning_and_drift_integration(self, temp_data_dir):
        """Test data versioning with drift detection."""
        manager = DataVersionManager(base_path=temp_data_dir)
        detector = DriftDetector()

        # Save version 1
        data_v1 = pd.DataFrame(
            {"Close": np.random.normal(30, 5, 500), "Volume": np.random.normal(1e6, 2e5, 500)}
        )
        version1 = manager.save(data_v1, "DRIFT.SA", metadata={"version": 1})

        # Save version 2 with drift
        data_v2 = pd.DataFrame(
            {
                "Close": np.random.normal(35, 5, 500),  # Drifted
                "Volume": np.random.normal(1e6, 2e5, 500),
            }
        )
        version2 = manager.save(data_v2, "DRIFT.SA", metadata={"version": 2})

        # Load both versions
        df_v1 = manager.load(version1, "DRIFT.SA")
        df_v2 = manager.load(version2, "DRIFT.SA")

        # Detect drift
        drift_report = detector.detect_drift(df_v1, df_v2)

        # Should detect drift in Close
        assert drift_report["has_drift"] is True
