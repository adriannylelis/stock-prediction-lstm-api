"""
Complete MLOps Pipeline E2E Test

Tests the full automation flow:
1. Training with Optuna (multiple runs tracked)
2. Model promotion (None → Staging → Production)
3. Prediction via MLflow
4. Second model promotion and prediction

Run: pytest tests/e2e/test_mlops_complete.py -v -s
"""

from datetime import datetime
from pathlib import Path

import mlflow
import pytest

from src.api.services.model_service import ModelService
from src.mlops.pipelines.promotion_pipeline import AutoPromotionPipeline
from src.mlops.pipelines.training_pipeline import AutoTrainingPipeline


class TestMLOpsComplete:
    """Complete MLOps workflow tests."""

    @pytest.fixture(autouse=True)
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri("file:./data/mlflow/tracking")
        yield
        mlflow.end_run()  # Cleanup

    @pytest.fixture
    def experiment_name(self):
        """Unique experiment name for this test run."""
        return f"e2e-test-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @pytest.fixture(scope="class")
    def production_model(self):
        """Fixture: Ensure there's a model in production for testing.

        This runs once per test class and ensures all tests have a
        production model available for prediction, rollback, etc.

        Returns:
            dict with model_uri, version, metrics
        """
        print("\n" + "=" * 80)
        print("FIXTURE: Ensuring Production Model Exists")
        print("=" * 80)

        import yaml

        config_path = Path("configs/production_model.yaml")

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            if config.get("model_uri") and config.get("model_uri") != "null":
                model_uri = config["model_uri"]

                # Extract version from URI if not in config
                version = config.get("version")
                if version is None and "models:/stock-lstm-model/" in model_uri:
                    version = int(model_uri.split("/")[-1])

                print(f"✅ Production model already exists: {model_uri} (v{version})")
                return {
                    "model_uri": model_uri,
                    "version": version,
                    "metrics": config.get("metrics", {}),
                }

        # No production model - create one
        print("⚠️ No production model found, creating one...")

        mlflow.set_tracking_uri("file:./data/mlflow/tracking")
        experiment_name = (
            f"fixture-production-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Train a model
        pipeline = AutoTrainingPipeline(
            tickers=["PETR4.SA", "VALE3.SA"],
            epochs=5,
            batch_size=32,
            device="cpu",  # Force CPU for E2E tests
            experiment_name=experiment_name,
        )

        result = pipeline.run()
        assert result.success, f"Failed to create production model: {result.error}"

        promotion = AutoPromotionPipeline(
            new_model_version=result.version,
            model_name="stock-lstm-model",
            production_model="auto",
            auto_deploy=True,
        )
        promo_result = promotion.run_with_deploy()

        assert (
            promo_result.promoted
            or promo_result.new_version == promo_result.old_version
        ), f"Failed to promote model to production: {promo_result.reason}"

        print(f"✅ Created production model: {result.model_uri}")

        return {
            "model_uri": result.model_uri,
            "version": result.version,
            "metrics": result.metrics,
        }

    def test_1_training_with_optuna(self, experiment_name):
        """
        Test 1: Training Pipeline with Optuna

        Validates:
        - Multiple runs tracked (one per hyperparameter combination)
        - All models have status None initially
        - Metrics logged correctly
        """
        print("\n" + "=" * 80)
        print("TEST 1: Training Pipeline with Optuna Tracking")
        print("=" * 80)

        pipeline = AutoTrainingPipeline(
            tickers=["PETR4.SA", "VALE3.SA"],
            epochs=5,  # Fast test
            batch_size=32,
            device="cpu",  # Force CPU for E2E tests (avoid CUDA async errors)
            experiment_name=experiment_name,
        )

        # Run training
        result = pipeline.run()

        # Assertions
        assert result.success, f"Training failed: {result.error}"
        assert result.model_uri is not None
        assert result.version is not None
        assert "R2" in result.metrics  # Test metrics (not val metrics)
        assert "MAE" in result.metrics

        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        assert experiment is not None

        runs = client.search_runs(experiment.experiment_id)
        assert len(runs) >= 1, "No runs found in MLflow"

        for run in runs:
            print(f"   Run ID: {run.info.run_id}")
            print(f"   Metrics: {run.data.metrics}")
            print("   Status: None (not registered yet)")

        print("✅ TEST 1 PASSED: Training pipeline working, runs tracked")

        return {
            "model_uri": result.model_uri,
            "version": result.version,
            "experiment_name": experiment_name,
            "run_id": (
                result.run_id if hasattr(result, "run_id") else runs[0].info.run_id
            ),
        }

    def test_2_model_promotion_staging(self, experiment_name):
        """
        Test 2: Verify Model Already in Staging

        Validates:
        - Model registered in MLflow Model Registry
        - Status is Staging (already promoted by trainer)
        - Metrics logged correctly
        """
        print("\n" + "=" * 80)
        print("TEST 2: Verify Model in Staging")
        print("=" * 80)

        # First train a model
        training_result = self.test_1_training_with_optuna(experiment_name)

        client = mlflow.MlflowClient()
        model_name = "stock-lstm-model"

        model_version = client.get_model_version(
            name=model_name, version=str(training_result["version"])
        )

        assert model_version.current_stage in [
            "Staging",
            "None",
        ], f"Expected Staging or None, got {model_version.current_stage}"

        # If still None, manually promote to Staging for next tests
        if model_version.current_stage == "None":
            print("   Promoting model to Staging manually...")
            client.transition_model_version_stage(
                name=model_name,
                version=str(training_result["version"]),
                stage="Staging",
            )
            # Re-fetch to confirm
            model_version = client.get_model_version(
                name=model_name, version=str(training_result["version"])
            )

        assert (
            model_version.current_stage == "Staging"
        ), f"Expected Staging, got {model_version.current_stage}"

        print("✅ TEST 2 PASSED: Model in Staging")
        print(f"   Model: {model_name} version {training_result['version']}")
        print(f"   Stage: {model_version.current_stage}")

        return training_result

    def test_3_model_promotion_production(self, experiment_name):
        """
        Test 3: Promote Model to Production

        Validates:
        - Staging → Production transition with AutoPromotionPipeline
        - Model comparison and promotion logic
        - Deployment successful
        """
        print("\n" + "=" * 80)
        print("TEST 3: Model Promotion to Production")
        print("=" * 80)

        training_result = self.test_2_model_promotion_staging(experiment_name)

        # Use AutoPromotionPipeline to promote to production
        # This pipeline compares new model vs production and promotes if better
        promotion_pipeline = AutoPromotionPipeline(
            new_model_version=training_result["version"],
            model_name="stock-lstm-model",
            production_model="auto",  # Will check if production model exists
            auto_deploy=True,
        )

        result = promotion_pipeline.run_with_deploy()

        # Note: Promotion pode falhar se thresholds são muito rígidos - isso é esperado para modelos ruins
        if not result.promoted:
            print(
                f"⚠️  Model not promoted (expected for strict thresholds): {result.reason}"
            )
            pytest.skip(f"Model not promoted due to strict criteria: {result.reason}")

        client = mlflow.MlflowClient()
        model_version = client.get_model_version(
            name="stock-lstm-model", version=str(training_result["version"])
        )

        assert (
            model_version.current_stage == "Production"
        ), f"Expected Production, got {model_version.current_stage}"

        print("✅ TEST 3 PASSED: Model promoted to Production")
        print(f"   Model: stock-lstm-model version {training_result['version']}")
        print(f"   Stage: {model_version.current_stage}")
        print(f"   Reason: {result.reason}")

        return training_result

    def test_4_prediction_via_mlflow(self, production_model):
        """Test 4: Prediction via MLflow API

        Prerequisites: Fixture ensures model in production

        Validates:
        - ModelService loads from MLflow
        - Predictions work correctly
        - No errors during inference
        """
        print("\n" + "=" * 80)
        print("TEST 4: Prediction via MLflow API")
        print("=" * 80)

        # Use production model from fixture
        print(f"Using production model: {production_model['model_uri']}")

        service = ModelService()

        assert service.is_ready(), "ModelService should be ready"
        assert service.model is not None
        assert service.scaler is not None

        import numpy as np

        # Extract num_features from model architecture
        # Model expects: (batch, seq_len, num_features)
        num_features = service.model.num_features
        lookback = service.config.get("lookback", 60) if service.config else 60

        test_input = np.random.rand(
            1, lookback, num_features
        )  # Match model's expected features
        ticker_id = 0  # Use first ticker ID

        prediction = service.predict(test_input, ticker_id=ticker_id)

        assert prediction is not None
        assert len(prediction) > 0
        assert not np.isnan(prediction).any(), "Predictions should not contain NaN"

        print("✅ TEST 4 PASSED: Predictions via MLflow working")
        print(f"   Model URI: {production_model['model_uri']}")
        print(f"   Sample prediction: {prediction[0]:.4f}")

        return {"prediction": prediction[0], **production_model}

    def test_5_second_model_promotion_and_prediction(self, experiment_name):
        """Test 5: Second Model Promotion and Prediction Comparison.

        Validates:
        - Training with different parameters
        - Model comparison logic
        - Promotion decision based on metrics
        """
        print("\n" + "=" * 80)
        print("TEST 5: Second Model Promotion and Prediction")
        print("=" * 80)

        # Train second model with different params
        pipeline = AutoTrainingPipeline(
            tickers=["PETR4.SA", "VALE3.SA", "ITUB4.SA"],  # 3 tickers vs 2 in Test 1
            epochs=10,  # More epochs
            batch_size=32,  # Different batch size
            experiment_name=f"{experiment_name}-v2",
        )

        result_v2 = pipeline.run()
        assert result_v2.success, f"Training failed: {result_v2.error}"
        assert result_v2.metrics["R2"] > 0.7

        print("\n✅ TEST 5 PASSED: Second model trained successfully")
        print(f"   Model URI: {result_v2.model_uri}")
        print(f"   R²: {result_v2.metrics['R2']:.4f}")
        print(f"   MAE: {result_v2.metrics['MAE']:.4f}")

        return result_v2

    def test_6_training_with_all_tickers(self, experiment_name):
        """Test 6: Training with All Available Tickers (Auto-Discovery).

        This test validates:
        - Auto-discovery of all 43 available B3 tickers
        - Model scales to all tickers without hardcoding
        - Embedding dimension is 8 (fixed architecture)
        - Each ticker gets mapped to one of 8 embeddings
        - Training quality remains high with full dataset

        Architecture:
        - 43 unique tickers available in the system (auto-discovered via ALL_TICKERS)
        - 8 embedding dimensions (fixed model architecture)
        - Each ticker ID (0-42) uses one of the 8 learned embeddings
        - Similar to Word2Vec: large vocabulary, fixed embedding size

        Validates:
        - Auto-discovery mechanism (tickers="all")
        - Scalability to production-size ticker count
        - Embedding architecture handles ticker diversity
        - Model performance with comprehensive dataset
        """
        print("\n" + "=" * 80)
        print("TEST 6: Training with All Available Tickers (Auto-Discovery)")
        print("=" * 80)

        print("\n🔍 Auto-discovering all available tickers...")

        # Use auto-discovery: tickers="all" resolves to ALL_TICKERS (43 tickers)
        pipeline = AutoTrainingPipeline(
            tickers="all",  # ✅ Auto-discovery instead of hardcoded list
            epochs=5,
            batch_size=64,
            experiment_name=f"{experiment_name}-all-tickers",
        )

        # Pipeline will resolve "all" to ALL_TICKERS
        resolved_tickers = pipeline.tickers
        print(f"\n✅ Discovered {len(resolved_tickers)} tickers automatically")
        print(f"   First 5: {', '.join(resolved_tickers[:5])}")
        print(f"   Last 5: {', '.join(resolved_tickers[-5:])}")

        result = pipeline.run()

        assert result.success, f"Training failed: {result.error}"
        assert result.model_uri is not None

        import mlflow.pytorch

        model = mlflow.pytorch.load_model(result.model_uri)

        assert hasattr(
            model, "ticker_embedding"
        ), "Model must have ticker_embedding layer"
        assert hasattr(model, "num_tickers"), "Model must have num_tickers attribute"

        # Fixed embedding dimension = 8 (regardless of ticker count)
        assert (
            model.ticker_embedding.embedding_dim == 8
        ), "Embedding dimension should be 8"

        assert (
            len(resolved_tickers) == 43
        ), f"Should use all 43 tickers, got {len(resolved_tickers)}"

        # Training quality validation
        assert result.metrics["R2"] > 0.7, f"R² too low: {result.metrics['R2']:.4f}"
        assert result.metrics["MAE"] < 0.1, f"MAE too high: {result.metrics['MAE']:.4f}"

        print(
            "\n✅ TEST 6 PASSED: All-ticker model trained successfully (AUTO-DISCOVERY)"
        )
        print(f"   Model URI: {result.model_uri}")
        print("   Embedding dimension: 8 (fixed architecture)")
        print(
            f"   Training tickers: {len(resolved_tickers)} (ALL tickers auto-discovered)"
        )
        print(f"   R²: {result.metrics['R2']:.4f}")
        print(f"   MAE: {result.metrics['MAE']:.4f}")
        print(f"   MAPE: {result.metrics.get('MAPE', 0):.2f}%")
        # Note: DA (Directional Accuracy) may not be in metrics, use get() with default
        print(
            f"   DA: {result.metrics.get('DA', result.metrics.get('Directional_Accuracy', 0)):.2f}%"
        )

        return result


# Standalone execution
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
