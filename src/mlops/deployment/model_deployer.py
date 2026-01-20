"""
Model Deployer - Deploy models to production.

Updates production configuration and performs validation.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import torch
import yaml
from loguru import logger
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class DeployResult:
    """Result of model deployment."""

    success: bool
    model_uri: str
    deployed_at: str
    deployed_by: str = "auto_pipeline"
    smoke_test_passed: bool = False
    error: Optional[str] = None

    def __post_init__(self):
        if self.deployed_at is None:
            self.deployed_at = datetime.now().isoformat()


class ModelDeployer:
    """Deploy models to production.

    Updates production config file that API reads on startup.

    Args:
        config_path: Path to production config file
        tracking_uri: MLflow tracking URI

    Example:
        >>> deployer = ModelDeployer()
        >>> result = deployer.deploy("models:/lstm-multi-ticker/5")
        >>> if result.success:
        >>>     print("Model deployed!")
    """

    def __init__(
        self,
        config_path: str = "configs/production_model.yaml",
        tracking_uri: str = None,
    ):
        self.config_path = Path(config_path)

        if tracking_uri is None:
            tracking_uri = f"file:{Path.cwd()}/data/mlflow/tracking"

        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

        logger.info("ModelDeployer initialized")
        logger.info(f"  Config: {self.config_path}")
        logger.info(f"  Tracking: {self.tracking_uri}")

    def _smoke_test(self, model_uri: str) -> tuple[bool, str]:
        """Run smoke test on model.

        Args:
            model_uri: MLflow model URI

        Returns:
            (success, message)
        """
        try:
            logger.info("🧪 Running smoke test...")

            # Load model
            model = mlflow.pytorch.load_model(model_uri)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

            # Test with dummy input
            # Assuming model expects (batch_size, seq_len, features)
            # Adjust based on your model architecture
            batch_size = 3
            seq_len = 60

            # Determine input size based on model architecture
            if hasattr(model, "ticker_embedding"):
                # Embedding-based model: use feature size before embedding concat
                # LSTM input_size = feature_size + embedding_dim
                # So feature_size = lstm_input_size - embedding_dim
                embedding_dim = (
                    model.embedding_dim if hasattr(model, "embedding_dim") else 8
                )
                lstm_input_size = (
                    model.lstm.input_size if hasattr(model, "lstm") else 27
                )
                input_size = lstm_input_size - embedding_dim
                logger.info(
                    f"Embedding model: using {input_size} features (LSTM expects {lstm_input_size} = {input_size} + {embedding_dim})"
                )
            else:
                # Legacy model without embedding
                input_size = model.lstm.input_size if hasattr(model, "lstm") else 11
                logger.info(f"Legacy model: using {input_size} features")

            dummy_input = torch.randn(batch_size, seq_len, input_size).to(device)

            # Run inference
            with torch.no_grad():
                # Check if model requires ticker_ids (embedding-based)
                if hasattr(model, "ticker_embedding"):
                    num_tickers = (
                        model.num_tickers if hasattr(model, "num_tickers") else 8
                    )
                    ticker_ids = torch.randint(0, num_tickers, (batch_size,)).to(device)
                    output = model(dummy_input, ticker_ids)
                else:
                    output = model(dummy_input)

            # Validate output
            if output is None:
                return False, "Model returned None"

            if torch.isnan(output).any():
                return False, "Model output contains NaN"

            if torch.isinf(output).any():
                return False, "Model output contains Inf"

            if output.shape[0] != batch_size:
                return False, f"Unexpected output shape: {output.shape}"

            logger.success(f"✅ Smoke test passed! Output shape: {output.shape}")
            return True, "Smoke test passed"

        except Exception as e:
            logger.error(f"❌ Smoke test failed: {str(e)}")
            return False, f"Smoke test error: {str(e)}"

    def _update_config(self, model_uri: str, metadata: dict = None) -> None:
        """Update production config file.

        Args:
            model_uri: MLflow model URI
            metadata: Additional metadata to save
        """
        # Extract version from URI (e.g., "models:/stock-lstm-model/26" -> 26)
        version = None
        if "models:/" in model_uri and "/" in model_uri:
            try:
                version = int(model_uri.split("/")[-1])
            except (ValueError, IndexError):
                logger.warning(f"Could not extract version from URI: {model_uri}")

        config = {
            "model_uri": model_uri,
            "version": version,
            "deployed_at": datetime.now().isoformat(),
            "deployed_by": "auto_pipeline",
            "tracking_uri": self.tracking_uri,
        }

        if metadata:
            config.update(metadata)

        # Create parent directory if doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write config
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.success(f"✅ Config updated: {self.config_path} (v{version})")

    def get_current_production_model(self) -> Optional[str]:
        """Get currently deployed model URI.

        Returns:
            Model URI or None if no model deployed
        """
        if not self.config_path.exists():
            return None

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            return config.get("model_uri")
        except Exception as e:
            logger.warning(f"Could not read production config: {e}")
            return None

    def deploy(
        self, model_uri: str, run_smoke_test: bool = True, metadata: dict = None
    ) -> DeployResult:
        """Deploy model to production.

        Args:
            model_uri: MLflow model URI to deploy
            run_smoke_test: Whether to run smoke test before deploying
            metadata: Additional metadata to save in config

        Returns:
            DeployResult with deployment status
        """
        logger.info("=" * 80)
        logger.info("🚀 Deploying Model to Production")
        logger.info("=" * 80)
        logger.info(f"Model URI: {model_uri}")

        try:
            # 1. Smoke test (pre-deploy)
            smoke_test_passed = False
            if run_smoke_test:
                is_valid, message = self._smoke_test(model_uri)
                if not is_valid:
                    return DeployResult(
                        success=False,
                        model_uri=model_uri,
                        deployed_at=datetime.now().isoformat(),
                        smoke_test_passed=False,
                        error=f"Pre-deploy smoke test failed: {message}",
                    )
                smoke_test_passed = True

            # 2. Update config
            logger.info("\n📝 Updating production config...")
            self._update_config(model_uri, metadata)

            # 3. Post-deploy validation
            logger.info("\n✅ Verifying deployment...")
            current_model = self.get_current_production_model()
            if current_model != model_uri:
                return DeployResult(
                    success=False,
                    model_uri=model_uri,
                    deployed_at=datetime.now().isoformat(),
                    smoke_test_passed=smoke_test_passed,
                    error="Config verification failed - model URI mismatch",
                )

            logger.success("\n" + "=" * 80)
            logger.success("✅ Model Successfully Deployed to Production!")
            logger.info(f"Model URI: {model_uri}")
            logger.info(f"Config: {self.config_path}")
            logger.success("=" * 80)

            return DeployResult(
                success=True,
                model_uri=model_uri,
                deployed_at=datetime.now().isoformat(),
                smoke_test_passed=smoke_test_passed,
            )

        except Exception as e:
            logger.error(f"\n❌ Deployment failed: {str(e)}")
            logger.exception(e)
            return DeployResult(
                success=False,
                model_uri=model_uri,
                deployed_at=datetime.now().isoformat(),
                error=str(e),
            )


if __name__ == "__main__":
    # Test deployer
    deployer = ModelDeployer()

    # Example: Deploy a model
    # result = deployer.deploy("models:/lstm-multi-ticker/5")
    # if result.success:
    #     print("✅ Deployment successful!")
    # else:
    #     print(f"❌ Deployment failed: {result.error}")

    logger.info("Model Deployer module loaded successfully")
