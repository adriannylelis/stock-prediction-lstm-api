"""
Automated Promotion Pipeline.

Compares models and automatically promotes the best one to Production.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.mlops.deployment.model_deployer import ModelDeployer
from src.mlops.monitoring.model_comparator import ModelComparator


@dataclass
class PromotionResult:
    """Result of promotion pipeline."""

    promoted: bool
    deployed: bool
    new_version: Optional[int]
    old_version: Optional[int]
    new_model_uri: str
    old_model_uri: Optional[str]
    comparison: Optional[Dict]
    reason: str
    timestamp: str = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class AutoPromotionPipeline:
    """Automated model promotion pipeline.

    Compares new model against production, promotes if better based on criteria.

    Args:
        new_model_version: Version number of new model to evaluate
        model_name: MLflow model name (default: lstm-multi-ticker)
        production_model: Production model URI or "auto" to get from config
        auto_promote_criteria: Dict with thresholds for auto-promotion:
            - r2_improvement: Minimum R² improvement (default: 0.05 = 5%)
            - mae_improvement: Minimum MAE improvement (default: 0.02 = 2%)
            - min_r2: Minimum absolute R² (default: 0.15)
            - max_mae: Maximum absolute MAE (default: 0.04)
        auto_deploy: Whether to auto-deploy if promoted (default: True)
        tracking_uri: MLflow tracking URI

    Example:
        >>> pipeline = AutoPromotionPipeline(
        ...     new_model_version=5,
        ...     auto_deploy=True
        ... )
        >>> result = pipeline.run_with_deploy()
        >>> if result.promoted:
        ...     print(f"Model v{result.new_version} promoted!")
    """

    def __init__(
        self,
        new_model_version: int,
        model_name: str = "lstm-multi-ticker",
        production_model: str = "auto",
        auto_promote_criteria: dict = None,
        auto_deploy: bool = True,
        tracking_uri: str = None,
    ):
        self.new_model_version = new_model_version
        self.model_name = model_name
        self.auto_deploy = auto_deploy

        if tracking_uri is None:
            tracking_uri = f"file:{Path.cwd()}/data/mlflow/tracking"

        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

        default_criteria = {
            "r2_improvement": 0.05,  # 5% better
            "mae_improvement": 0.02,  # 2% better
            "min_r2": 0.15,  # Minimum R²
            "max_mae": 0.04,  # Maximum MAE
        }
        self.criteria = {**default_criteria, **(auto_promote_criteria or {})}

        # Resolve production model URI
        if production_model == "auto":
            self.production_model_uri = self._get_production_model_uri()
        else:
            self.production_model_uri = production_model

        self.new_model_uri = f"models:/{model_name}/{new_model_version}"

        logger.info("AutoPromotionPipeline initialized")
        logger.info(f"  New model: {self.new_model_uri}")
        logger.info(
            f"  Production: {self.production_model_uri or 'None (first deployment)'}"
        )

    def _get_production_model_uri(self) -> Optional[str]:
        """Get current production model from config or MLflow."""
        deployer = ModelDeployer(tracking_uri=self.tracking_uri)
        config_model = deployer.get_current_production_model()
        if config_model:
            return config_model

        try:
            versions = self.client.search_model_versions(
                f"name='{self.model_name}' AND tags.stage='Production'"
            )
            if versions:
                return f"models:/{self.model_name}/{versions[0].version}"
        except:
            pass

        return None

    def _should_promote(
        self, new_metrics: dict, prod_metrics: Optional[dict]
    ) -> tuple[bool, str]:
        """Determine if new model should be promoted.

        Args:
            new_metrics: Metrics from new model
            prod_metrics: Metrics from production model (None if no production)

        Returns:
            (should_promote, reason)
        """
        # If no production model, auto-promote (first deployment)
        # Note: Metrics might be unreliable if evaluated on random data
        if prod_metrics is None:
            return True, (
                f"First production deployment: Auto-promoted "
                f"(R²={new_metrics['r2']:.4f}, MAE={new_metrics['mae']:.4f})"
            )

        r2_current = prod_metrics["r2"]
        mae_current = prod_metrics["mae"]

        r2_new = new_metrics["r2"]
        mae_new = new_metrics["mae"]

        # Calculate improvements
        r2_improvement = (
            (r2_new - r2_current) / abs(r2_current) if r2_current != 0 else 0
        )
        mae_improvement = (
            (mae_current - mae_new) / mae_current if mae_current != 0 else 0
        )

        r2_better = r2_improvement >= self.criteria["r2_improvement"]
        mae_better = mae_improvement >= self.criteria["mae_improvement"]

        r2_acceptable = r2_new >= self.criteria["min_r2"]
        mae_acceptable = mae_new <= self.criteria["max_mae"]

        if r2_better and mae_better and r2_acceptable and mae_acceptable:
            return True, (
                f"Model improved: "
                f"R² {r2_improvement*100:+.1f}% (>={self.criteria['r2_improvement']*100}%), "
                f"MAE {mae_improvement*100:+.1f}% (>={self.criteria['mae_improvement']*100}%)"
            )
        else:
            reasons = []
            if not r2_better:
                reasons.append(
                    f"R² improvement {r2_improvement*100:.1f}% < {self.criteria['r2_improvement']*100}%"
                )
            if not mae_better:
                reasons.append(
                    f"MAE improvement {mae_improvement*100:.1f}% < {self.criteria['mae_improvement']*100}%"
                )
            if not r2_acceptable:
                reasons.append(f"R² {r2_new:.4f} < {self.criteria['min_r2']}")
            if not mae_acceptable:
                reasons.append(f"MAE {mae_new:.4f} > {self.criteria['max_mae']}")

            return False, "Not promoted: " + "; ".join(reasons)

    def _promote_model(self) -> bool:
        """Promote model to Production stage in MLflow.

        Returns:
            True if successful
        """
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=self.new_model_version,
                stage="Production",
                archive_existing_versions=True,  # Archive old Production versions
            )
            logger.success(
                f"✅ Model v{self.new_model_version} promoted to Production stage"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to promote model: {e}")
            return False

    def run_with_deploy(self) -> PromotionResult:
        """Run promotion pipeline with optional deployment.

        Returns:
            PromotionResult with promotion decision and deployment status
        """
        logger.info("=" * 80)
        logger.info("🎯 Auto Promotion Pipeline Started")
        logger.info("=" * 80)
        logger.info(f"New model: {self.new_model_uri}")
        logger.info(f"Production: {self.production_model_uri or 'None'}")
        logger.info(f"Criteria: {self.criteria}")
        logger.info("=" * 80)

        try:
            # 1. Compare models
            logger.info("\n📊 Step 1/3: Comparing models...")
            comparator = ModelComparator(tracking_uri=self.tracking_uri)

            if self.production_model_uri:
                model_uris = [self.new_model_uri, self.production_model_uri]
            else:
                model_uris = [self.new_model_uri]

            comparison_report = comparator.compare(model_uris)
            comparison_report.print_report()

            # Extract metrics
            new_model_metrics = comparison_report.models[0]
            prod_model_metrics = (
                comparison_report.models[1]
                if len(comparison_report.models) > 1
                else None
            )

            new_metrics = {
                "mae": new_model_metrics.mae,
                "rmse": new_model_metrics.rmse,
                "r2": new_model_metrics.r2,
                "mape": new_model_metrics.mape,
            }

            prod_metrics = (
                {
                    "mae": prod_model_metrics.mae,
                    "rmse": prod_model_metrics.rmse,
                    "r2": prod_model_metrics.r2,
                    "mape": prod_model_metrics.mape,
                }
                if prod_model_metrics
                else None
            )

            # 2. Decide if should promote
            logger.info("\n🤔 Step 2/3: Evaluating promotion criteria...")
            should_promote, reason = self._should_promote(new_metrics, prod_metrics)

            logger.info(f"Decision: {reason}")

            if not should_promote:
                logger.info("\n⏭️  Model not promoted - current model still better")
                return PromotionResult(
                    promoted=False,
                    deployed=False,
                    new_version=self.new_model_version,
                    old_version=(
                        prod_model_metrics.version if prod_model_metrics else None
                    ),
                    new_model_uri=self.new_model_uri,
                    old_model_uri=self.production_model_uri,
                    comparison={"new": new_metrics, "production": prod_metrics},
                    reason=reason,
                )

            # 3. Promote
            logger.info("\n🚀 Step 3/3: Promoting model...")
            promote_success = self._promote_model()

            if not promote_success:
                return PromotionResult(
                    promoted=False,
                    deployed=False,
                    new_version=self.new_model_version,
                    old_version=(
                        prod_model_metrics.version if prod_model_metrics else None
                    ),
                    new_model_uri=self.new_model_uri,
                    old_model_uri=self.production_model_uri,
                    comparison={"new": new_metrics, "production": prod_metrics},
                    reason="Promotion failed",
                    error="Failed to update MLflow stage",
                )

            # 4. Deploy (if auto_deploy)
            deployed = False
            if self.auto_deploy:
                logger.info("\n📦 Deploying to production...")
                deployer = ModelDeployer(tracking_uri=self.tracking_uri)
                deploy_result = deployer.deploy(
                    self.new_model_uri,
                    metadata={
                        "version": self.new_model_version,
                        "metrics": new_metrics,
                        "promoted_from": "Staging",
                        "reason": reason,
                    },
                )
                deployed = deploy_result.success

                if not deployed:
                    logger.warning(
                        f"⚠️ Promotion succeeded but deployment failed: {deploy_result.error}"
                    )

            logger.success("\n" + "=" * 80)
            logger.success("✅ Promotion Pipeline Completed!")
            logger.info("Promoted: Yes")
            logger.info(f"Deployed: {deployed}")
            logger.info(f"New version: {self.new_model_version}")
            logger.success("=" * 80)

            return PromotionResult(
                promoted=True,
                deployed=deployed,
                new_version=self.new_model_version,
                old_version=prod_model_metrics.version if prod_model_metrics else None,
                new_model_uri=self.new_model_uri,
                old_model_uri=self.production_model_uri,
                comparison={
                    "new": new_metrics,
                    "production": prod_metrics,
                    "improvements": (
                        {
                            "r2": (
                                (
                                    (new_metrics["r2"] - prod_metrics["r2"])
                                    / abs(prod_metrics["r2"])
                                    * 100
                                )
                                if prod_metrics
                                else 0
                            ),
                            "mae": (
                                (
                                    (prod_metrics["mae"] - new_metrics["mae"])
                                    / prod_metrics["mae"]
                                    * 100
                                )
                                if prod_metrics
                                else 0
                            ),
                        }
                        if prod_metrics
                        else {}
                    ),
                },
                reason=reason,
            )

        except Exception as e:
            logger.error(f"\n❌ Promotion pipeline failed: {str(e)}")
            logger.exception(e)
            return PromotionResult(
                promoted=False,
                deployed=False,
                new_version=self.new_model_version,
                old_version=None,
                new_model_uri=self.new_model_uri,
                old_model_uri=self.production_model_uri,
                comparison=None,
                reason="Pipeline error",
                error=str(e),
            )


if __name__ == "__main__":
    # Test promotion pipeline
    # pipeline = AutoPromotionPipeline(
    #     new_model_version=5,
    #     auto_deploy=True
    # result = pipeline.run_with_deploy()
    # print(f"Promoted: {result.promoted}, Deployed: {result.deployed}")

    logger.info("Promotion Pipeline module loaded successfully")
