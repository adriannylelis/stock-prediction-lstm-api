"""
Model Comparator - Compare multiple models side by side.

Provides detailed comparison reports for model selection and A/B testing.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlflow
import numpy as np
import torch
from loguru import logger
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ml.training.metrics import calculate_all_metrics


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    model_uri: str
    version: Optional[int]
    mae: float
    rmse: float
    r2: float
    mape: float
    directional_accuracy: float
    inference_time_ms: float


@dataclass
class ComparisonReport:
    """Comparison report for multiple models."""
    models: List[ModelMetrics]
    winner_index: int
    winner_model: ModelMetrics
    comparison_table: str
    timestamp: str

    def print_report(self):
        """Print formatted comparison report."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 MODEL COMPARISON REPORT")
        logger.info("=" * 80)
        print(self.comparison_table)
        logger.info("=" * 80)
        logger.success(f"🏆 Winner: {self.winner_model.model_uri}")
        logger.info("=" * 80)


class ModelComparator:
    """Compare multiple models on the same test dataset.
    
    Args:
        tracking_uri: MLflow tracking URI
    
    Example:
        >>> comparator = ModelComparator()
        >>> report = comparator.compare([
        ...     "models:/lstm-multi-ticker/5",
        ...     "models:/lstm-multi-ticker/4"
        ... ])
        >>> report.print_report()
    """

    def __init__(self, tracking_uri: str = None):
        if tracking_uri is None:
            tracking_uri = f"file:{Path.cwd()}/data/mlflow/tracking"

        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

    def _load_model_and_get_metrics(
        self,
        model_uri: str,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        device: torch.device
    ) -> ModelMetrics:
        """Load model and calculate metrics.
        
        Args:
            model_uri: MLflow model URI
            X_test: Test features
            y_test: Test targets
            device: Torch device
            
        Returns:
            ModelMetrics with all metrics calculated
        """
        try:
            # Load model
            model = mlflow.pytorch.load_model(model_uri)
            model = model.to(device)
            model.eval()

            # Extract version from URI
            version = None
            if "/models:/" in model_uri or model_uri.startswith("models:/"):
                parts = model_uri.split("/")
                if len(parts) >= 3:
                    try:
                        version = int(parts[-1])
                    except:
                        version = parts[-1]  # Might be "Production", "Staging"

            # Measure inference time
            start_time = time.time()

            predictions = []
            actuals = []

            # All models MUST use embedding architecture (no legacy support)
            if not hasattr(model, 'ticker_embedding'):
                raise ValueError(
                    f"Model {model_uri} is incompatible (no embedding layer). "
                    "Please retrain using current StockLSTM architecture."
                )

            num_tickers = model.num_tickers if hasattr(model, 'num_tickers') else 2
            logger.info(f"Model uses embedding, generating random ticker_ids (0-{num_tickers-1})")

            with torch.no_grad():
                # Run in batches for more realistic timing
                batch_size = 32
                for i in range(0, len(X_test), batch_size):
                    X_batch = X_test[i:i+batch_size].to(device)
                    y_batch = y_test[i:i+batch_size].to(device)

                    # ALL models require ticker_ids
                    ticker_ids = torch.randint(0, num_tickers, (X_batch.shape[0],)).to(device)
                    preds = model(X_batch, ticker_ids)

                    predictions.extend(preds.cpu().numpy().flatten())
                    actuals.extend(y_batch.cpu().numpy().flatten())

            inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            metrics = calculate_all_metrics(actuals, predictions)

            return ModelMetrics(
                model_uri=model_uri,
                version=version,
                mae=metrics["MAE"],
                rmse=metrics["RMSE"],
                r2=metrics["R2"],
                mape=metrics["MAPE"],
                directional_accuracy=metrics["Directional_Accuracy"],
                inference_time_ms=inference_time
            )

        except Exception as e:
            logger.error(f"Failed to evaluate {model_uri}: {str(e)}")
            raise

    def compare(
        self,
        model_uris: List[str],
        X_test: torch.Tensor = None,
        y_test: torch.Tensor = None,
        test_size: int = 1000
    ) -> ComparisonReport:
        """Compare multiple models.
        
        Args:
            model_uris: List of MLflow model URIs
            X_test: Test features (optional, will use random if None)
            y_test: Test targets (optional, will use random if None)
            test_size: Size of random test set if X_test/y_test not provided
            
        Returns:
            ComparisonReport with detailed comparison
        """
        logger.info("=" * 80)
        logger.info(f"📊 Comparing {len(model_uris)} models")
        logger.info("=" * 80)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate test data if not provided
        if X_test is None or y_test is None:
            logger.warning("No test data provided, using random data for comparison")
            # Detect input_size from first model to generate compatible test data
            try:
                first_model = mlflow.pytorch.load_model(model_uris[0], map_location=device)
                input_size = first_model.input_size if hasattr(first_model, 'input_size') else 11
                logger.info(f"Detected input_size={input_size} from model")

                # If model uses embedding, subtract embedding_dim to get original num_features
                if hasattr(first_model, 'ticker_embedding'):
                    embedding_dim = first_model.ticker_embedding.embedding_dim
                    num_features = input_size - embedding_dim
                    logger.info(f"Model uses embedding (dim={embedding_dim}), using {num_features} features for test data")
                    input_size = num_features

            except Exception as e:
                logger.warning(f"Failed to detect input_size: {e}. Using default 11")
                input_size = 11

            # Generate dummy data matching model's expected shape
            X_test = torch.randn(test_size, 60, input_size)  # (samples, lookback, features)
            y_test = torch.randn(test_size, 1)

        # Evaluate each model
        model_metrics = []
        for i, model_uri in enumerate(model_uris, 1):
            logger.info(f"\n[{i}/{len(model_uris)}] Evaluating: {model_uri}")
            metrics = self._load_model_and_get_metrics(model_uri, X_test, y_test, device)
            model_metrics.append(metrics)
            logger.success(f"✓ MAE: {metrics.mae:.4f}, R²: {metrics.r2:.4f}")

        # Determine winner (lowest MAE, highest R²)
        # Priority: R² > MAE > RMSE
        winner_idx = 0
        best_r2 = model_metrics[0].r2
        best_mae = model_metrics[0].mae

        for i, m in enumerate(model_metrics[1:], 1):
            # Better R² wins
            if m.r2 > best_r2 + 0.01 or abs(m.r2 - best_r2) <= 0.01 and m.mae < best_mae:  # 1% improvement threshold
                winner_idx = i
                best_r2 = m.r2
                best_mae = m.mae

        # Build comparison table
        table = self._build_comparison_table(model_metrics, winner_idx)

        from datetime import datetime

        return ComparisonReport(
            models=model_metrics,
            winner_index=winner_idx,
            winner_model=model_metrics[winner_idx],
            comparison_table=table,
            timestamp=datetime.now().isoformat()
        )

    def _build_comparison_table(
        self,
        models: List[ModelMetrics],
        winner_idx: int
    ) -> str:
        """Build formatted comparison table."""
        lines = []

        # Header
        lines.append("\n| Model | Version | MAE | RMSE | R² | MAPE | Dir.Acc | Inf.Time |")
        lines.append("|-------|---------|-----|------|----|----- |---------|----------|")

        # Rows
        for i, m in enumerate(models):
            is_winner = i == winner_idx
            prefix = "🏆 " if is_winner else "   "
            version = str(m.version) if m.version else "N/A"

            # Format model name
            model_name = m.model_uri.split("/")[-2] if "/" in m.model_uri else m.model_uri
            if len(model_name) > 20:
                model_name = model_name[:17] + "..."

            line = (
                f"| {prefix}{model_name:<18} | {version:<7} | "
                f"{m.mae:.4f} | {m.rmse:.4f} | {m.r2:.4f} | "
                f"{m.mape:.2f}% | {m.directional_accuracy:.2f}% | "
                f"{m.inference_time_ms:.1f}ms |"
            )
            lines.append(line)

        # Improvement row (compare to baseline - first model)
        if len(models) > 1:
            lines.append("|-------|---------|-----|------|----|----- |---------|----------|")
            baseline = models[0]
            winner = models[winner_idx]

            mae_imp = ((baseline.mae - winner.mae) / baseline.mae * 100)
            rmse_imp = ((baseline.rmse - winner.rmse) / baseline.rmse * 100)
            r2_imp = ((winner.r2 - baseline.r2) / abs(baseline.r2) * 100) if baseline.r2 != 0 else 0
            time_diff = ((winner.inference_time_ms - baseline.inference_time_ms) / baseline.inference_time_ms * 100)

            imp_line = (
                f"| **Improvement** | - | "
                f"{'✅' if mae_imp > 0 else '❌'}{mae_imp:+.1f}% | "
                f"{'✅' if rmse_imp > 0 else '❌'}{rmse_imp:+.1f}% | "
                f"{'✅' if r2_imp > 0 else '❌'}{r2_imp:+.1f}% | "
                f"- | - | "
                f"{'⚠️' if abs(time_diff) > 20 else '✅'}{time_diff:+.1f}% |"
            )
            lines.append(imp_line)

        return "\n".join(lines)


if __name__ == "__main__":
    # Test comparator
    comparator = ModelComparator()

    # Example: Compare two model versions
    # report = comparator.compare([
    #     "models:/lstm-multi-ticker/5",
    #     "models:/lstm-multi-ticker/4"
    # ])
    # report.print_report()

    logger.info("Model Comparator module loaded successfully")
