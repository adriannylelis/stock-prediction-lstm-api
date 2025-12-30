"""Training pipeline orchestration.

Complete end-to-end training pipeline: ingestion → preprocessing → training → evaluation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from ..data.feature_engineering import TechnicalIndicators
from ..data.ingestion import StockDataIngestion
from ..data.preprocessing import StockPreprocessor
from ..models.lstm import create_model
from ..training.metrics import calculate_all_metrics
from ..training.trainer import Trainer
from ..utils.device import get_device
from ..utils.persistence import ArtifactManager, DataVersionManager
from ..utils.seed import set_seed


class TrainPipeline:
    """End-to-end training pipeline.

    Orchestrates the complete ML workflow:
    1. Data Ingestion (yfinance)
    2. Feature Engineering (technical indicators)
    3. Preprocessing (normalization, sequences)
    4. Model Training (with early stopping + MLflow)
    5. Evaluation (test metrics)

    Example:
        >>> pipeline = TrainPipeline(ticker="PETR4.SA", experiment_name="lstm-petr4")
        >>> results = pipeline.run()
        >>> print(results["test_metrics"])
    """

    def __init__(
        self,
        ticker: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        lookback: int = 60,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        # Model params
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        # Training params
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        # MLflow
        experiment_name: Optional[str] = None,
        # Paths
        model_save_path: str = "artifacts/models/best_model.pt",
        metrics_save_path: Optional[str] = None,
        # Other
        seed: int = 42,
        device: Optional[str] = None,
    ):
        """Initialize training pipeline.

        Args:
            ticker: Stock ticker symbol (e.g., PETR4.SA).
            start_date: Start date for data.
            end_date: End date for data (None = today).
            lookback: Lookback period for sequences.
            train_ratio: Training set ratio.
            val_ratio: Validation set ratio.
            test_ratio: Test set ratio.
            hidden_size: LSTM hidden size.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
            learning_rate: Learning rate.
            weight_decay: L2 regularization.
            batch_size: Batch size.
            epochs: Maximum training epochs.
            early_stopping_patience: Early stopping patience.
            experiment_name: MLflow experiment name.
            model_save_path: Path to save best model.
            metrics_save_path: Path to save metrics JSON.
            seed: Random seed.
            device: Device (cpu/cuda/auto).
        """
        # Data params
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Model params
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # MLflow
        self.experiment_name = experiment_name or f"lstm-{ticker.replace('.SA', '').lower()}"

        # Paths
        self.model_save_path = model_save_path
        self.metrics_save_path = (
            metrics_save_path
            or f"artifacts/metrics/{ticker.replace('.SA', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Setup
        self.seed = seed
        self.device = get_device() if device == "auto" or device is None else torch.device(device)

        # Storage for results
        self.data = None
        self.model = None
        self.trainer = None
        self.training_history = None
        self.preprocessor = None

        # Persistence managers
        self.data_version_manager = DataVersionManager()
        self.artifact_manager = ArtifactManager()
        self.raw_data_version = None
        self.processed_data_version = None

        logger.info(f"Initialized TrainPipeline for {ticker}")

    def run(self) -> Dict[str, Any]:
        """Run complete training pipeline.

        Returns:
            Dictionary with results:
                - model_path: Path to saved model
                - training_history: Training metrics per epoch
                - test_metrics: Final test metrics
                - metadata: Pipeline metadata
        """
        set_seed(self.seed)

        logger.info(f"{'=' * 60}")
        logger.info(f"🚀 Starting Training Pipeline: {self.ticker}")
        logger.info(f"{'=' * 60}")

        # Step 1: Data Ingestion
        logger.info("📥 Step 1/5: Data Ingestion")
        df = self._ingest_data()
        logger.success(f"✓ Downloaded {len(df)} records")

        # Step 2: Feature Engineering
        logger.info("🔧 Step 2/5: Feature Engineering")
        df = self._engineer_features(df)
        logger.success(f"✓ Generated {df.shape[1]} features")

        # Step 3: Preprocessing
        logger.info("⚙️ Step 3/5: Preprocessing")
        self.data = self._preprocess_data(df)
        logger.success(
            f"✓ Train: {len(self.data['X_train'])}, "
            f"Val: {len(self.data['X_val'])}, "
            f"Test: {len(self.data['X_test'])}"
        )

        # Step 4: Training
        logger.info("🏋️ Step 4/5: Training")
        self.model, self.training_history = self._train_model()
        best_val_loss = min(self.training_history["val_loss"])
        logger.success(f"✓ Training complete! Best val loss: {best_val_loss:.6f}")

        # Step 5: Evaluation
        logger.info("📊 Step 5/5: Evaluation")
        test_metrics = self._evaluate_model()
        logger.success(f"✅ Test MAE: {test_metrics['MAE']:.4f}, RMSE: {test_metrics['RMSE']:.4f}")

        # Save results
        results = self._save_results(test_metrics)

        logger.info(f"{'=' * 60}")
        logger.success("✅ Pipeline Complete!")
        logger.info(f"Model: {self.model_save_path}")
        logger.info(f"Metrics: {self.metrics_save_path}")
        logger.info(f"{'=' * 60}")

        return results

    def _ingest_data(self):
        """Ingest data from yfinance."""
        ingestion = StockDataIngestion(
            ticker=self.ticker, start_date=self.start_date, end_date=self.end_date
        )
        df = ingestion.download_and_validate()

        # Save raw data version
        self.raw_data_version = self.data_version_manager.save(
            df, ticker=self.ticker, metadata={"stage": "raw", "source": "yfinance"}
        )

        return df

    def _engineer_features(self, df):
        """Add technical indicators."""
        tech_ind = TechnicalIndicators(df)
        df = tech_ind.add_all_indicators()
        df = tech_ind.fill_missing_values()

        # Save processed data version
        self.processed_data_version = self.data_version_manager.save(
            df,
            ticker=self.ticker,
            metadata={
                "stage": "processed",
                "features_added": list(df.columns),
                "raw_version": self.raw_data_version,
            },
        )

        return df

    def _preprocess_data(self, df):
        """Preprocess data (normalize + sequences)."""
        self.preprocessor = StockPreprocessor(
            lookback_period=self.lookback,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
        )
        data = self.preprocessor.prepare_data(df)

        # Save scaler for later use
        self._save_scaler()

        return data

    def _train_model(self):
        """Train LSTM model."""
        # Create model
        input_size = self.data["X_train"].shape[2]
        model = create_model(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device,
        )

        # Create dataloaders
        train_dataset = TensorDataset(self.data["X_train"], self.data["y_train"])
        val_dataset = TensorDataset(self.data["X_val"], self.data["y_val"])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Create trainer
        trainer = Trainer(
            model=model,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            loss_function="MSE",
            early_stopping_patience=self.early_stopping_patience,
            experiment_name=self.experiment_name,
            checkpoint_dir=str(Path(self.model_save_path).parent),
        )

        # Train
        history = trainer.train(
            train_loader=train_loader, val_loader=val_loader, epochs=self.epochs
        )

        return model, history

    def _evaluate_model(self):
        """Evaluate model on test set."""
        self.model.eval()

        test_dataset = TensorDataset(self.data["X_test"], self.data["y_test"])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(X_batch)

                # Flatten predictions and actuals to 1D
                predictions.extend(preds.cpu().numpy().flatten())
                actuals.extend(y_batch.cpu().numpy().flatten())

        return calculate_all_metrics(np.array(actuals), np.array(predictions))

    def _save_results(self, test_metrics):
        """Save pipeline results."""
        # Get actual model path (Trainer saves as best_model.pt)
        actual_model_path = Path(self.model_save_path).parent / "best_model.pt"

        results = {
            "model_path": str(actual_model_path),
            "training_history": self.training_history,
            "test_metrics": test_metrics,
            "metadata": {
                "ticker": self.ticker,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "lookback": self.lookback,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs_trained": len(self.training_history["train_loss"]),
                "seed": self.seed,
                "timestamp": datetime.now().isoformat(),
                "data_versions": {
                    "raw": self.raw_data_version,
                    "processed": self.processed_data_version,
                },
            },
        }

        # Save metrics
        metrics_path = Path(self.metrics_save_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _save_scaler(self):
        """Save fitted scaler for predictions."""
        from datetime import datetime

        import joblib

        scaler_name = f"{self.ticker.replace('.SA', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.artifact_manager.save_scaler(
            self.preprocessor.scaler,
            scaler_name,
            metadata={
                "ticker": self.ticker,
                "lookback": self.lookback,
                "data_version": getattr(self, "processed_data_version", None),
            },
        )

        # Also save to model directory for convenience
        scaler_path = Path(self.model_save_path).parent / "scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor.scaler, scaler_path)
        logger.info(f"💾 Saved scaler to: {scaler_path}")
