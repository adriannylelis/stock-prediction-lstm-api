"""Trainer class for LSTM model training.

This module implements the training loop for LSTM models with support for
early stopping, checkpointing, and MLflow tracking.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from .early_stopping import EarlyStopping
from .experiment_tracker import ExperimentTracker
from .metrics import calculate_all_metrics


class Trainer:
    """Trainer for LSTM stock prediction models.

    This class handles the complete training loop including:
    - Training and validation
    - Early stopping
    - Model checkpointing
    - MLflow experiment tracking
    - Metrics calculation and logging

    Attributes:
        model: PyTorch LSTM model.
        device: Device to train on (CPU or CUDA).
        criterion: Loss function.
        optimizer: Optimizer.
        early_stopping: Early stopping callback (optional).
        tracker: MLflow experiment tracker (optional).

    Example:
        >>> trainer = Trainer(
        ...     model=model, device=device, learning_rate=0.001, experiment_name="stock-lstm"
        ... )
        >>> history = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        loss_function: str = "MSE",
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0001,
        experiment_name: Optional[str] = None,
        tracking_uri: str = "file:data/mlflow/tracking",
        checkpoint_dir: str = "artifacts/models",
        extra_params: Optional[Dict] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PyTorch model to train.
            device: Device (CPU or CUDA).
            learning_rate: Learning rate for optimizer.
            weight_decay: L2 regularization weight.
            loss_function: Loss function name ('MSE', 'MAE', 'Huber').
            early_stopping_patience: Patience for early stopping. None to disable.
            early_stopping_min_delta: Min improvement for early stopping.
            experiment_name: MLflow experiment name. None to disable tracking.
            tracking_uri: MLflow tracking URI.
            checkpoint_dir: Directory to save model checkpoints.
            extra_params: Additional parameters to log in MLflow (ticker, lookback, hidden_layer, etc).
        """
        self.model = model.to(device)
        self.device = device

        # Loss function
        self.criterion = self._get_loss_function(loss_function)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Early stopping
        if early_stopping_patience:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience, min_delta=early_stopping_min_delta, mode="min"
            )
        else:
            self.early_stopping = None

        # MLflow tracking
        self.experiment_name = experiment_name  # Store for logging
        if experiment_name:
            self.tracker = ExperimentTracker(
                experiment_name=experiment_name, tracking_uri=tracking_uri
            )
        else:
            self.tracker = None

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Extra parameters for MLflow
        self.extra_params = extra_params or {}

        # Training history
        self.history = {"train_loss": [], "val_loss": [], "epoch": []}

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        logger.info(f"Initialized Trainer on device: {device}")

    @staticmethod
    def _get_loss_function(name: str) -> nn.Module:
        """Get loss function by name.

        Args:
            name: Loss function name.

        Returns:
            PyTorch loss function.

        Raises:
            ValueError: If loss function name is invalid.
        """
        loss_functions = {"MSE": nn.MSELoss(), "MAE": nn.L1Loss(), "Huber": nn.HuberLoss()}

        if name not in loss_functions:
            raise ValueError(
                f"Invalid loss function: {name}. Choose from {list(loss_functions.keys())}"
            )

        return loss_functions[name]

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Unpack batch - pode ter 2 ou 3 elementos (X, y) ou (X, y, ticker_ids)
            # Batch order: (X, y, ticker_ids)
            if len(batch) == 3:
                X_batch, y_batch, ticker_ids_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)  # Already 1D from pipeline
                ticker_ids_batch = ticker_ids_batch.long().to(self.device)  # Force long dtype

                # Forward pass com ticker_ids (embedding model)
                outputs, _ = self.model(X_batch, ticker_ids_batch)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.squeeze().to(self.device)  # Ensure 1D: (batch,)

                # Forward pass sem ticker_ids (backward compatibility)
                outputs, _ = self.model(X_batch)

            loss = self.criterion(outputs.squeeze(), y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Average validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch - pode ter 2 ou 3 elementos (X, y) ou (X, y, ticker_ids)
                if len(batch) == 3:
                    X_batch, y_batch, ticker_ids_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)  # Already 1D from pipeline
                    ticker_ids_batch = ticker_ids_batch.long().to(self.device)  # Force long dtype

                    # Forward pass com ticker_ids (embedding model)
                    outputs, _ = self.model(X_batch, ticker_ids_batch)
                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.squeeze().to(self.device)  # Ensure 1D: (batch,)

                    # Forward pass sem ticker_ids (backward compatibility)
                    outputs, _ = self.model(X_batch)

                loss = self.criterion(outputs.squeeze(), y_batch)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        run_name: Optional[str] = None,
        save_best_only: bool = True,
        log_every_n_epochs: int = 10,
    ) -> Dict:
        """Complete training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train.
            run_name: MLflow run name.
            save_best_only: If True, save only the best model.
            log_every_n_epochs: Print progress every N epochs.

        Returns:
            Dictionary with training history.
        """
        # Start MLflow run
        if self.tracker:
            self.tracker.start_run(run_name=run_name)

            # Log hyperparameters
            params = {
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "model_class": self.model.__class__.__name__,
                "loss_function": self.criterion.__class__.__name__,
                "device": str(self.device),
            }

            # Add extra parameters if provided
            if self.extra_params:
                params.update(self.extra_params)

            self.tracker.log_params(params)

        # ✅ Store sample data for model signature
        try:
            sample_batch = next(iter(train_loader))
            self.X_train_sample = sample_batch[0][:5]  # Store 5 samples (X)
            # Store ticker_ids if available (for embedding models)
            # Batch order is: (X, y, ticker_ids), so ticker_ids is at index 2
            if len(sample_batch) >= 3:
                self.ticker_ids_sample = sample_batch[2][:5]  # ticker_ids at index 2 (not 1!)
                logger.debug(f"Captured ticker_ids samples: min={self.ticker_ids_sample.min().item()}, max={self.ticker_ids_sample.max().item()}")
            else:
                self.ticker_ids_sample = None
        except Exception:
            self.X_train_sample = None
            self.ticker_ids_sample = None

        logger.info(f"Starting training for {epochs} epochs...")

        try:
            for epoch in range(epochs):
                # Train
                train_loss = self.train_epoch(train_loader)

                # Validate
                val_loss = self.validate_epoch(val_loader)

                # Update history
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["epoch"].append(epoch + 1)

                # Log to MLflow
                if self.tracker:
                    self.tracker.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                    )

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    if save_best_only:
                        self.save_checkpoint(epoch, is_best=True)

                # Print progress
                if (epoch + 1) % log_every_n_epochs == 0:
                    logger.info(
                        f"Epoch [{epoch + 1:3d}/{epochs}] | "
                        f"Train Loss: {train_loss:.6f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"Best: {self.best_val_loss:.6f} (at epoch {self.best_epoch})"
                    )

                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_loss, epoch + 1):
                        logger.warning(f"Early stopping at epoch {epoch + 1}")
                        break

            logger.info(
                f"✓ Training complete! Best val loss: {self.best_val_loss:.6f} (achieved at epoch {self.best_epoch})"
            )

            # Log best metrics
            if self.tracker:
                self.tracker.log_metrics(
                    {"best_val_loss": self.best_val_loss, "best_epoch": self.best_epoch}
                )

                # Log model artifact and register in MLflow Model Registry
                model_path = self.checkpoint_dir / "best_model.pt"
                if model_path.exists():
                    self.tracker.log_artifact(str(model_path))

                    # ✅ Register model with complete metadata
                    from datetime import datetime

                    import mlflow.pytorch
                    from mlflow.tracking import MlflowClient

                    # Handle both single ticker and multi-ticker models
                    ticker = self.extra_params.get('ticker')
                    tickers_str = self.extra_params.get('tickers')
                    model_type = self.extra_params.get('model_type', 'single')

                    # Always use standard model name for consistency
                    model_name = "stock-lstm-model"

                    # Prepare sample data for model signature
                    sample_input = None
                    sample_output = None
                    input_example = None
                    signature = None

                    if hasattr(self, 'X_train_sample') and self.X_train_sample is not None:
                        with torch.no_grad():
                            # Check if model needs ticker_ids (embedding model)
                            if hasattr(self, 'ticker_ids_sample') and self.ticker_ids_sample is not None:
                                # Embedding model: create input with both features and ticker_ids
                                features_sample = self.X_train_sample.cpu().numpy()
                                ticker_ids_sample = self.ticker_ids_sample.cpu().numpy()

                                logger.debug(f"Creating MLflow signature with ticker_ids: min={ticker_ids_sample.min()}, max={ticker_ids_sample.max()}, shape={ticker_ids_sample.shape}")
                                logger.debug(f"Model expects num_tickers={self.model.num_tickers}")

                                # Create structured input as dictionary
                                sample_input = {
                                    "features": features_sample,
                                    "ticker_ids": ticker_ids_sample
                                }

                                # Get model output (model returns tuple: (outputs, hidden))
                                sample_output, _ = self.model(
                                    torch.tensor(features_sample, dtype=torch.float32).to(self.device),
                                    torch.tensor(ticker_ids_sample, dtype=torch.long).to(self.device)
                                )
                                sample_output = sample_output.cpu().numpy()

                                # Create input example with first sample
                                input_example = {
                                    "features": features_sample[:1],
                                    "ticker_ids": ticker_ids_sample[:1]
                                }

                                # Infer signature from structured input
                                signature = mlflow.models.infer_signature(sample_input, sample_output)
                            else:
                                # Backward compatibility (old one-hot models)
                                sample_input = self.X_train_sample.cpu().numpy()
                                sample_output, _ = self.model(
                                    torch.tensor(sample_input).to(self.device)
                                )
                                sample_output = sample_output.cpu().numpy()
                                input_example = sample_input[:1]
                                signature = mlflow.models.infer_signature(sample_input, sample_output)

                    # Log model with signature and metadata
                    try:
                        # ✅ Log scaler and preprocessing config as artifacts FIRST
                        scaler_path = self.extra_params.get('scaler_path')
                        if scaler_path and Path(scaler_path).exists():
                            self.tracker.log_artifact(scaler_path)
                            logger.info(f"📦 Logged X scaler artifact: {scaler_path}")

                        # Log y_scaler (for multi-ticker models)
                        y_scaler_path = self.extra_params.get('y_scaler_path')
                        if y_scaler_path and Path(y_scaler_path).exists():
                            self.tracker.log_artifact(y_scaler_path)
                            logger.info(f"📦 Logged y scaler artifact: {y_scaler_path}")

                        preprocessing_config_path = self.extra_params.get('preprocessing_config_path')
                        if preprocessing_config_path and Path(preprocessing_config_path).exists():
                            self.tracker.log_artifact(preprocessing_config_path)
                            logger.info(f"📦 Logged preprocessing config: {preprocessing_config_path}")

                        model_info = mlflow.pytorch.log_model(
                            self.model,
                            "model",  # artifact name (not artifact_path)
                            registered_model_name=model_name,
                            signature=signature,
                            input_example=input_example,
                            pip_requirements=[
                                f"torch=={torch.__version__}",
                                "numpy>=1.24.0",
                                "scikit-learn>=1.3.0",
                            ]
                        )

                        # Get latest version
                        client = MlflowClient()
                        run_id = self.tracker.get_run_id()
                        model_versions = client.search_model_versions(f"name='{model_name}'")

                        if model_versions:
                            latest_version = max([int(v.version) for v in model_versions])

                            # ✅ Add descriptive tags
                            client.set_model_version_tag(
                                model_name,
                                str(latest_version),
                                "validation_loss",
                                f"{self.best_val_loss:.6f}"
                            )

                            client.set_model_version_tag(
                                model_name,
                                str(latest_version),
                                "training_date",
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )

                            client.set_model_version_tag(
                                model_name,
                                str(latest_version),
                                "ticker",
                                ticker
                            )

                            client.set_model_version_tag(
                                model_name,
                                str(latest_version),
                                "best_epoch",
                                str(self.best_epoch)
                            )

                            # ✅ Add detailed description
                            description = (
                                f"LSTM Stock Prediction Model\\n"
                                f"- Ticker: {ticker}\\n"
                                f"- Validation Loss: {self.best_val_loss:.6f}\\n"
                                f"- Best Epoch: {self.best_epoch}\\n"
                                f"- Hidden Size: {self.extra_params.get('hidden_size', 'N/A')}\\n"
                                f"- Num Layers: {self.extra_params.get('num_layers', 'N/A')}\\n"
                                f"- Dropout: {self.extra_params.get('dropout', 'N/A')}\\n"
                                f"- Lookback: {self.extra_params.get('lookback', 'N/A')}"
                            )

                            client.update_model_version(
                                model_name,
                                str(latest_version),
                                description=description
                            )

                            # ✅ Auto-transition to Staging if performance is good
                            staging_threshold = self.extra_params.get('staging_threshold', 0.01)
                            if self.best_val_loss < staging_threshold:
                                client.transition_model_version_stage(
                                    name=model_name,
                                    version=str(latest_version),
                                    stage="Staging",
                                    archive_existing_versions=False
                                )
                                logger.success(
                                    f"✅ Model v{latest_version} → Staging "
                                    f"(val_loss={self.best_val_loss:.6f} < {staging_threshold})"
                                )
                            else:
                                logger.info(
                                    f"Model v{latest_version} remains in None stage "
                                    f"(val_loss={self.best_val_loss:.6f} >= {staging_threshold})"
                                )

                            logger.success(
                                f"📝 Model registered: {model_name} v{latest_version} "
                                f"(tracked in experiment: {self.experiment_name})"
                            )
                        else:
                            logger.info(
                                f"Model registered in MLflow: {model_name} "
                                f"(tracked in experiment: {self.experiment_name})"
                            )

                    except Exception as e:
                        logger.warning(f"Model registration failed: {e}")
                        logger.info("Model saved locally but not registered in MLflow")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.tracker:
                self.tracker.end_run(status="FAILED")
            raise

        finally:
            if self.tracker:
                self.tracker.end_run()

        return self.history

    def save_checkpoint(
        self, epoch: int, is_best: bool = False, filename: Optional[str] = None
    ) -> None:
        """Save model checkpoint with automatic versioning.

        Args:
            epoch: Current epoch number.
            is_best: If True, save as best model (creates timestamped version + best_model.pt).
            filename: Custom filename. If None, uses default naming.
        """
        from datetime import datetime

        if filename is None:
            if is_best:
                # Save versioned copy with timestamp for history
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ticker = self.extra_params.get('ticker', 'model')
                versioned_filename = f"best_model_{ticker.replace('.SA', '')}_{timestamp}.pt"

                # Save versioned copy first
                versioned_path = self.checkpoint_dir / versioned_filename
                self._save_checkpoint_file(versioned_path, epoch)
                logger.info(f"💾 Versioned: {versioned_filename}")

                # Then save as best_model.pt for backward compatibility
                filename = "best_model.pt"
            else:
                filename = f"checkpoint_epoch_{epoch + 1}.pt"

        filepath = self.checkpoint_dir / filename
        self._save_checkpoint_file(filepath, epoch)
        logger.debug(f"Saved checkpoint: {filepath}")

    def _save_checkpoint_file(self, filepath: Path, epoch: int) -> None:
        """Save checkpoint to file.
        
        Args:
            filepath: Path to save checkpoint.
            epoch: Current epoch number.
        """
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            # Save complete model architecture for inference (ticker embedding support)
            "num_tickers": self.model.num_tickers,
            "num_features": self.model.num_features,
            "embedding_dim": self.model.embedding_dim,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
            "dropout": self.model.dropout_rate,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", {})

        logger.info(f"Loaded checkpoint from {filepath}")

    def evaluate(self, test_loader: DataLoader, scaler=None) -> Dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_loader: Test data loader.
            scaler: Scaler to denormalize predictions (optional).

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch - pode ter 2 ou 3 elementos (X, y) ou (X, y, ticker_ids)
                if len(batch) == 3:
                    X_batch, y_batch, ticker_ids_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    ticker_ids_batch = ticker_ids_batch.long().to(self.device)  # Force long dtype

                    # Forward pass com ticker_ids (embedding model)
                    outputs, _ = self.model(X_batch, ticker_ids_batch)
                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass sem ticker_ids (backward compatibility)
                    outputs, _ = self.model(X_batch)

                all_predictions.extend(outputs.squeeze().cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        predictions = torch.tensor(all_predictions).numpy()
        targets = torch.tensor(all_targets).numpy()

        # Denormalize if scaler provided
        if scaler:
            # Ensure 2D for scaler (handles both 1D and 2D arrays)
            pred_2d = predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions
            targ_2d = targets.reshape(-1, 1) if targets.ndim == 1 else targets
            
            # Check if scaler has multiple features (need to extract Close column)
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ > 1:
                # Scaler is for all features - need to create dummy array with only Close column
                # Close is at index 0 in the feature array
                logger.debug(f"Scaler has {scaler.n_features_in_} features, using only Close column (index 0) for denormalization")
                
                # Create dummy arrays with zeros for other features, predictions/targets at index 0
                dummy_pred = np.zeros((len(pred_2d), scaler.n_features_in_))
                dummy_pred[:, 0] = pred_2d.flatten()
                dummy_targ = np.zeros((len(targ_2d), scaler.n_features_in_))
                dummy_targ[:, 0] = targ_2d.flatten()
                
                # Denormalize and extract Close column
                predictions = scaler.inverse_transform(dummy_pred)[:, 0]
                targets = scaler.inverse_transform(dummy_targ)[:, 0]
            else:
                # Scaler is for single feature (y_scaler) - direct denormalization
                predictions = scaler.inverse_transform(pred_2d).flatten()
                targets = scaler.inverse_transform(targ_2d).flatten()

        # Calculate metrics
        metrics = calculate_all_metrics(targets, predictions)

        # Log to MLflow
        if self.tracker and self.tracker.run:
            test_metrics = {f"test_{k}": v for k, v in metrics.items()}
            self.tracker.log_metrics(test_metrics)

        logger.info("Test metrics calculated")

        return metrics
