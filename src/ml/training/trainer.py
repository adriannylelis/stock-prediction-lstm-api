"""Trainer class for LSTM model training.

This module implements the training loop for LSTM models with support for
early stopping, checkpointing, and MLflow tracking.
"""

from pathlib import Path
from typing import Dict, Optional

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
        tracking_uri: str = "file:./mlruns",
        checkpoint_dir: str = "artifacts/models",
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
        if experiment_name:
            self.tracker = ExperimentTracker(
                experiment_name=experiment_name, tracking_uri=tracking_uri
            )
        else:
            self.tracker = None

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            outputs = self.model(X_batch)
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
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
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
            self.tracker.log_params(
                {
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epochs": epochs,
                    "batch_size": train_loader.batch_size,
                    "model_class": self.model.__class__.__name__,
                    "loss_function": self.criterion.__class__.__name__,
                    "device": str(self.device),
                }
            )

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
                        f"Best: {self.best_val_loss:.6f} (epoch {self.best_epoch})"
                    )

                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_loss, epoch + 1):
                        logger.warning(f"Early stopping at epoch {epoch + 1}")
                        break

            logger.info(
                f"✓ Training complete! Best val loss: {self.best_val_loss:.6f} (epoch {self.best_epoch})"
            )

            # Log best metrics
            if self.tracker:
                self.tracker.log_metrics(
                    {"best_val_loss": self.best_val_loss, "best_epoch": self.best_epoch}
                )

                # Log model
                model_path = self.checkpoint_dir / "best_model.pt"
                if model_path.exists():
                    self.tracker.log_artifact(str(model_path))

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
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: If True, save as best model.
            filename: Custom filename. If None, uses default naming.
        """
        if filename is None:
            if is_best:
                filename = "best_model.pt"
            else:
                filename = f"checkpoint_epoch_{epoch + 1}.pt"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            # Save model architecture for inference
            "input_size": self.model.input_size,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
            "dropout": self.model.dropout_prob,  # Save dropout probability, not layer
        }

        torch.save(checkpoint, filepath)
        logger.debug(f"Saved checkpoint: {filepath}")

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
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)

                all_predictions.extend(outputs.squeeze().cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        predictions = torch.tensor(all_predictions).numpy()
        targets = torch.tensor(all_targets).numpy()

        # Denormalize if scaler provided
        if scaler:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

        # Calculate metrics
        metrics = calculate_all_metrics(targets, predictions)

        # Log to MLflow
        if self.tracker and self.tracker.run:
            test_metrics = {f"test_{k}": v for k, v in metrics.items()}
            self.tracker.log_metrics(test_metrics)

        logger.info("Test metrics calculated")

        return metrics
