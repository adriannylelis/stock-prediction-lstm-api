"""Hyperparameter tuning with Optuna.

This module implements Bayesian optimization for LSTM hyperparameters using Optuna.
"""

from typing import Dict, Optional

import optuna
import torch
from loguru import logger
from optuna.trial import Trial
from torch.utils.data import DataLoader, TensorDataset

from ..models.lstm import create_model
from .trainer import Trainer


class HyperparameterTuner:
    """Optuna-based hyperparameter tuner for LSTM models.

    Performs Bayesian optimization to find best hyperparameters:
    - Learning rate
    - Hidden size
    - Number of layers
    - Dropout rate
    - Batch size

    Example:
        >>> tuner = HyperparameterTuner(X_train, y_train, X_val, y_val, n_trials=50, device="cuda")
        >>> best_params = tuner.optimize()
        >>> print(best_params)
    """

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        device: str = "cpu",
        study_name: str = "lstm_hyperparameter_tuning",
        storage: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Initialize hyperparameter tuner.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            n_trials: Number of optimization trials.
            timeout: Timeout in seconds (None = no limit).
            device: Device to run training on.
            study_name: Optuna study name.
            storage: Database URL for study persistence (None = in-memory).
            experiment_name: MLflow experiment name for tracking trials.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.n_trials = n_trials
        self.timeout = timeout
        self.device = torch.device(device)
        self.experiment_name = experiment_name

        # Create Optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            direction="minimize",  # Minimize validation loss
            storage=storage,
            load_if_exists=True,
        )

        logger.info(f"Initialized HyperparameterTuner: {n_trials} trials, device={device}")

    def _create_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders.

        Args:
            batch_size: Batch size for dataloaders.

        Returns:
            Tuple of (train_loader, val_loader).
        """
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def objective(self, trial: Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial object.

        Returns:
            Validation loss (metric to minimize).
        """
        # Suggest hyperparameters
        hidden_size = trial.suggest_int("hidden_size", 32, 256, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        logger.info(
            f"Trial {trial.number}: "
            f"hidden={hidden_size}, layers={num_layers}, "
            f"dropout={dropout:.3f}, lr={learning_rate:.6f}, "
            f"batch={batch_size}, wd={weight_decay:.6f}"
        )

        # Create model
        input_size = self.X_train.shape[2]  # Number of features
        model = create_model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=self.device,
        )

        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders(batch_size)

        # Create trainer
        trainer = Trainer(
            model=model,
            device=self.device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_function="MSE",
            early_stopping_patience=10,
            early_stopping_min_delta=0.0001,
            experiment_name=self.experiment_name if self.experiment_name else None,
            tracking_uri="file:./mlruns",
        )

        # Train with reduced epochs for faster trials
        max_epochs = 30  # Reduced for tuning speed

        try:
            history = trainer.train(
                train_loader=train_loader, val_loader=val_loader, epochs=max_epochs
            )

            # Get best validation loss
            best_val_loss = min(history["val_loss"])

            logger.info(f"Trial {trial.number}: Best Val Loss = {best_val_loss:.6f}")

            return best_val_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float("inf")

    def optimize(self, show_progress: bool = True, callbacks: Optional[list] = None) -> Dict:
        """Run hyperparameter optimization.

        Args:
            show_progress: Show progress bar.
            callbacks: List of Optuna callbacks.

        Returns:
            Dictionary with best hyperparameters.
        """
        logger.info(f"Starting hyperparameter optimization: {self.n_trials} trials")

        # Add progress bar callback if requested
        if show_progress and callbacks is None:
            callbacks = [optuna.study.MaxTrialsCallback(self.n_trials, states=None)]

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=callbacks,
            show_progress_bar=show_progress,
        )

        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value

        logger.success(
            f"Optimization complete! Best Val Loss: {best_value:.6f}\nBest params: {best_params}"
        )

        return best_params

    def get_optimization_history(self) -> Dict:
        """Get optimization history.

        Returns:
            Dictionary with trials history.
        """
        trials_df = self.study.trials_dataframe()

        return {
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial": self.study.best_trial.number,
            "n_trials": len(self.study.trials),
            "trials_df": trials_df,
        }

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history.

        Args:
            save_path: Path to save plot (None = show only).
        """
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot optimization history
            plot_optimization_history(self.study, ax=ax1)
            ax1.set_title("Optimization History")

            # Plot parameter importances
            plot_param_importances(self.study, ax=ax2)
            ax2.set_title("Hyperparameter Importances")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Saved optimization plots to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")

    def get_best_model_config(self) -> Dict:
        """Get configuration for best model.

        Returns:
            Dictionary with model configuration.
        """
        best_params = self.study.best_params

        return {
            "input_size": self.X_train.shape[2],
            "hidden_size": best_params["hidden_size"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"],
            "learning_rate": best_params["learning_rate"],
            "batch_size": best_params["batch_size"],
            "weight_decay": best_params["weight_decay"],
        }
