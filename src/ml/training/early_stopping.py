"""Early stopping callback for training.

This module implements early stopping to prevent overfitting during training.
Training stops when the monitored metric stops improving.
"""

from typing import Literal

import numpy as np
from loguru import logger


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving.

    This callback monitors a metric (typically validation loss) and stops
    training when the metric has stopped improving for a specified number
    of epochs (patience).

    Attributes:
        patience: Number of epochs with no improvement to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better).
        counter: Number of epochs since last improvement.
        best_score: Best score achieved so far.
        early_stop: Whether to stop training.

    Example:
        >>> early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_loss = train_one_epoch()
        ...     early_stopping(val_loss)
        ...     if early_stopping.early_stop:
        ...         print("Early stopping triggered!")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        verbose: bool = True,
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' to minimize metric, 'max' to maximize.
            verbose: If True, print messages when improvement occurs.

        Raises:
            ValueError: If patience <= 0 or mode not in ['min', 'max'].
        """
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        # For min mode, we want lower values
        self.mode_worse = np.inf if mode == "min" else -np.inf

        logger.info(
            f"Initialized EarlyStopping: patience={patience}, min_delta={min_delta}, mode={mode}"
        )

    def __call__(self, score: float, epoch: int = 0) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value to monitor.
            epoch: Current epoch number (for logging).

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                logger.info(f"Initial score: {score:.6f}")
            return False

        # Check if score improved
        if self._is_improvement(score):
            improvement = abs(score - self.best_score)
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"Epoch {epoch}: Score improved by {improvement:.6f} to {score:.6f}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                logger.debug(
                    f"Epoch {epoch}: No improvement. "
                    f"Counter: {self.counter}/{self.patience}. "
                    f"Best: {self.best_score:.6f} (epoch {self.best_epoch})"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning(
                    f"Early stopping triggered at epoch {epoch}! "
                    f"Best score: {self.best_score:.6f} (epoch {self.best_epoch})"
                )
                return True

            return False

    def _is_improvement(self, score: float) -> bool:
        """Check if current score is an improvement over best score.

        Args:
            score: Current score.

        Returns:
            True if score improved, False otherwise.
        """
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            return score > (self.best_score + self.min_delta)

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        logger.debug("EarlyStopping reset")

    def state_dict(self) -> dict:
        """Get state dictionary for checkpointing.

        Returns:
            Dictionary with early stopping state.
        """
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from dictionary.

        Args:
            state_dict: Dictionary with early stopping state.
        """
        self.counter = state_dict.get("counter", 0)
        self.best_score = state_dict.get("best_score")
        self.early_stop = state_dict.get("early_stop", False)
        self.best_epoch = state_dict.get("best_epoch", 0)
        logger.debug("EarlyStopping state loaded")
