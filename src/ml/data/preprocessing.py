"""Data preprocessing for LSTM training.

This module handles data normalization, sequence creation (sliding window),
and train/val/test splitting for time series data.
"""

from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import MinMaxScaler


class StockPreprocessor:
    """Preprocessor for stock data to prepare for LSTM training.

    This class handles:
    1. Feature selection
    2. Normalization (MinMaxScaler)
    3. Sequence creation (sliding window for LSTM input)
    4. Train/validation/test splitting
    5. Conversion to PyTorch tensors

    Attributes:
        scaler: MinMaxScaler instance for normalization.
        lookback_period: Number of time steps to look back for LSTM input.
        feature_cols: Columns to use as features.

    Example:
        >>> preprocessor = StockPreprocessor(lookback_period=60)
        >>> X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.prepare_data(df)
    """

    def __init__(
        self,
        lookback_period: int = 60,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "Close",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            lookback_period: Number of past time steps to use as input.
            feature_cols: List of columns to use as features. If None, uses only 'Close'.
            target_col: Column to predict (typically 'Close').
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.

        Raises:
            ValueError: If ratios don't sum to 1.0.
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        self.lookback_period = lookback_period
        self.feature_cols = feature_cols or ["Close"]
        self.target_col = target_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

        logger.info(
            f"Initialized preprocessor: lookback={lookback_period}, features={self.feature_cols}"
        )

    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize data using MinMaxScaler.

        Args:
            data: Array to normalize (shape: [samples, features]).
            fit: If True, fit the scaler. If False, use existing scaler.

        Returns:
            Normalized array in range [0, 1].

        Raises:
            ValueError: If trying to transform without fitting first.
        """
        if fit:
            normalized = self.scaler.fit_transform(data)
            self.is_fitted = True
            logger.debug(f"Fitted scaler on data with shape {data.shape}")
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call normalize(fit=True) first.")
            normalized = self.scaler.transform(data)
            logger.debug(f"Transformed data with shape {data.shape}")

        return normalized

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization to get original scale.

        Args:
            data: Normalized array to denormalize.

        Returns:
            Array in original scale.

        Raises:
            ValueError: If scaler not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Cannot denormalize.")

        return self.scaler.inverse_transform(data)

    def create_sequences(
        self, data: np.ndarray, lookback: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences for LSTM input using sliding window (vectorized with PyTorch).

        Args:
            data: Normalized data array (shape: [samples, features]).
            lookback: Lookback period. If None, uses self.lookback_period.

        Returns:
            Tuple of (X, y) as PyTorch tensors:
                - X: shape [num_sequences, lookback, features]
                - y: shape [num_sequences] (next value to predict)

        Example:
            >>> data = np.array([[1], [2], [3], [4], [5]])
            >>> X, y = preprocessor.create_sequences(data, lookback=3)
            >>> # X = tensor([[[1], [2], [3]]]), y = tensor([4])
        """
        lookback = lookback or self.lookback_period

        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data)
        else:
            data_tensor = data

        # Vectorized sequence creation using unfold
        # unfold(dimension, size, step) creates sliding windows

        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(1)  # [T] -> [T, 1]

        # Create sequences: [samples - lookback, lookback, features]
        # unfold gera [num_windows, features, lookback], então precisamos ajustar
        X = data_tensor.unfold(0, lookback, 1).permute(
            0, 2, 1
        )  # -> [num_windows, lookback, features]

        # Remover a última sequência porque não tem target (y)
        X = X[:-1]  # Agora temos lookback -> lookback+1, ..., n-2 -> n-1 (falta o n)

        # Target é o próximo valor após cada sequência
        y = data_tensor[lookback:, 0]  # Predict first feature (Close price)

        logger.debug(f"Created sequences with PyTorch: X shape={X.shape}, y shape={y.shape}")

        return X, y

    def split_data(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split data into train/validation/test sets using PyTorch.

        Uses chronological split (no shuffling) to maintain time series order.

        Args:
            X: Feature tensor (sequences).
            y: Target tensor.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) as PyTorch tensors.
        """
        n = len(X)

        train_size = int(n * self.train_ratio)
        val_size = int(n * self.val_ratio)
        # test_size = n - train_size - val_size (resto)

        # PyTorch split: divide tensor into chunks
        X_splits = torch.split(X, [train_size, val_size, n - train_size - val_size])
        y_splits = torch.split(y, [train_size, val_size, n - train_size - val_size])

        X_train, X_val, X_test = X_splits
        y_train, y_val, y_test = y_splits

        logger.info(
            f"Split data with PyTorch: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def to_device(
        self, *tensors: torch.Tensor, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Move tensors to specified device (CPU/GPU).

        Args:
            *tensors: Variable number of tensors to move.
            device: Target device. If None, uses CPU.

        Returns:
            Tuple of tensors on target device.

        Example:
            >>> X_train, y_train = preprocessor.to_device(X_train, y_train, device="cuda")
        """
        device = device or torch.device("cpu")
        moved_tensors = tuple(t.to(device) for t in tensors)

        logger.debug(f"Moved {len(tensors)} tensors to {device}")

        return moved_tensors if len(moved_tensors) > 1 else moved_tensors[0]

    def prepare_data(self, df: pd.DataFrame) -> dict:
        """Complete preprocessing pipeline.

        Performs all preprocessing steps:
        1. Feature selection
        2. Normalization
        3. Sequence creation
        4. Train/val/test split
        5. Tensor conversion

        Args:
            df: DataFrame with stock data and features.

        Returns:
            Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test
            (all as PyTorch tensors).

        Example:
            >>> data = preprocessor.prepare_data(df)
            >>> X_train = data["X_train"]
        """
        logger.info("Starting data preprocessing pipeline...")

        # 1. Select features
        if not all(col in df.columns for col in self.feature_cols):
            missing = [col for col in self.feature_cols if col not in df.columns]
            raise ValueError(f"Missing feature columns: {missing}")

        data = df[self.feature_cols].values
        logger.info(f"Selected features: {self.feature_cols}, shape={data.shape}")

        # 2. Normalize
        data_normalized = self.normalize(data, fit=True)

        # 3. Create sequences (já retorna tensors PyTorch)
        X, y = self.create_sequences(data_normalized)

        # 4. Split (opera diretamente em tensors)
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(X, y)

        logger.info("✓ Preprocessing complete! All data as PyTorch tensors.")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "scaler": self.scaler,
        }

    def save_scaler(self, filepath: str) -> None:
        """Save scaler to file for later use in production.

        Args:
            filepath: Path to save scaler (e.g., 'artifacts/scaler.pkl').
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Cannot save.")

        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath: str) -> None:
        """Load scaler from file.

        Args:
            filepath: Path to scaler file.
        """
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {filepath}")
