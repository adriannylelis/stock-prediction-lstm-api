"""Training pipeline with MLflow-first architecture and multi-ticker support.

This module replaces the old train_pipeline.py and integrates logic from scripts/train_unified_model.py.
Supports both single-ticker and multi-ticker training with proper MLflow persistence.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from ..data.feature_engineering import TechnicalIndicators
from ..data.ingestion import StockDataIngestion
from ..data.preprocessing import StockPreprocessor
from ..models.lstm import create_model
from ..training.trainer import Trainer
from ..utils.device import get_device
from ..utils.seed import set_seed


class TrainPipeline:
    """MLflow-first training pipeline with multi-ticker support.
    
    Supports two modes:
    1. Single-ticker: Traditional training for one stock
    2. Multi-ticker: Unified model for multiple stocks using ticker embeddings
    
    Architecture:
    - MLflow = Source of truth (models, scalers, configs, metrics)
    - Local artifacts = Minimal fallback only (best_model.pt)
    - Automatic cleanup of old checkpoints
    
    Example (single-ticker):
        >>> pipeline = TrainPipeline(ticker="PETR4.SA")
        >>> results = pipeline.run()
    
    Example (multi-ticker):
        >>> pipeline = TrainPipeline(tickers=["PETR4.SA", "VALE3.SA"])
        >>> results = pipeline.run()
    """

    def __init__(
        self,
        ticker: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        lookback: int = 60,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        # Model params
        hidden_size: int = 100,
        num_layers: int = 3,
        dropout: float = 0.3,
        embedding_dim: int = 8,  # For multi-ticker
        # Training params
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        # MLflow
        experiment_name: Optional[str] = None,
        tracking_uri: str = "file:data/mlflow/tracking",
        # Paths
        model_save_path: str = "artifacts/models/best_model.pt",
        # Other
        seed: int = 42,
        device: Optional[str] = None,
    ):
        """Initialize training pipeline.
        
        Args:
            ticker: Single ticker (for single-ticker mode)
            tickers: List of tickers (for multi-ticker mode)
            start_date: Start date for data
            lookback: Lookback period for sequences
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            embedding_dim: Ticker embedding dimension (multi-ticker only)
            learning_rate: Learning rate
            weight_decay: L2 regularization
            batch_size: Batch size
            epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
            model_save_path: Local fallback path
            seed: Random seed
            device: Device (cpu/cuda/mps/auto)
        """
        set_seed(seed)

        # Validate ticker configuration
        if ticker and tickers:
            raise ValueError("Provide either 'ticker' OR 'tickers', not both")
        if not ticker and not tickers:
            raise ValueError("Must provide either 'ticker' or 'tickers'")

        # Mode detection
        self.is_multi_ticker = bool(tickers)
        self.ticker = ticker

        # ✅ Remove duplicates from tickers list (some may appear in multiple categories)
        if tickers:
            unique_tickers = sorted(list(set(tickers)))
            if len(unique_tickers) != len(tickers):
                logger.warning(f"Removed {len(tickers) - len(unique_tickers)} duplicate tickers. Using {len(unique_tickers)} unique tickers.")
            self.tickers = unique_tickers
        else:
            self.tickers = [ticker]

        # Data params
        self.start_date = start_date
        self.lookback = lookback
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Model params
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # MLflow
        self.tracking_uri = f"file:{Path.cwd()}/{tracking_uri.replace('file:', '')}"
        self.experiment_name = experiment_name or (
            "lstm-multi-ticker" if self.is_multi_ticker else f"lstm-{ticker.replace('.SA', '').lower()}"
        )

        # Paths
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = get_device() if device in [None, "auto"] else torch.device(device)

        # Seed (expose as attribute for tests)
        self.seed = seed

        # State
        self.data = None
        self.model = None
        self.trainer = None

        logger.info(f"{'='*80}")
        logger.info("🚀 Initialized TrainPipeline")
        logger.info(f"{'='*80}")
        logger.info(f"Mode: {'Multi-ticker' if self.is_multi_ticker else 'Single-ticker'}")
        logger.info(f"Tickers: {', '.join(self.tickers)}")
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Tracking URI: {self.tracking_uri}")
        logger.info(f"{'='*80}\n")

    def run(self) -> Dict:
        """Execute full training pipeline.
        
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("🏃 Starting training pipeline...")

        # 1. Prepare data
        self.data = self._prepare_data()

        # 2. Create model
        self.model = self._create_model()

        # 3. Create dataloaders
        train_loader, val_loader, test_loader = self._create_dataloaders()

        # 4. Save scalers to disk for MLflow artifacts
        import json

        import joblib

        scaler_dir = self.model_save_path.parent / "scalers"
        scaler_dir.mkdir(exist_ok=True, parents=True)

        # Save X features scaler (19 columns)
        scaler_path = scaler_dir / "scaler.pkl"
        joblib.dump(self.data["scaler"], scaler_path)
        logger.info(f"💾 Saved X scaler (19 features): {scaler_path}")

        # Save y target scaler (1 column) if available
        y_scaler_path = scaler_dir / "y_scaler.pkl"
        if "y_scaler" in self.data and self.data["y_scaler"] is not None:
            joblib.dump(self.data["y_scaler"], y_scaler_path)
            logger.info(f"💾 Saved y scaler (1 feature): {y_scaler_path}")

        # Save preprocessing config (ticker mappings, etc.)
        preprocessing_config_path = scaler_dir / "preprocessing_config.json"
        preprocessing_config = {
            "num_features": self.data["num_features"],
            "num_tickers": self.data["num_tickers"],
            "lookback": self.lookback,
        }

        if "feature_cols" in self.data and self.data["feature_cols"]:
            preprocessing_config["feature_cols"] = self.data["feature_cols"]

        if "ticker_to_id" in self.data:
            preprocessing_config["ticker_to_id"] = self.data["ticker_to_id"]
            preprocessing_config["ticker_list"] = list(self.data["ticker_to_id"].keys())

        with open(preprocessing_config_path, 'w') as f:
            json.dump(preprocessing_config, f, indent=2)
        logger.info(f"💾 Saved preprocessing config: {preprocessing_config_path}")

        # 5. Create trainer and train
        self.trainer = self._create_trainer()
        training_history = self.trainer.train(train_loader, val_loader, epochs=self.epochs)

        # 5. Calculate test metrics (use y_scaler for denormalization)
        y_scaler = self.data.get("y_scaler", self.data.get("scaler"))  # Fallback to old scaler
        test_metrics = self.trainer.evaluate(test_loader, scaler=y_scaler)

        # 6. Get actual saved model path (Trainer saves as best_model.pt)
        actual_model_path = self.trainer.checkpoint_dir / "best_model.pt"

        # 7. Build results dictionary
        results = {
            "model_path": str(actual_model_path),
            "training_history": training_history,
            "test_metrics": test_metrics,
            "metadata": {
                "ticker": self.ticker if not self.is_multi_ticker else ", ".join(self.tickers),
                "epochs_trained": len(training_history["epoch"]),
                "best_val_loss": min(training_history["val_loss"]),
                "num_tickers": self.data["num_tickers"],
                "lookback": self.lookback,
            }
        }

        # 8. Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.success("✅ Training pipeline completed!")
        return results

    def _prepare_data(self) -> Dict:
        """Prepare data for training (single or multi-ticker).
        
        Returns:
            Dictionary with train/val/test splits and metadata
        """
        if self.is_multi_ticker:
            return self._prepare_multi_ticker_data()
        else:
            return self._prepare_single_ticker_data()

    def _prepare_single_ticker_data(self) -> Dict:
        """Prepare data for single ticker."""
        logger.info(f"📊 Preparing data for {self.ticker}...")

        # Ingest data
        ingestion = StockDataIngestion(
            ticker=self.ticker,
            start_date=self.start_date
        )
        df = ingestion.download_and_validate()

        # Feature engineering (using only SMA_20 and SMA_50 to match API with 60-day window)
        feature_eng = TechnicalIndicators(df)
        df_features = feature_eng.add_all_indicators(sma_windows=[20, 50])
        df_features = feature_eng.fill_missing_values()

        # Derivar dinamicamente as colunas de features (esperado: 18 após indicadores com SMA_20 e SMA_50)
        feature_cols = list(df_features.columns)
        if len(feature_cols) != 18:
            raise ValueError(
                f"Esperado 18 features após engenharia, obtido {len(feature_cols)}: {feature_cols}"
            )

        # Preprocessing
        preprocessor = StockPreprocessor(
            lookback_period=self.lookback,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            feature_cols=feature_cols,
        )

        data = preprocessor.prepare_data(df_features)

        # Data is already PyTorch tensors, just move to device
        X_train_tensor = data["X_train"].to(self.device)
        y_train_tensor = data["y_train"].to(self.device)
        X_val_tensor = data["X_val"].to(self.device)
        y_val_tensor = data["y_val"].to(self.device)
        X_test_tensor = data["X_test"].to(self.device)
        y_test_tensor = data["y_test"].to(self.device)

        logger.success(f"✓ Prepared {len(X_train_tensor)} train sequences")

        return {
            "X_train": X_train_tensor,
            "y_train": y_train_tensor,
            "X_val": X_val_tensor,
            "y_val": y_val_tensor,
            "X_test": X_test_tensor,
            "y_test": y_test_tensor,
            "scaler": preprocessor.scaler,
            "ticker_to_id": {self.ticker: 0},
            "num_tickers": 1,
            "num_features": X_train_tensor.shape[2],
            "feature_cols": feature_cols,
        }

    def _prepare_multi_ticker_data(self) -> Dict:
        """Prepare combined data for multiple tickers."""
        logger.info(f"📊 Preparing data for {len(self.tickers)} tickers...")

        ticker_to_id = {ticker: idx for idx, ticker in enumerate(self.tickers)}

        # Schema de referência: todas as ações devem ter as mesmas colunas/ordem
        base_feature_cols = None

        all_sequences_X = []
        all_sequences_y = []
        ticker_ids_list = []
        scalers_list = []  # For X features
        y_scalers_list = []  # For y targets

        # Process each ticker
        for idx, ticker in enumerate(self.tickers, 1):
            try:
                logger.info(f"[{idx}/{len(self.tickers)}] Processing {ticker}...")

                # Ingest
                ingestion = StockDataIngestion(ticker=ticker, start_date=self.start_date)
                df = ingestion.download_and_validate()

                # Feature engineering
                feature_eng = TechnicalIndicators(df)
                df_features = feature_eng.add_all_indicators(sma_windows=[20, 50])
                df_features = feature_eng.fill_missing_values()

                feature_cols = list(df_features.columns)
                if len(feature_cols) != 19:
                    raise ValueError(
                        f"Esperado 19 features para {ticker}, obtido {len(feature_cols)}: {feature_cols}"
                    )

                if base_feature_cols is None:
                    base_feature_cols = feature_cols
                elif feature_cols != base_feature_cols:
                    raise ValueError(
                        f"Schema de features divergente em {ticker}. Esperado {base_feature_cols}, obtido {feature_cols}"
                    )

                # Preprocessing
                preprocessor = StockPreprocessor(
                    lookback_period=self.lookback,
                    feature_cols=feature_cols,
                )
                X, y = preprocessor.create_sequences(df_features.values)

                if len(X) == 0:
                    logger.warning(f"⚠️ No sequences for {ticker}, skipping")
                    continue

                # Normalize X
                X_normalized = preprocessor.normalize(X.reshape(-1, X.shape[2]))
                X_normalized = X_normalized.reshape(X.shape)

                # ✅ CRITICAL FIX: Normalize y (target) separately
                # y is the Close price (column 0), needs its own scaler [0,1]
                from sklearn.preprocessing import MinMaxScaler
                y_scaler = MinMaxScaler(feature_range=(0, 1))
                y_normalized = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

                # Convert to tensors
                X_tensor = torch.FloatTensor(X_normalized).to(self.device)
                y_tensor = torch.FloatTensor(y_normalized).to(self.device)  # Already 1D, no unsqueeze needed

                # Create ticker IDs
                ticker_id_tensor = torch.full((len(X_tensor),), ticker_to_id[ticker], dtype=torch.long).to(self.device)

                all_sequences_X.append(X_tensor)
                all_sequences_y.append(y_tensor)
                ticker_ids_list.append(ticker_id_tensor)
                scalers_list.append(preprocessor.scaler)  # X features scaler
                y_scalers_list.append(y_scaler)  # y target scaler

                logger.success(f"  ✓ {len(X_tensor)} sequences")

            except Exception as e:
                logger.error(f"  ✗ Failed to process {ticker}: {e}")
                continue

        if not all_sequences_X:
            raise ValueError("No valid data for any ticker")

        # Combine all data
        logger.info("\n📦 Combining all data...")
        X_combined = torch.cat(all_sequences_X, dim=0)
        y_combined = torch.cat(all_sequences_y, dim=0)
        ticker_ids_combined = torch.cat(ticker_ids_list, dim=0)

        logger.info(f"  - Total sequences: {X_combined.shape[0]}")
        logger.info(f"  - Features: {X_combined.shape[2]}")
        logger.info(f"  - Tickers processed: {len(scalers_list)}/{len(self.tickers)}")

        # Shuffle and split
        logger.info("\n🔀 Shuffling and splitting...")
        n_total = X_combined.shape[0]
        indices = torch.randperm(n_total)

        X_shuffled = X_combined[indices]
        y_shuffled = y_combined[indices]
        ticker_ids_shuffled = ticker_ids_combined[indices]

        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        X_train = X_shuffled[:n_train]
        y_train = y_shuffled[:n_train]
        ticker_ids_train = ticker_ids_shuffled[:n_train]

        X_val = X_shuffled[n_train:n_train + n_val]
        y_val = y_shuffled[n_train:n_train + n_val]
        ticker_ids_val = ticker_ids_shuffled[n_train:n_train + n_val]

        X_test = X_shuffled[n_train + n_val:]
        y_test = y_shuffled[n_train + n_val:]
        ticker_ids_test = ticker_ids_shuffled[n_train + n_val:]

        logger.success(f"✓ Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}\n")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "ticker_ids_train": ticker_ids_train,
            "X_val": X_val,
            "y_val": y_val,
            "ticker_ids_val": ticker_ids_val,
            "X_test": X_test,
            "y_test": y_test,
            "ticker_ids_test": ticker_ids_test,
            "scaler": scalers_list[0] if scalers_list else None,  # X features scaler (for compatibility)
            "y_scaler": y_scalers_list[0] if y_scalers_list else None,  # y target scaler (CRITICAL for denormalization)
            "ticker_to_id": ticker_to_id,
            "num_tickers": len(self.tickers),
            "num_features": X_combined.shape[2],
            "feature_cols": base_feature_cols,
        }

    def _create_model(self):
        """Create LSTM model (single or multi-ticker)."""
        logger.info("🏗️ Creating model...")

        # Even for single-ticker, we need embedding > 0 (model architecture requires it)
        # For single-ticker: use small embedding (4), for multi-ticker: use configured embedding_dim
        embedding_dim_to_use = 4 if not self.is_multi_ticker else self.embedding_dim

        model = create_model(
            num_tickers=self.data["num_tickers"],
            num_features=self.data["num_features"],
            embedding_dim=embedding_dim_to_use,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device
        )

        logger.info(f"  - Num tickers: {self.data['num_tickers']}")
        logger.info(f"  - Num features: {self.data['num_features']}")
        logger.info(f"  - Embedding dim: {embedding_dim_to_use}")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Num layers: {self.num_layers}")
        logger.info(f"  - Dropout: {self.dropout}")
        logger.success("✓ Model created\n")

        return model

    def _create_dataloaders(self):
        """Create train/val/test dataloaders."""
        if self.is_multi_ticker:
            # Order: (X, y, ticker_ids) to match trainer expectations
            train_dataset = TensorDataset(
                self.data["X_train"],
                self.data["y_train"],
                self.data["ticker_ids_train"]
            )
            val_dataset = TensorDataset(
                self.data["X_val"],
                self.data["y_val"],
                self.data["ticker_ids_val"]
            )
            test_dataset = TensorDataset(
                self.data["X_test"],
                self.data["y_test"],
                self.data["ticker_ids_test"]
            )
        else:
            # Single-ticker: create dummy ticker_ids (all zeros)
            train_dataset = TensorDataset(
                self.data["X_train"],
                self.data["y_train"],
                torch.zeros(len(self.data["X_train"]), dtype=torch.long).to(self.device)
            )
            val_dataset = TensorDataset(
                self.data["X_val"],
                self.data["y_val"],
                torch.zeros(len(self.data["X_val"]), dtype=torch.long).to(self.device)
            )
            test_dataset = TensorDataset(
                self.data["X_test"],
                self.data["y_test"],
                torch.zeros(len(self.data["X_test"]), dtype=torch.long).to(self.device)
            )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _create_trainer(self):
        """Create Trainer with MLflow tracking."""
        # Scaler paths for MLflow artifacts
        scaler_dir = self.model_save_path.parent / "scalers"
        scaler_path = scaler_dir / "scaler.pkl"
        y_scaler_path = scaler_dir / "y_scaler.pkl"
        preprocessing_config_path = scaler_dir / "preprocessing_config.json"

        # Extra params for MLflow
        extra_params = {
            "tickers": ", ".join(self.tickers),
            "num_tickers": self.data["num_tickers"],
            "lookback": self.lookback,
            "model_type": "multi-ticker" if self.is_multi_ticker else "single-ticker",
            "ticker_to_id": str(self.data["ticker_to_id"]),
            "scaler_path": str(scaler_path) if scaler_path.exists() else None,
            "y_scaler_path": str(y_scaler_path) if y_scaler_path.exists() else None,
            "preprocessing_config_path": str(preprocessing_config_path) if preprocessing_config_path.exists() else None,
        }

        trainer = Trainer(
            model=self.model,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            checkpoint_dir=str(self.model_save_path.parent),  # Use directory from model_save_path
            early_stopping_patience=self.early_stopping_patience,
            extra_params=extra_params
        )

        return trainer

    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Remove old versioned checkpoints."""
        try:
            checkpoint_dir = self.model_save_path.parent
            checkpoints = sorted(
                [f for f in checkpoint_dir.glob("best_model_*.pt") if f.name != "best_model.pt"],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            for old_checkpoint in checkpoints[keep:]:
                old_checkpoint.unlink()
                logger.debug(f"🧹 Removed old checkpoint: {old_checkpoint.name}")

            if len(checkpoints) > keep:
                logger.info(f"🧹 Cleaned up {len(checkpoints) - keep} old checkpoints (kept {keep} most recent)")
        except Exception as e:
            logger.debug(f"Checkpoint cleanup failed: {e}")
