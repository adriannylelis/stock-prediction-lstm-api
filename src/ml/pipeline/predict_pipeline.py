"""Prediction pipeline orchestration.

Complete end-to-end prediction pipeline: data ingestion → preprocessing → inference.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from loguru import logger

from ..data.feature_engineering import TechnicalIndicators
from ..data.ingestion import StockDataIngestion
from ..data.preprocessing import StockPreprocessor
from ..models.lstm import StockLSTM


class PredictPipeline:
    """End-to-end prediction pipeline.

    Orchestrates batch predictions:
    1. Load trained model
    2. Ingest latest data
    3. Feature engineering
    4. Preprocessing
    5. Multi-step predictions
    6. Denormalization

    Example:
        >>> pipeline = PredictPipeline(
        ...     model_path="artifacts/models/best_model.pt", ticker="PETR4.SA"
        ... )
        >>> predictions_df = pipeline.predict(days_ahead=5)
        >>> print(predictions_df)
    """

    def __init__(
        self, model_path: str, ticker: str, lookback: int = 60, device: Optional[str] = None
    ):
        """Initialize prediction pipeline.

        Args:
            model_path: Path to trained model (.pt file).
            ticker: Stock ticker symbol.
            lookback: Lookback period (must match training).
            device: Device (cpu/cuda/auto).
        """
        self.model_path = Path(model_path)
        self.ticker = ticker
        self.lookback = lookback

        if device == "auto" or device is None:
            from ..utils.device import get_device

            self.device = get_device()
        else:
            self.device = torch.device(device)

        # Load model
        self.model = None
        self.scaler = None
        self._load_model()

        logger.info(f"Initialized PredictPipeline for {ticker}")

    def _load_model(self):
        """Load trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Create model
        self.model = StockLSTM(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.success(
            f"✓ Model loaded: {checkpoint['hidden_size']} hidden, {checkpoint['num_layers']} layers"
        )

    def predict(self, days_ahead: int = 5, output_path: Optional[str] = None) -> pd.DataFrame:
        """Generate multi-step predictions.

        Args:
            days_ahead: Number of days to predict.
            output_path: Path to save predictions CSV (optional).

        Returns:
            DataFrame with columns: Date, Predicted_Close
        """
        logger.info(f"🔮 Generating {days_ahead}-day predictions for {self.ticker}")

        # Step 1: Ingest latest data
        logger.info("📥 Step 1/4: Data Ingestion")
        df = self._ingest_latest_data()
        logger.success(f"✓ Fetched {len(df)} records")

        # Step 2: Feature engineering
        logger.info("🔧 Step 2/4: Feature Engineering")
        df = self._engineer_features(df)
        logger.success(f"✓ Generated {df.shape[1]} features")

        # Step 3: Preprocess
        logger.info("⚙️ Step 3/4: Preprocessing")
        last_sequence = self._preprocess_latest(df)
        logger.success(f"✓ Prepared sequence: {last_sequence.shape}")

        # Step 4: Multi-step prediction
        logger.info(f"🔮 Step 4/4: Predicting {days_ahead} days ahead")
        predictions = self._predict_multi_step(last_sequence, days_ahead)
        logger.success(f"✓ Generated {len(predictions)} predictions")

        # Format results
        results_df = self._format_results(df, predictions)

        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"💾 Saved predictions to: {output_path}")

        return results_df

    def _ingest_latest_data(self):
        """Ingest latest data from yfinance."""
        from datetime import datetime, timedelta

        # Get last 2 years of data to have enough for indicators + lookback
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # ~2 years

        ingestion = StockDataIngestion(ticker=self.ticker, start_date=start_date, end_date=end_date)
        return ingestion.download_and_validate()

    def _engineer_features(self, df):
        """Add technical indicators."""
        tech_ind = TechnicalIndicators(df)
        df = tech_ind.add_all_indicators()
        df = tech_ind.fill_missing_values()
        return df

    def _preprocess_latest(self, df):
        """Preprocess latest data for prediction."""
        # Take only last lookback points from Close column
        last_data = df[["Close"]].tail(self.lookback).values

        # Normalize
        preprocessor = StockPreprocessor(lookback_period=self.lookback)
        normalized = preprocessor.normalize(last_data, fit=True)

        # Store scaler for denormalization
        self.scaler = preprocessor.scaler

        # Create sequence: (1, lookback, 1) shape
        X = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        return X

    def _predict_multi_step(self, initial_sequence, days_ahead: int) -> List[float]:
        """Generate multi-step predictions (rolling window).

        Args:
            initial_sequence: Initial sequence tensor [1, lookback, features].
            days_ahead: Number of days to predict.

        Returns:
            List of denormalized predictions.
        """
        predictions = []
        current_sequence = initial_sequence

        with torch.no_grad():
            for day in range(days_ahead):
                # Predict next value
                pred = self.model(current_sequence)
                predictions.append(pred.cpu().item())

                # Update sequence (rolling window)
                # Remove oldest, append prediction
                new_point = pred.unsqueeze(1)  # [1, 1, 1]
                current_sequence = torch.cat([current_sequence[:, 1:, :], new_point], dim=1)

        # Denormalize predictions
        predictions_denorm = self.scaler.inverse_transform([[p] for p in predictions])
        predictions_denorm = [p[0] for p in predictions_denorm]

        return predictions_denorm

    def _format_results(self, historical_df, predictions: List[float]) -> pd.DataFrame:
        """Format predictions as DataFrame.

        Args:
            historical_df: Historical data (for last date).
            predictions: List of predicted prices.

        Returns:
            DataFrame with Date and Predicted_Close columns.
        """
        last_date = historical_df.index[-1]

        # Generate future dates (business days)
        prediction_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=len(predictions), freq="D"
        )

        results_df = pd.DataFrame({"Date": prediction_dates, "Predicted_Close": predictions})

        return results_df
