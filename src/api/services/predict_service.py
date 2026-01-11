import logging
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import torch

from src.api.services.data_service import DataService
from src.api.services.model_service import ModelService
from src.api.utils.exceptions import (
    InsufficientDataError,
    InvalidTickerError,
    ModelInferenceError,
    ServiceUnavailableError,
    TickerNotFoundError,
)
from src.api.utils.validators import normalize_ticker

logger = logging.getLogger(__name__)


class PredictService:
    """Orquestra pipeline completo de predição."""

    def __init__(self):
        self.model_service = ModelService()
        self.data_service = DataService(lookback_days=60)

        if not self.model_service.is_ready():
            raise RuntimeError("ModelService não foi inicializado corretamente")

    def predict(self, ticker: str) -> Dict[str, Any]:
        try:
            ticker = normalize_ticker(ticker)
            logger.info(f"Iniciando previsão para {ticker}")

            df = self.data_service.fetch_data(ticker)

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep only OHLCV columns to match training data (5 columns)
            # Remove: Adj Close, Dividends, Stock Splits
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            extra_cols = [col for col in df.columns if col not in required_cols]
            if extra_cols:
                df = df.drop(columns=extra_cols)
                logger.debug(f"Removed extra columns: {extra_cols}")

            # Add technical indicators (18 features total: 5 OHLCV + 13 indicadores)
            # Using SMA_20 and SMA_50 (no SMA_200 to match training with 60-day window)
            from src.ml.data.feature_engineering import TechnicalIndicators
            ti = TechnicalIndicators(df)
            df_features = ti.add_all_indicators(sma_windows=[20, 50])
            df_features = ti.fill_missing_values()

            # Handle MultiIndex again if created by indicators
            if isinstance(df_features.columns, pd.MultiIndex):
                df_features.columns = [col[0] if col[1] == '' else col[0] for col in df_features.columns]

            current_price = float(df_features['Close'].iloc[-1])
            # Get all 18 features (OHLCV + technical indicators)
            all_features = df_features.values  # Shape: (samples, 18)

            # Check for NaN in features
            print(f"🔍 Features shape: {all_features.shape}")
            print(f"🔍 Features has NaN: {pd.isna(all_features).any()}")
            nan_counts = pd.isna(all_features).sum(axis=0)
            print(f"🔍 NaN count per feature: {nan_counts}")
            if nan_counts.sum() > 0:
                print(f"❌ FOUND {nan_counts.sum()} NaN values in features!")
                print(f"   Columns with NaN: {list(df_features.columns[nan_counts > 0])}")

            scaler = self.model_service.get_scaler()
            scaled_data = scaler.transform(all_features)

            print(f"🔍 Scaled data has NaN: {pd.isna(scaled_data).any()}")

            model = self.model_service.get_model()

            # Detect device (CPU, CUDA, or MPS)
            device = next(model.parameters()).device

            # Get last 60 sequences (lookback window)
            lookback = 60
            if len(scaled_data) < lookback:
                raise InsufficientDataError(
                    ticker=ticker,
                    required=lookback,
                    available=len(scaled_data)
                )

            X = torch.FloatTensor(scaled_data[-lookback:]).unsqueeze(0).to(device)  # Shape: (1, 60, 18)

            # Get correct ticker_id (supports both single and multi-ticker models)
            ticker_id = self.model_service.get_ticker_id(ticker)
            ticker_ids = torch.tensor([ticker_id], dtype=torch.long).to(device)

            is_embedding_model = hasattr(model, "ticker_embedding")
            print(f"🔍 DEBUG Ticker: {ticker}, Ticker ID: {ticker_id}")
            print(f"🔍 DEBUG Is embedding model: {is_embedding_model}")
            if self.model_service.is_multi_ticker():
                logger.info(f"Using multi-ticker model: {ticker} → ID {ticker_id}")

            with torch.no_grad():
                if is_embedding_model:
                    prediction_scaled, _ = model(X, ticker_ids)  # Unpack tuple: (outputs, hidden_state)
                else:
                    prediction_scaled, _ = model(X)  # Unpack tuple: (outputs, hidden_state)

            # Debug prediction
            print(f"🔍 DEBUG Prediction shape: {prediction_scaled.shape}")
            print(f"🔍 DEBUG Prediction values: {prediction_scaled}")
            print(f"🔍 DEBUG Has NaN: {torch.isnan(prediction_scaled).any()}")
            print(f"🔍 DEBUG Has Inf: {torch.isinf(prediction_scaled).any()}")

            # Denormalize prediction using y_scaler (1 column)
            y_scaler = self.model_service.get_y_scaler()
            prediction_scaled_np = prediction_scaled.cpu().numpy().reshape(-1, 1)
            print(f"🔍 DEBUG Prediction scaled numpy: {prediction_scaled_np}")

            predicted_price_array = y_scaler.inverse_transform(prediction_scaled_np)
            print(f"🔍 DEBUG Prediction after inverse_transform: {predicted_price_array}")

            predicted_price = float(predicted_price_array[0, 0])

            change_percent = ((predicted_price - current_price) / current_price) * 100
            prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            if abs(change_percent) < 2:
                confidence = "high"
            elif abs(change_percent) < 5:
                confidence = "medium"
            else:
                confidence = "low"

            logger.info(
                f"Previsão concluída para {ticker}: "
                f"atual={current_price:.2f}, previsto={predicted_price:.2f}, "
                f"mudança={change_percent:.2f}%"
            )

            # Preparar histórico dos últimos 30 dias para gráfico
            history_days = min(30, len(df_features))
            history = []
            for i in range(-history_days, 0):
                date = df_features.index[i]
                close = float(df_features['Close'].iloc[i])
                history.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "price": round(close, 2)
                })

            return {
                "ticker": ticker,
                "prediction": {
                    "date": prediction_date,
                    "price": round(predicted_price, 2),
                    "change_percent": round(change_percent, 2),
                    "change_direction": "alta" if change_percent > 0 else "baixa" if change_percent < 0 else "neutra",
                    "confidence": confidence
                },
                "current": {
                    "price": round(current_price, 2),
                    "date": df_features.index[-1].strftime('%Y-%m-%d')
                },
                "history": history,
                "timestamp": datetime.utcnow().isoformat()
            }

        except (InvalidTickerError, TickerNotFoundError, InsufficientDataError,
                ServiceUnavailableError, ModelInferenceError):
            raise

        except Exception as e:
            error_msg = str(e).lower()

            if "tensor" in error_msg or "shape" in error_msg or "dimension" in error_msg:
                logger.error(f"Erro de inferência do modelo para {ticker}: {str(e)}", exc_info=True)
                raise ModelInferenceError(ticker=ticker, error_detail=str(e))

            logger.error(f"Erro inesperado ao realizar previsão para {ticker}: {str(e)}", exc_info=True)
            raise ModelInferenceError(ticker=ticker, error_detail=str(e))
