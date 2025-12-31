import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

from src.api.services.model_service import ModelService
from src.api.services.data_service import DataService
from src.api.utils.validators import normalize_ticker
from src.api.utils.exceptions import (
    ModelInferenceError,
    InvalidTickerError,
    TickerNotFoundError,
    InsufficientDataError,
    ServiceUnavailableError
)

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
            current_price = float(df['Close'].iloc[-1])
            
            close_prices = df['Close'].values.reshape(-1, 1)
            
            scaler = self.model_service.get_scaler()
            scaled_data = scaler.transform(close_prices)
            
            X = torch.FloatTensor(scaled_data).unsqueeze(0)
            
            model = self.model_service.get_model()
            with torch.no_grad():
                prediction_scaled = model(X)
            
            prediction_scaled_np = prediction_scaled.numpy().reshape(-1, 1)
            predicted_price = float(scaler.inverse_transform(prediction_scaled_np)[0, 0])
            
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
            
            return {
                "ticker": ticker,
                "predicted_price": round(predicted_price, 2),
                "current_price": round(current_price, 2),
                "change_percent": round(change_percent, 2),
                "change_direction": "up" if change_percent > 0 else "down" if change_percent < 0 else "neutral",
                "prediction_date": prediction_date,
                "confidence": confidence,
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
