import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import pandas as pd

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
        self.data_service = DataService(lookback_days=60, min_days=30)
        
        if not self.model_service.is_ready():
            raise RuntimeError("ModelService não foi inicializado corretamente")
    
    def _pad_sequence(self, data: np.ndarray, target_length: int = 60) -> np.ndarray:

        current_length = len(data)
        
        if current_length >= target_length:
            return data[-target_length:]
        else:
            padding_length = target_length - current_length
            padding = np.zeros((padding_length, 1))
            return np.vstack([padding, data])
    
    def _format_historical_data(self, df: pd.DataFrame, days: int = 30) -> List[Dict[str, Any]]:

        historical_df = df.tail(days).copy()
        historical_df = historical_df.reset_index()
        
        historical_data = []
        for _, row in historical_df.iterrows():
            historical_data.append({
                "date": row['Date'].strftime('%Y-%m-%d'),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "volume": int(row['Volume'])
            })
        
        return historical_data
    
    def predict(self, ticker: str, include_history: bool = False) -> Dict[str, Any]:

        try:
            ticker = normalize_ticker(ticker)
            logger.info(f"Iniciando previsão para {ticker} (include_history={include_history})")
            
            df = self.data_service.fetch_data(ticker)
            days_available = len(df)
            logger.info(f"Dados disponíveis para {ticker}: {days_available} dias")
            
            current_price = float(df['Close'].iloc[-1])
            
            close_prices = df['Close'].values.reshape(-1, 1)
            
            scaler = self.model_service.get_scaler()
            scaled_data = scaler.transform(close_prices)
            
            if days_available < 60:
                logger.warning(f"{ticker}: apenas {days_available} dias disponíveis, aplicando zero-padding")
                scaled_data = self._pad_sequence(scaled_data, target_length=60)
            
            X = torch.FloatTensor(scaled_data).unsqueeze(0)
            
            model = self.model_service.get_model()
            with torch.no_grad():
                prediction_scaled = model(X)
            
            prediction_scaled_np = prediction_scaled.numpy().reshape(-1, 1)
            predicted_price = float(scaler.inverse_transform(prediction_scaled_np)[0, 0])
            
            change_percent = ((predicted_price - current_price) / current_price) * 100
            prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if abs(change_percent) < 2:
                confidence = "alta"
            elif abs(change_percent) < 5:
                confidence = "média"
            else:
                confidence = "baixa"
            
            logger.info(
                f"Previsão concluída para {ticker}: "
                f"atual={current_price:.2f}, previsto={predicted_price:.2f}, "
                f"mudança={change_percent:.2f}%"
            )
            
            result = {
                "ticker": ticker,
                "predicted_price": round(predicted_price, 2),
                "current_price": round(current_price, 2),
                "change_percent": round(change_percent, 2),
                "change_direction": "alta" if change_percent > 0 else "baixa" if change_percent < 0 else "neutra",
                "prediction_date": prediction_date,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Adicionar dados históricos se solicitado
            if include_history:
                result["historical_data"] = self._format_historical_data(df, days=30)
                logger.info(f"Incluídos {len(result['historical_data'])} dias de dados históricos")
            
            return result
            
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
