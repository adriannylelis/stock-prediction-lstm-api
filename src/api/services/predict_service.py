"""
Serviço de previsão integrado.

Orquestra busca de dados, normalização e inferência do modelo.
"""

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
    """
    Serviço completo de previsão.
    
    Combina data fetching, normalização e inferência do modelo
    para gerar previsões de preço.
    """
    
    def __init__(self):
        """Inicializa serviços necessários."""
        self.model_service = ModelService()
        self.data_service = DataService(lookback_days=60)
        
        # Validar que serviços foram inicializados
        if not self.model_service.is_ready():
            raise RuntimeError("ModelService não foi inicializado corretamente")
    
    def predict(self, ticker: str) -> Dict[str, Any]:
        """
        Realiza previsão de preço para um ticker.
        
        Pipeline completo:
        1. Normalizar ticker
        2. Buscar dados históricos (60 dias)
        3. Normalizar com scaler
        4. Fazer inferência com modelo
        5. Desnormalizar resultado
        6. Calcular métricas adicionais
        
        Args:
            ticker (str): Símbolo da ação.
        
        Returns:
            dict: Resultado com previsão e metadados.
        
        Example:
            {
                "ticker": "AAPL",
                "predicted_price": 178.45,
                "current_price": 175.20,
                "change_percent": 1.85,
                "prediction_date": "2025-12-30",
                "confidence": "medium",
                "timestamp": "2025-12-29T10:30:00"
            }
        """
        try:
            # 1. Normalizar ticker
            ticker = normalize_ticker(ticker)
            logger.info(f"Iniciando previsão para {ticker}")
            
            # 2. Buscar dados históricos (pode lançar exceções customizadas)
            df = self.data_service.fetch_data(ticker)
            current_price = float(df['Close'].iloc[-1])
            
            # 3. Preparar sequência (apenas Close price)
            close_prices = df['Close'].values.reshape(-1, 1)
            
            # 4. Normalizar dados
            scaler = self.model_service.get_scaler()
            scaled_data = scaler.transform(close_prices)
            
            # 5. Criar tensor PyTorch [batch_size, seq_len, features]
            # batch_size=1, seq_len=60, features=1
            X = torch.FloatTensor(scaled_data).unsqueeze(0)
            
            # 6. Fazer inferência
            model = self.model_service.get_model()
            with torch.no_grad():
                prediction_scaled = model(X)
            
            # 7. Desnormalizar previsão
            prediction_scaled_np = prediction_scaled.numpy().reshape(-1, 1)
            predicted_price = float(scaler.inverse_transform(prediction_scaled_np)[0, 0])
            
            # 8. Calcular métricas
            change_percent = ((predicted_price - current_price) / current_price) * 100
            prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # 9. Determinar nível de confiança baseado em % de mudança
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
            
            # 10. Retornar resultado
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
            # Re-raise exceções customizadas que já estão corretas
            raise
            
        except Exception as e:
            # Erro inesperado - tentar inferir o tipo
            error_msg = str(e).lower()
            
            # Erros de inferência do modelo (tensor/shape errors)
            if "tensor" in error_msg or "shape" in error_msg or "dimension" in error_msg:
                logger.error(f"Erro de inferência do modelo para {ticker}: {str(e)}", exc_info=True)
                raise ModelInferenceError(ticker=ticker, error_detail=str(e))
            
            # Erro genérico
            logger.error(f"Erro inesperado ao realizar previsão para {ticker}: {str(e)}", exc_info=True)
            raise ModelInferenceError(ticker=ticker, error_detail=str(e))
