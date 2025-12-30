"""
Serviço para buscar dados de ações via yfinance.

Responsável por baixar dados históricos e validar disponibilidade.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

from src.api.utils.exceptions import (
    TickerNotFoundError,
    InsufficientDataError,
    ServiceUnavailableError
)

logger = logging.getLogger(__name__)


class DataService:
    """
    Serviço para buscar dados de ações.
    
    Usa yfinance para obter dados históricos de preços.
    """
    
    def __init__(self, lookback_days: int = 60):
        """
        Inicializa serviço de dados.
        
        Args:
            lookback_days (int): Número de dias históricos necessários.
        """
        self.lookback_days = lookback_days
    
    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        Busca dados históricos de um ticker.
        
        Args:
            ticker (str): Símbolo da ação (ex: AAPL, PETR4.SA).
        
        Returns:
            pd.DataFrame: DataFrame com colunas [Open, High, Low, Close, Volume].
        
        Raises:
            TickerNotFoundError: Se ticker não for encontrado.
            InsufficientDataError: Se não houver dados suficientes.
            ServiceUnavailableError: Se Yahoo Finance estiver indisponível.
        """
        try:
            # Calcular período (adicionar margem para dias não-úteis)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 30)
            
            logger.info(f"Buscando dados para {ticker} de {start_date.date()} a {end_date.date()}")
            
            # Baixar dados
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            # Validar dados
            if df.empty:
                raise TickerNotFoundError(ticker)
            
            # Verificar se tem dados suficientes
            if len(df) < self.lookback_days:
                raise InsufficientDataError(
                    ticker=ticker,
                    days_available=len(df),
                    days_required=self.lookback_days
                )
            
            # Pegar últimos N dias
            df = df.tail(self.lookback_days)
            
            logger.info(f"Dados obtidos: {len(df)} registros, último preço: {df['Close'].iloc[-1]:.2f}")
            
            return df
            
        except (TickerNotFoundError, InsufficientDataError):
            raise  # Re-raise custom exceptions
        except ConnectionError as e:
            logger.error(f"Erro de conexão ao buscar dados para {ticker}: {str(e)}")
            raise ServiceUnavailableError(service="Yahoo Finance", retry_after=60)
        except TimeoutError as e:
            logger.error(f"Timeout ao buscar dados para {ticker}: {str(e)}")
            raise ServiceUnavailableError(service="Yahoo Finance", retry_after=30)
        except Exception as e:
            logger.error(f"Erro inesperado ao buscar dados para {ticker}: {str(e)}")
            # Se for erro de rede, tratar como serviço indisponível
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise ServiceUnavailableError(service="Yahoo Finance")
            # Caso contrário, pode ser ticker inválido
            raise TickerNotFoundError(ticker)
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Obtém último preço de fechamento de um ticker.
        
        Args:
            ticker (str): Símbolo da ação.
        
        Returns:
            float: Último preço de fechamento, ou None se não disponível.
        """
        try:
            df = self.fetch_data(ticker)
            return float(df['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Não foi possível obter último preço para {ticker}: {str(e)}")
            return None
