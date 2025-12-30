"""
Serviço para buscar dados de ações via yfinance.

Responsável por baixar dados históricos e validar disponibilidade.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

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
            ValueError: Se não conseguir obter dados suficientes.
            RuntimeError: Se ocorrer erro na busca.
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
                raise ValueError(f"Nenhum dado encontrado para ticker '{ticker}'")
            
            # Verificar se tem dados suficientes
            if len(df) < self.lookback_days:
                raise ValueError(
                    f"Dados insuficientes para {ticker}. "
                    f"Necessário: {self.lookback_days} dias, "
                    f"Obtido: {len(df)} dias"
                )
            
            # Pegar últimos N dias
            df = df.tail(self.lookback_days)
            
            logger.info(f"Dados obtidos: {len(df)} registros, último preço: {df['Close'].iloc[-1]:.2f}")
            
            return df
            
        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error(f"Erro ao buscar dados para {ticker}: {str(e)}")
            raise RuntimeError(f"Falha ao buscar dados do ticker: {str(e)}")
    
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
