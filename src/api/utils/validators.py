"""
Validadores para entrada de dados da API.

Funções para validar tickers, datas e outros inputs.
"""

import re
from typing import Tuple


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Valida formato de um ticker de ação.
    
    Regras:
        - Deve ter entre 2 e 10 caracteres
        - Apenas letras maiúsculas, números, pontos e hífens
        - Não pode começar com número
        - Exemplos válidos: AAPL, PETR4.SA, BRK-B
    
    Args:
        ticker (str): Ticker a ser validado.
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    
    Examples:
        >>> validate_ticker("AAPL")
        (True, "")
        
        >>> validate_ticker("A")
        (False, "Ticker deve ter entre 2 e 10 caracteres")
        
        >>> validate_ticker("123")
        (False, "Ticker não pode começar com número")
    """
    # Verificar se é string
    if not isinstance(ticker, str):
        return False, "Ticker deve ser uma string"
    
    # Remover espaços
    ticker = ticker.strip()
    
    # Verificar se está vazio
    if not ticker:
        return False, "Ticker não pode ser vazio"
    
    # Verificar tamanho mínimo
    if len(ticker) < 2:
        return False, "Ticker deve ter entre 2 e 10 caracteres"
    
    # Verificar tamanho máximo
    if len(ticker) > 10:
        return False, "Ticker não pode ter mais de 10 caracteres"
    
    # Verificar formato (letras, números, ponto, hífen)
    # Deve começar com letra, seguido de letras, números, ponto ou hífen
    pattern = r'^[A-Z][A-Z0-9\.\-]{1,9}$'
    if not re.match(pattern, ticker.upper()):
        return False, "Ticker deve conter apenas letras, números, pontos e hífens"
    
    # Validação passou
    return True, ""


def normalize_ticker(ticker: str) -> str:
    """
    Normaliza um ticker para formato padrão (maiúsculo, sem espaços).
    
    Args:
        ticker (str): Ticker a ser normalizado.
    
    Returns:
        str: Ticker normalizado.
    
    Examples:
        >>> normalize_ticker("  aapl  ")
        "AAPL"
    """
    return ticker.strip().upper()
