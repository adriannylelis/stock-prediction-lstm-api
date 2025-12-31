import re
from typing import Tuple


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """Valida formato de ticker. Retorna (is_valid, error_message)."""
    if not isinstance(ticker, str):
        return False, "Ticker deve ser uma string"
    
    ticker = ticker.strip()
    
    if not ticker:
        return False, "Ticker não pode ser vazio"
    
    if len(ticker) < 2:
        return False, "Ticker deve ter entre 2 e 10 caracteres"
    
    if len(ticker) > 10:
        return False, "Ticker não pode ter mais de 10 caracteres"
    
    pattern = r'^[A-Z][A-Z0-9\.\-]{1,9}$'
    if not re.match(pattern, ticker.upper()):
        return False, "Ticker deve conter apenas letras, números, pontos e hífens"
    
    return True, ""


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()
