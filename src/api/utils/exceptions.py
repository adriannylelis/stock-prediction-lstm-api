"""
Exceções customizadas para a API.

Define exceções específicas para diferentes tipos de erros.
"""


class APIException(Exception):
    """Exceção base para erros da API."""
    
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        """
        Inicializa exceção da API.
        
        Args:
            message (str): Mensagem de erro.
            status_code (int): Código HTTP do erro.
            details (dict): Detalhes adicionais do erro.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}
    
    def to_dict(self):
        """
        Converte exceção para dicionário JSON.
        
        Returns:
            dict: Dicionário com estrutura de erro.
        """
        error_dict = {
            "error": self.__class__.__name__.replace("Exception", "").replace("Error", ""),
            "message": self.message,
            "status": self.status_code
        }
        
        if self.details:
            error_dict["details"] = self.details
        
        return error_dict


class InvalidTickerError(APIException):
    """Ticker inválido ou formato incorreto."""
    
    def __init__(self, ticker: str, suggestion: str = None):
        details = {"ticker": ticker}
        if suggestion:
            details["suggestion"] = suggestion
        
        super().__init__(
            message=f"Ticker '{ticker}' é inválido ou não encontrado",
            status_code=400,
            details=details
        )


class InsufficientDataError(APIException):
    """Dados insuficientes para realizar previsão."""
    
    def __init__(self, ticker: str, days_available: int, days_required: int):
        super().__init__(
            message=f"Não há dados suficientes para previsão de {ticker}",
            status_code=400,
            details={
                "ticker": ticker,
                "days_available": days_available,
                "days_required": days_required,
                "suggestion": f"Ticker precisa ter pelo menos {days_required} dias de histórico"
            }
        )


class TickerNotFoundError(APIException):
    """Ticker não encontrado no Yahoo Finance."""
    
    def __init__(self, ticker: str):
        super().__init__(
            message=f"Ticker '{ticker}' não encontrado",
            status_code=404,
            details={
                "ticker": ticker,
                "suggestion": "Verifique se o ticker está correto. Exemplos: AAPL, PETR4.SA, VALE3.SA"
            }
        )


class ModelInferenceError(APIException):
    """Erro durante inferência do modelo."""
    
    def __init__(self, ticker: str, error_detail: str = None):
        details = {"ticker": ticker}
        if error_detail:
            details["error_detail"] = error_detail
        
        super().__init__(
            message="Erro ao realizar inferência do modelo",
            status_code=500,
            details=details
        )


class ServiceUnavailableError(APIException):
    """Serviço externo indisponível (ex: Yahoo Finance)."""
    
    def __init__(self, service: str = "Yahoo Finance", retry_after: int = None):
        details = {"service": service}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=f"{service} está temporariamente indisponível",
            status_code=503,
            details=details
        )
