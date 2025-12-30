"""
Rota para previsões do modelo.

Endpoint para receber ticker e retornar previsão de preço.
"""

from flask import Blueprint, request, jsonify, current_app
from src.api.utils.validators import validate_ticker
from src.api.services.predict_service import PredictService
from src.api.utils.exceptions import (
    APIException,
    InvalidTickerError,
    TickerNotFoundError,
    InsufficientDataError,
    ModelInferenceError,
    ServiceUnavailableError
)

prediction_bp = Blueprint('prediction', __name__)

# Instância do serviço de previsão (singleton)
predict_service = None


def get_predict_service():
    """
    Obtém instância singleton do serviço de previsão.
    
    Returns:
        PredictService: Instância do serviço.
    """
    global predict_service
    if predict_service is None:
        predict_service = PredictService()
    return predict_service


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    Realiza previsão de preço para um ticker.
    
    Espera JSON no corpo da requisição:
        {
            "ticker": "AAPL"
        }
    
    Retorna:
        JSON com previsão e metadados.
        
    Exemplo de resposta:
        {
            "ticker": "AAPL",
            "predicted_price": 178.45,
            "current_price": 175.20,
            "change_percent": 1.85,
            "prediction_date": "2025-12-30",
            "confidence": "medium"
        }
    
    Status codes:
        200: Previsão realizada com sucesso
        400: Dados de entrada inválidos ou insuficientes
        404: Ticker não encontrado
        500: Erro interno do servidor / erro de inferência
        503: Serviço do Yahoo Finance indisponível
    """
    try:
        # Validar Content-Type
        if not request.is_json:
            return jsonify({
                "error": "Invalid Content-Type",
                "message": "Content-Type deve ser application/json",
                "status": 400
            }), 400
        
        # Obter dados do request
        data = request.get_json()
        
        # Validar presença do ticker
        if 'ticker' not in data:
            return jsonify({
                "error": "Missing Field",
                "message": "Campo 'ticker' é obrigatório",
                "status": 400
            }), 400
        
        ticker = data['ticker']
        
        # Validar formato do ticker
        is_valid, error_message = validate_ticker(ticker)
        if not is_valid:
            raise InvalidTickerError(ticker=ticker, suggestion=error_message)
        
        # Realizar previsão (pode lançar exceções customizadas)
        service = get_predict_service()
        result = service.predict(ticker)
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
        
    except InvalidTickerError as e:
        current_app.logger.warning(f"Ticker inválido: {str(e)}")
        return jsonify(e.to_dict()), e.status_code
        
    except TickerNotFoundError as e:
        current_app.logger.warning(f"Ticker não encontrado: {str(e)}")
        return jsonify(e.to_dict()), e.status_code
        
    except InsufficientDataError as e:
        current_app.logger.warning(f"Dados insuficientes: {str(e)}")
        return jsonify(e.to_dict()), e.status_code
        
    except ServiceUnavailableError as e:
        current_app.logger.error(f"Yahoo Finance indisponível: {str(e)}")
        return jsonify(e.to_dict()), e.status_code
        
    except ModelInferenceError as e:
        current_app.logger.error(f"Erro de inferência do modelo: {str(e)}")
        return jsonify(e.to_dict()), e.status_code
        
    except APIException as e:
        # Catch-all para outras exceções customizadas
        current_app.logger.error(f"Erro da API: {str(e)}")
        return jsonify(e.to_dict()), e.status_code
        
    except Exception as e:
        # Erro inesperado
        current_app.logger.error(f"Erro inesperado na previsão: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": "Erro interno do servidor",
            "status": 500
        }), 500
