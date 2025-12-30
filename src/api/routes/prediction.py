"""
Rota para previsões do modelo.

Endpoint para receber ticker e retornar previsão de preço.
"""

from flask import Blueprint, request, jsonify, current_app
from src.api.utils.validators import validate_ticker
from src.api.services.predict_service import PredictService

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
            return jsonify({
                "error": "Invalid Ticker",
                "message": error_message,
                "status": 400
            }), 400
        
        # Realizar previsão
        service = get_predict_service()
        result = service.predict(ticker)
        
        return jsonify(result), 200
        
    except ValueError as e:
        current_app.logger.warning(f"Erro de validação: {str(e)}")
        return jsonify({
            "error": "Validation Error",
            "message": str(e),
            "status": 400
        }), 400
        
    except Exception as e:
        current_app.logger.error(f"Erro na previsão: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Prediction Error",
            "message": "Erro ao realizar previsão",
            "status": 500
        }), 500
