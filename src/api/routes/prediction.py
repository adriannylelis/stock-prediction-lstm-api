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

predict_service = None


def get_predict_service():
    global predict_service
    if predict_service is None:
        predict_service = PredictService()
    return predict_service


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para predição de preço de ação.
    
    Query Parameters:
        include_history (bool): Se 'true', inclui últimos 30 dias de dados históricos
        
    Request Body:
        ticker (str): Símbolo da ação (ex: AAPL, PETR4.SA)
        
    Returns:
        JSON com predição e opcionalmente dados históricos
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Invalid Content-Type",
                "message": "Content-Type deve ser application/json",
                "status": 400
            }), 400
        
        data = request.get_json()
        
        if 'ticker' not in data:
            return jsonify({
                "error": "Missing Field",
                "message": "Campo 'ticker' é obrigatório",
                "status": 400
            }), 400
        
        ticker = data['ticker']
        
        include_history = request.args.get('include_history', 'false').lower() == 'true'
        
        is_valid, error_message = validate_ticker(ticker)
        if not is_valid:
            raise InvalidTickerError(ticker=ticker, suggestion=error_message)
        
        service = get_predict_service()
        result = service.predict(ticker, include_history=include_history)
        
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
        current_app.logger.error(f"Erro da API: {str(e)}")
        return jsonify(e.to_dict()), e.status_code
        
    except Exception as e:
        current_app.logger.error(f"Erro inesperado na previsão: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": "Erro interno do servidor",
            "status": 500
        }), 500
