from flask import Blueprint, current_app, jsonify, request

from src.api.utils.exceptions import (
    APIException,
    InsufficientDataError,
    InvalidTickerError,
    ModelInferenceError,
    ServiceUnavailableError,
    TickerNotFoundError,
)
from src.api.utils.validators import validate_ticker

prediction_bp = Blueprint('prediction', __name__)

predict_service = None


def get_predict_service():
    """Lazy-load PredictService to avoid import-time crash."""
    global predict_service
    if predict_service is None:
        from src.api.services.predict_service import PredictService
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

        is_valid, error_message = validate_ticker(ticker)
        if not is_valid:
            raise InvalidTickerError(ticker=ticker, suggestion=error_message)

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

    except RuntimeError as e:
        # Modelo não disponível (não treinado ou não encontrado no MLflow)
        if "Failed to load model" in str(e) or "Falha ao inicializar modelo" in str(e):
            current_app.logger.error(f"Modelo não disponível: {str(e)}")
            return jsonify({
                "error": "Service Unavailable",
                "message": "Modelo não está disponível. Por favor, treine um modelo primeiro usando o cli.",
                "details": str(e),
                "status": 503
            }), 503
        # Outros RuntimeErrors
        current_app.logger.error(f"Runtime error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": str(e),
            "status": 500
        }), 500

    except APIException as e:
        current_app.logger.error(f"Erro da API: {str(e)}")
        return jsonify(e.to_dict()), e.status_code

    except Exception as e:
        current_app.logger.error(f"Erro inesperado na previsão: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": "Erro interno do servidor. Verifique os logs para mais detalhes.",
            "details": str(e) if current_app.debug else None,
            "status": 500
        }), 500
