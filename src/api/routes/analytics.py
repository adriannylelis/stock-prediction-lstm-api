"""
Endpoints de analytics para histórico de predições.

Retorna predições passadas comparadas com valores reais
e métricas de acurácia do modelo.
"""
import logging
from flask import Blueprint, jsonify, request, current_app

from src.api.services.firestore_service import FirestoreService
from src.api.utils.validators import validate_ticker

analytics_bp = Blueprint('analytics', __name__)
logger = logging.getLogger(__name__)


@analytics_bp.route('/analytics/<ticker>', methods=['GET'])
def get_analytics(ticker: str):
    """
    Retorna histórico de predições e métricas de acurácia para um ticker.
    
    Args:
        ticker: Símbolo da ação (ex: PETR4.SA)
    
    Query Parameters:
        limit (int): Número máximo de predições a retornar (default: 30)
        include_pending (bool): Incluir predições sem preço real (default: false)
    
    Response (200 OK):
    {
        "success": true,
        "data": {
            "ticker": "PETR4.SA",
            "predictions": [
                {
                    "id": "abc123",
                    "prediction_date": "2026-01-15",
                    "predicted_price": 38.20,
                    "actual_price": 38.45,
                    "error": 0.25,
                    "error_percent": 0.65,
                    "predicted_at": "2026-01-14T15:30:00Z"
                },
                ...
            ],
            "metrics": {
                "total": 45,
                "mae": 0.42,
                "mape": 1.1,
                "rmse": 0.53
            }
        }
    }
    
    Response (404):
    {
        "success": false,
        "error": "No predictions found for ticker"
    }
    
    Response (503):
    {
        "success": false,
        "error": "Firestore service unavailable"
    }
    """
    try:
        # Validar ticker
        is_valid, error_msg = validate_ticker(ticker)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": "Invalid ticker",
                "message": error_msg
            }), 400
        
        # Query parameters
        limit = request.args.get('limit', default=30, type=int)
        include_pending = request.args.get('include_pending', default='false').lower() == 'true'
        
        # Validar limit
        if limit < 1 or limit > 200:
            return jsonify({
                "success": False,
                "error": "Invalid limit",
                "message": "Limit must be between 1 and 200"
            }), 400
        
        # Inicializar Firestore
        firestore_svc = FirestoreService()
        
        if not firestore_svc.is_available():
            logger.error("Firestore service unavailable")
            return jsonify({
                "success": False,
                "error": "Firestore service unavailable",
                "message": "Unable to retrieve prediction history at this time"
            }), 503
        
        # Buscar predições
        all_predictions = firestore_svc.get_predictions(ticker, limit=limit)
        
        if not all_predictions:
            return jsonify({
                "success": False,
                "error": "No predictions found",
                "message": f"No prediction history found for {ticker}"
            }), 404
        
        # Filtrar predições com preço real (se solicitado)
        if not include_pending:
            predictions = [p for p in all_predictions if p.get('actual_price') is not None]
        else:
            predictions = all_predictions
        
        # Buscar métricas de acurácia
        metrics = firestore_svc.get_accuracy_metrics(ticker, limit=100)
        
        # Formatar resposta
        return jsonify({
            "success": True,
            "data": {
                "ticker": ticker,
                "total_predictions": len(all_predictions),
                "predictions_with_actual": len([p for p in all_predictions if p.get('actual_price')]),
                "predictions": predictions[:limit],  # Limitar resultado
                "metrics": metrics
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_analytics for {ticker}: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred while retrieving analytics"
        }), 500


@analytics_bp.route('/analytics/<ticker>/pending', methods=['GET'])
def get_pending_predictions(ticker: str):
    """
    Retorna predições pendentes (sem preço real) para um ticker.
    
    Útil para saber quais predições ainda precisam ser atualizadas.
    
    Args:
        ticker: Símbolo da ação
    
    Response (200 OK):
    {
        "success": true,
        "data": {
            "ticker": "PETR4.SA",
            "pending_count": 3,
            "predictions": [
                {
                    "id": "xyz789",
                    "prediction_date": "2026-01-20",
                    "predicted_price": 38.90,
                    "current_price": 38.45,
                    "predicted_at": "2026-01-19T15:30:00Z"
                },
                ...
            ]
        }
    }
    """
    try:
        # Validar ticker
        is_valid, error_msg = validate_ticker(ticker)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": "Invalid ticker",
                "message": error_msg
            }), 400
        
        # Inicializar Firestore
        firestore_svc = FirestoreService()
        
        if not firestore_svc.is_available():
            return jsonify({
                "success": False,
                "error": "Firestore service unavailable"
            }), 503
        
        # Buscar predições pendentes
        pending = firestore_svc.get_pending_predictions(ticker)
        
        return jsonify({
            "success": True,
            "data": {
                "ticker": ticker,
                "pending_count": len(pending),
                "predictions": pending
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_pending_predictions for {ticker}: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500


@analytics_bp.route('/analytics/<ticker>/accuracy', methods=['GET'])
def get_accuracy(ticker: str):
    """
    Retorna apenas métricas de acurácia para um ticker.
    
    Args:
        ticker: Símbolo da ação
    
    Query Parameters:
        limit (int): Número de predições recentes a considerar (default: 100)
    
    Response (200 OK):
    {
        "success": true,
        "data": {
            "ticker": "PETR4.SA",
            "total": 45,
            "mae": 0.42,
            "mape": 1.1,
            "rmse": 0.53
        }
    }
    """
    try:
        # Validar ticker
        is_valid, error_msg = validate_ticker(ticker)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": "Invalid ticker",
                "message": error_msg
            }), 400
        
        # Query parameters
        limit = request.args.get('limit', default=100, type=int)
        
        if limit < 1 or limit > 500:
            return jsonify({
                "success": False,
                "error": "Invalid limit",
                "message": "Limit must be between 1 and 500"
            }), 400
        
        # Inicializar Firestore
        firestore_svc = FirestoreService()
        
        if not firestore_svc.is_available():
            return jsonify({
                "success": False,
                "error": "Firestore service unavailable"
            }), 503
        
        # Calcular métricas
        metrics = firestore_svc.get_accuracy_metrics(ticker, limit=limit)
        
        if metrics['total'] == 0:
            return jsonify({
                "success": False,
                "error": "No data available",
                "message": f"No completed predictions found for {ticker}"
            }), 404
        
        return jsonify({
            "success": True,
            "data": metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Error in get_accuracy for {ticker}: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
