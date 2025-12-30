"""
Rota de health check da API.

Endpoint para verificar se a API está rodando e saudável.
"""

from flask import Blueprint, jsonify
from datetime import datetime

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Retorna:
        JSON com status da API e timestamp.
        
    Exemplo de resposta:
        {
            "status": "healthy",
            "timestamp": "2025-12-29T10:30:00",
            "service": "stock-prediction-lstm-api"
        }
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "stock-prediction-lstm-api"
    }), 200
