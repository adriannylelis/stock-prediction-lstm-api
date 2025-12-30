"""
Rota para informações do modelo.

Endpoint para obter metadados e configurações do modelo LSTM.
"""

from flask import Blueprint, jsonify, current_app
import json
from pathlib import Path

model_info_bp = Blueprint('model_info', __name__)


@model_info_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """
    Retorna informações sobre o modelo carregado.
    
    Lê o arquivo model_config.json e retorna seus dados.
    
    Retorna:
        JSON com configuração e métricas do modelo.
        
    Exemplo de resposta:
        {
            "model_type": "LSTM",
            "architecture": "LSTM-1x16",
            "input_features": 1,
            "sequence_length": 60,
            "metrics": {
                "mape": 1.21,
                "r2": 0.90
            }
        }
    """
    try:
        # Caminho para o arquivo de configuração
        config_path = Path(__file__).parent.parent.parent.parent / 'artifacts' / 'model_config.json'
        
        # Verificar se arquivo existe
        if not config_path.exists():
            return jsonify({
                "error": "Config Not Found",
                "message": "Arquivo de configuração do modelo não encontrado",
                "status": 404
            }), 404
        
        # Ler configuração
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return jsonify(config), 200
        
    except json.JSONDecodeError as e:
        current_app.logger.error(f"Erro ao decodificar JSON: {str(e)}")
        return jsonify({
            "error": "Invalid Config",
            "message": "Arquivo de configuração inválido",
            "status": 500
        }), 500
        
    except Exception as e:
        current_app.logger.error(f"Erro ao buscar info do modelo: {str(e)}")
        return jsonify({
            "error": "Internal Server Error",
            "message": "Erro ao buscar informações do modelo",
            "status": 500
        }), 500
