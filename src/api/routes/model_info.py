from flask import Blueprint, jsonify, current_app
import json
from pathlib import Path

model_info_bp = Blueprint('model_info', __name__)


@model_info_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """Retorna configuração e métricas do modelo."""
    try:
        config_path = Path(__file__).parent.parent.parent.parent / 'artifacts' / 'model_config.json'
        
        if not config_path.exists():
            return jsonify({
                "error": "Config Not Found",
                "message": "Arquivo de configuração do modelo não encontrado",
                "status": 404
            }), 404
        
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
