"""
Aplicação Flask para servir modelo LSTM via API REST.

Este módulo implementa o Application Factory Pattern, permitindo
criar múltiplas instâncias da aplicação para diferentes ambientes.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import logging
from pathlib import Path


def create_app(config=None):
    """
    Factory para criar e configurar a aplicação Flask.
    
    Args:
        config (dict, optional): Configurações customizadas para a app.
    
    Returns:
        Flask: Instância configurada da aplicação.
    """
    app = Flask(__name__)
    
    # Configurações padrão
    app.config.update({
        'JSON_SORT_KEYS': False,
        'JSONIFY_PRETTYPRINT_REGULAR': True,
        'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max
    })
    
    # Aplicar configurações customizadas
    if config:
        app.config.update(config)
    
    # Configurar CORS
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Registrar blueprints
    register_blueprints(app)
    
    # Registrar error handlers
    register_error_handlers(app)
    
    # Log de inicialização
    app.logger.info("API Flask inicializada com sucesso")
    
    return app


def register_blueprints(app):
    """
    Registra todos os blueprints (rotas) na aplicação.
    
    Args:
        app (Flask): Instância da aplicação Flask.
    """
    from src.api.routes.health import health_bp
    from src.api.routes.model_info import model_info_bp
    from src.api.routes.prediction import prediction_bp
    
    app.register_blueprint(health_bp)
    app.register_blueprint(model_info_bp)
    app.register_blueprint(prediction_bp)
    
    app.logger.info("Blueprints registrados: health, model_info, prediction")


def register_error_handlers(app):
    """
    Registra handlers globais de erro.
    
    Args:
        app (Flask): Instância da aplicação Flask.
    """
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "O endpoint solicitado não existe",
            "status": 404
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "error": "Method Not Allowed",
            "message": "Método HTTP não permitido para este endpoint",
            "status": 405
        }), 405
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Erro interno: {str(error)}")
        return jsonify({
            "error": "Internal Server Error",
            "message": "Erro interno do servidor",
            "status": 500
        }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        app.logger.error(f"Exceção não tratada: {str(error)}", exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": "Ocorreu um erro inesperado",
            "status": 500
        }), 500


if __name__ == '__main__':
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
