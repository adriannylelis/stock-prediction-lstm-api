import logging

from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from src.api.config.rate_limit import RATE_LIMIT_ENABLED, RATE_LIMIT_STORAGE

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=RATE_LIMIT_STORAGE,
    default_limits=["50 per minute"] if RATE_LIMIT_ENABLED else [],
    enabled=RATE_LIMIT_ENABLED,
    headers_enabled=True,
    swallow_errors=True,
)


def create_app(config=None):
    app = Flask(__name__)

    app.config.update(
        {
            "JSON_SORT_KEYS": False,
            "JSONIFY_PRETTYPRINT_REGULAR": True,
            "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,
        }
    )

    if config:
        app.config.update(config)

    CORS(
        app,
        resources={
            r"/*": {
                "origins": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"],
            }
        },
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    limiter.init_app(app)
    app.logger.info(
        f"Rate limiting {'habilitado' if RATE_LIMIT_ENABLED else 'desabilitado'}"
    )

    register_blueprints(app)
    register_error_handlers(app)

    app.logger.info("API Flask inicializada com sucesso")

    return app


def register_blueprints(app):
    from src.api.routes.analytics import analytics_bp
    from src.api.routes.health import health_bp
    from src.api.routes.model_info import model_info_bp
    from src.api.routes.prediction import prediction_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(model_info_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(analytics_bp)

    app.logger.info(
        "Blueprints registrados: health, model_info, prediction, analytics"
    )


def register_error_handlers(app):
    from flask_limiter.errors import RateLimitExceeded

    from src.api.utils.exceptions import (APIException, InsufficientDataError,
                                          InvalidTickerError,
                                          ModelInferenceError,
                                          ServiceUnavailableError,
                                          TickerNotFoundError)

    @app.errorhandler(RateLimitExceeded)
    def handle_rate_limit_exceeded(error):
        app.logger.warning(f"Rate limit excedido: {error.description}")
        return (
            jsonify(
                {
                    "error": "RateLimitExceeded",
                    "message": "Limite de requisições excedido. Tente novamente em alguns instantes.",
                    "status": 429,
                    "retry_after": error.description,
                }
            ),
            429,
        )

    @app.errorhandler(APIException)
    def handle_api_exception(error):
        app.logger.warning(f"API Exception: {error.__class__.__name__} - {str(error)}")
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(InvalidTickerError)
    def handle_invalid_ticker(error):
        app.logger.warning(f"Ticker inválido: {str(error)}")
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(TickerNotFoundError)
    def handle_ticker_not_found(error):
        app.logger.warning(f"Ticker não encontrado: {str(error)}")
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(InsufficientDataError)
    def handle_insufficient_data(error):
        app.logger.warning(f"Dados insuficientes: {str(error)}")
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(ModelInferenceError)
    def handle_model_inference_error(error):
        app.logger.error(f"Erro de inferência: {str(error)}")
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(ServiceUnavailableError)
    def handle_service_unavailable(error):
        app.logger.error(f"Serviço indisponível: {str(error)}")
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(404)
    def not_found(error):
        return (
            jsonify(
                {
                    "error": "Not Found",
                    "message": "O endpoint solicitado não existe",
                    "status": 404,
                }
            ),
            404,
        )

    @app.errorhandler(405)
    def method_not_allowed(error):
        return (
            jsonify(
                {
                    "error": "Method Not Allowed",
                    "message": "Método HTTP não permitido para este endpoint",
                    "status": 405,
                }
            ),
            405,
        )

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Erro interno: {str(error)}")
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "Erro interno do servidor",
                    "status": 500,
                }
            ),
            500,
        )

    @app.errorhandler(Exception)
    def handle_exception(error):
        app.logger.error(f"Exceção não tratada: {str(error)}", exc_info=True)
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "Ocorreu um erro inesperado",
                    "status": 500,
                }
            ),
            500,
        )


if __name__ == "__main__":
    import os

    port = int(os.environ.get("FLASK_PORT", os.environ.get("PORT", 5001)))

    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)
