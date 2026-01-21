from datetime import datetime

from flask import Blueprint, jsonify

from src.api.main import limiter

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
@limiter.limit("100 per minute")
def health_check():
    return (
        jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "stock-prediction-lstm-api",
            }
        ),
        200,
    )
