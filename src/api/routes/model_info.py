import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify

from src.api.main import limiter

PROD_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "configs" / "production_model.yaml"
)
PREP_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "artifacts"
    / "models"
    / "scalers"
    / "preprocessing_config.json"
)
CHECKPOINT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "artifacts"
    / "models"
    / "best_model.pt"
)

model_info_bp = Blueprint("model_info", __name__)


@model_info_bp.route("/model/info", methods=["GET"])
@limiter.limit("30 per minute")
def get_model_info():
    """Retorna configuração e métricas do modelo."""
    try:
        response = {
            "architecture": "StockLSTM",
            "lookback": 60,
            "features": [],
            "metrics": {"val_loss": "unknown"},
            "model_uri": "unknown",
            "stage": "unknown",
        }

        # Enriquecer com preprocessing_config.json se existir
        if PREP_CONFIG_PATH.exists():
            with open(PREP_CONFIG_PATH, "r") as f:
                prep_cfg = json.load(f)
            response["lookback"] = prep_cfg.get("lookback", response["lookback"])
            response["features"] = prep_cfg.get("feature_cols", response["features"])
            response["num_features"] = prep_cfg.get("num_features")
            response["num_tickers"] = prep_cfg.get("num_tickers")
            response["tickers"] = prep_cfg.get("ticker_list")

        # Enriquecer com production_model.yaml se existir
        if PROD_CONFIG_PATH.exists():
            import yaml

            with open(PROD_CONFIG_PATH, "r") as f:
                prod_cfg = yaml.safe_load(f) or {}
            response["model_uri"] = prod_cfg.get("model_uri", response["model_uri"])
            response["stage"] = prod_cfg.get("stage", response["stage"])
            response["deployed_at"] = prod_cfg.get("deployed_at")

        # Melhor esforço para ler métricas do checkpoint
        if CHECKPOINT_PATH.exists():
            import torch

            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            if "best_val_loss" in ckpt:
                response["metrics"]["val_loss"] = ckpt["best_val_loss"]
            response["num_features"] = ckpt.get(
                "num_features", response.get("num_features")
            )
            response["num_tickers"] = ckpt.get(
                "num_tickers", response.get("num_tickers")
            )

        return jsonify(response), 200

    except json.JSONDecodeError as e:
        current_app.logger.error(f"Erro ao decodificar JSON: {str(e)}")
        return (
            jsonify(
                {
                    "error": "Invalid Config",
                    "message": "Arquivo de configuração inválido",
                    "status": 500,
                }
            ),
            500,
        )

    except Exception as e:
        current_app.logger.error(f"Erro ao buscar info do modelo: {str(e)}")
        return (
            jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "Erro ao buscar informações do modelo",
                    "status": 500,
                }
            ),
            500,
        )
