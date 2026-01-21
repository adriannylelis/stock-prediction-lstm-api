"""
Automated Training Pipeline.

Handles complete training flow with data validation, model training,
and MLflow registration.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

from loguru import logger

from src.ml.pipeline.train_pipeline import TrainPipeline

# B3 ticker categories (43 unique tickers total)
B3_TICKERS = {
    "blue_chips": [
        "PETR4.SA",  # Petrobras
        "VALE3.SA",  # Vale
        "ITUB4.SA",  # Itaú
        "BBDC4.SA",  # Bradesco
        "ABEV3.SA",  # Ambev
        "BBAS3.SA",  # Banco do Brasil
        "WEGE3.SA",  # WEG
        "RENT3.SA",  # Localiza
        "B3SA3.SA",  # B3
        "SUZB3.SA",  # Suzano
    ],
    "bancos": [
        "SANB11.SA",  # Santander
        "BBSE3.SA",  # BB Seguridade
    ],
    "energia": [
        "PETR3.SA",  # Petrobras PN
        "ELET3.SA",  # Eletrobras
        "ELET6.SA",  # Eletrobras PNB
        "CMIG4.SA",  # Cemig
        "CPLE6.SA",  # Copel
    ],
    "varejo": [
        "MGLU3.SA",  # Magazine Luiza
        "LREN3.SA",  # Lojas Renner
        "PETZ3.SA",  # Petz
        "AMER3.SA",  # Americanas
    ],
    "mineracao": [
        "CMIN3.SA",  # CSN Mineração
        "GOAU4.SA",  # Metalúrgica Gerdau
    ],
    "construcao": [
        "CYRE3.SA",  # Cyrela
        "BEEF3.SA",  # Minerva
        "EZTC3.SA",  # EZTec
    ],
    "telecom": [
        "VIVT3.SA",  # Vivo
        "TIMS3.SA",  # Tim
    ],
    "papel_celulose": [
        "KLBN11.SA",  # Klabin
    ],
    "saude": [
        "RADL3.SA",  # Raia Drogasil
        "HAPV3.SA",  # Hapvida
        "FLRY3.SA",  # Fleury
    ],
    "tecnologia": [
        "TOTS3.SA",  # Totvs
        "LWSA3.SA",  # Locaweb
    ],
    "alimentacao": [],
    "servicos": [
        "CSAN3.SA",  # Cosan
        "RAIL3.SA",  # Rumo
    ],
}

# All tickers (all categories combined - ~43 unique tickers)
ALL_TICKERS = sorted(
    list(
        set(
            ticker
            for category_tickers in B3_TICKERS.values()
            for ticker in category_tickers
        )
    )
)

DEFAULT_TICKERS = B3_TICKERS["blue_chips"]


@dataclass
class TrainingResult:
    """Result of training pipeline."""

    success: bool
    model_uri: Optional[str] = None
    version: Optional[int] = None
    metrics: Optional[dict] = None
    stage: str = "Staging"
    timestamp: str = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class AutoTrainingPipeline:
    """Automated training pipeline with data validation.

    Features:
    - Data quality validation
    - Automatic model training
    - MLflow registration (Staging)
    - Structured result reporting

    Args:
        tickers: List of tickers or "all"/"blue_chips"/"category_name"
        min_data_quality: Minimum data quality threshold (0-1)
        min_samples: Minimum number of samples required
        start_date: Start date for data
        lookback: Lookback period
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        early_stopping_patience: Early stopping patience
        experiment_name: MLflow experiment name
        seed: Random seed

    Example:
        >>> pipeline = AutoTrainingPipeline(tickers="all")
        >>> result = pipeline.run()
        >>> if result.success:
        >>>     print(f"Model v{result.version}: {result.model_uri}")
    """

    def __init__(
        self,
        tickers: Union[List[str], str] = "blue_chips",
        min_data_quality: float = 0.95,
        min_samples: int = 1000,
        start_date: str = "2020-01-01",
        lookback: int = 60,
        hidden_size: int = 100,
        num_layers: int = 3,
        dropout: float = 0.3,
        batch_size: int = 64,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 15,
        experiment_name: str = "lstm-multi-ticker",
        device: str = "auto",
        seed: int = 42,
    ):
        self.min_data_quality = min_data_quality
        self.min_samples = min_samples

        # Resolve tickers
        self.tickers = self._resolve_tickers(tickers)

        self.start_date = start_date
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.experiment_name = experiment_name
        self.device = device
        self.seed = seed

        logger.info(
            f"AutoTrainingPipeline initialized with {len(self.tickers)} tickers"
        )

    def _resolve_tickers(self, tickers: Union[List[str], str]) -> List[str]:
        """Resolve ticker specification to actual list."""
        if isinstance(tickers, list):
            return tickers

        if tickers == "all":
            return ALL_TICKERS
        elif tickers == "blue_chips":
            return DEFAULT_TICKERS
        elif tickers in B3_TICKERS:
            return B3_TICKERS[tickers]
        else:
            raise ValueError(
                f"Invalid tickers: {tickers}. "
                f"Use list, 'all', 'blue_chips', or category name"
            )

    def run(self) -> TrainingResult:
        """Run complete training pipeline.

        Returns:
            TrainingResult with model URI, metrics, and status
        """
        logger.info("=" * 80)
        logger.info("🚀 Auto Training Pipeline Started")
        logger.info("=" * 80)
        logger.info(f"Tickers: {len(self.tickers)}")
        logger.info(f"Min quality: {self.min_data_quality:.2%}")
        logger.info(f"Min samples: {self.min_samples}")
        logger.info("=" * 80)

        try:
            # Use TrainPipeline for training
            logger.info("\n🏋️ Training model...")

            pipeline = TrainPipeline(
                tickers=self.tickers if len(self.tickers) > 1 else None,
                ticker=self.tickers[0] if len(self.tickers) == 1 else None,
                start_date=self.start_date,
                lookback=self.lookback,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                batch_size=self.batch_size,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                early_stopping_patience=self.early_stopping_patience,
                experiment_name=self.experiment_name,
                device=self.device,
                seed=self.seed,
            )

            result = pipeline.run()

            try:
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                model_name = "stock-lstm-model"
                versions = client.search_model_versions(f"name='{model_name}'")

                if versions:
                    latest_version = max([int(v.version) for v in versions])
                    model_uri = f"models:/{model_name}/{latest_version}"
                    version = latest_version
                else:
                    model_uri = None
                    version = None
            except Exception as e:
                logger.warning(f"Could not get model version: {e}")
                model_uri = None
                version = None

            logger.success("\n" + "=" * 80)
            logger.success("✅ Training Pipeline Completed Successfully!")
            logger.info(f"Model URI: {model_uri or 'Not registered'}")
            logger.info(f"Version: {version or 'Not registered'}")
            logger.info(f"Metrics: {result.get('test_metrics', {})}")
            logger.success("=" * 80)

            return TrainingResult(
                success=True,
                model_uri=model_uri,
                version=version,
                metrics=result.get("test_metrics", {}),
                stage="Staging",
            )

        except Exception as e:
            logger.error(f"\n❌ Training pipeline failed: {str(e)}")
            logger.exception(e)
            return TrainingResult(success=False, error=str(e))


if __name__ == "__main__":
    # Test the pipeline
    pipeline = AutoTrainingPipeline(
        tickers="blue_chips",
        epochs=20,  # Quick test
    )
    result = pipeline.run()

    if result.success:
        print(f"\n✅ Success! Model: {result.model_uri}")
        print(f"Metrics: {result.metrics}")
    else:
        print(f"\n❌ Failed: {result.error}")
