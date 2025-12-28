"""Data versioning and persistence utilities.

Handles saving/loading of datasets with versioning support.
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient


class DataVersionManager:
    """Manage dataset versions with metadata tracking."""

    def __init__(
        self,
        base_path: str = "data/versioned",
        auto_cleanup: bool = False,
        max_versions: int = 10,
        compress_after_days: int = 30,
    ):
        """Initialize data version manager."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.auto_cleanup = auto_cleanup
        self.max_versions = max_versions
        self.compress_after_days = compress_after_days

        logger.info(f"Initialized DataVersionManager: {self.base_path}")
        if auto_cleanup:
            logger.info(f"  Auto-cleanup: ON (max {max_versions} versions)")
            logger.info(f"  Compression: After {compress_after_days} days")

    def save(self, df: pd.DataFrame, ticker: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save DataFrame with version and metadata."""
        import time

        # Generate unique version with milliseconds
        timestamp = datetime.now()
        millis = timestamp.microsecond // 1000  # Convert to milliseconds
        version = timestamp.strftime("%Y%m%d_%H%M%S") + f"_{millis:03d}"
        ticker_clean = ticker.replace(".SA", "").replace(".", "_")

        version_dir = self.base_path / ticker_clean / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Add small sleep to ensure unique timestamps in rapid saves
        time.sleep(0.01)

        data_path = version_dir / "data.parquet"
        df.to_parquet(data_path, index=True)

        full_metadata = {
            "version": version,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "n_records": len(df),
            "n_features": df.shape[1],
            "date_range": {"start": str(df.index.min()), "end": str(df.index.max())},
            "features": list(df.columns),
            "data_path": str(data_path),
            "format": "parquet",
        }

        if metadata:
            full_metadata["custom"] = metadata

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        logger.success(f"💾 Saved dataset version: {version}")

        if self.auto_cleanup:
            self._cleanup_old_versions(ticker)

        self._compress_old_datasets(ticker)

        return version

    def load(self, version: str, ticker: str) -> pd.DataFrame:
        """Load specific dataset version."""
        ticker_clean = ticker.replace(".SA", "").replace(".", "_")
        version_dir = self.base_path / ticker_clean / version

        if not version_dir.exists():
            raise FileNotFoundError(f"Version not found: {version}")

        data_path = version_dir / "data.parquet"
        compressed_path = version_dir / "data.parquet.gz"

        if compressed_path.exists():
            df = pd.read_parquet(compressed_path)
        else:
            df = pd.read_parquet(data_path)

        # Restore index if it was a DatetimeIndex
        if "Date" in df.columns or "date" in df.columns:
            date_col = "Date" if "Date" in df.columns else "date"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        elif df.columns[0] in ["index", "Unnamed: 0"]:  # Common index column names
            df = df.set_index(df.columns[0])

        logger.info(f"📂 Loaded dataset version: {version}")
        return df

    def get_metadata(self, version: str, ticker: str) -> Dict[str, Any]:
        """Get metadata for specific version."""
        ticker_clean = ticker.replace(".SA", "").replace(".", "_")
        metadata_path = self.base_path / ticker_clean / version / "metadata.json"

        with open(metadata_path) as f:
            return json.load(f)

    def list_versions(self, ticker: str) -> list:
        """List all available versions for a ticker."""
        ticker_clean = ticker.replace(".SA", "").replace(".", "_")
        ticker_dir = self.base_path / ticker_clean

        if not ticker_dir.exists():
            return []

        versions = []
        for version_dir in sorted(ticker_dir.iterdir(), reverse=True):
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        versions.append(json.load(f))

        return versions

    def get_latest_version(self, ticker: str) -> Optional[str]:
        """Get latest version identifier."""
        versions = self.list_versions(ticker)
        if versions:
            return versions[0]["version"]
        return None

    def load_latest(self, ticker: str) -> pd.DataFrame:
        """Load latest dataset version."""
        latest = self.get_latest_version(ticker)
        if latest is None:
            raise FileNotFoundError(f"No versions found for {ticker}")
        return self.load(latest, ticker)

    def _cleanup_old_versions(self, ticker: str) -> None:
        """Remove old versions."""
        versions = self.list_versions(ticker)

        if len(versions) > self.max_versions:
            versions_sorted = sorted(versions, key=lambda x: x["timestamp"], reverse=True)
            to_remove = versions_sorted[self.max_versions :]
            ticker_clean = ticker.replace(".SA", "").replace(".", "_")

            for version_meta in to_remove:
                version_dir = self.base_path / ticker_clean / version_meta["version"]
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                    logger.info(f"🗑️  Removed old version: {version_meta['version']}")

    def _compress_old_datasets(self, ticker: str) -> None:
        """Compress datasets older than threshold."""
        versions = self.list_versions(ticker)
        ticker_clean = ticker.replace(".SA", "").replace(".", "_")
        cutoff_date = datetime.now() - timedelta(days=self.compress_after_days)

        for version_meta in versions:
            version_timestamp = datetime.fromisoformat(version_meta["timestamp"])

            if version_timestamp < cutoff_date:
                version_dir = self.base_path / ticker_clean / version_meta["version"]
                data_path = version_dir / "data.parquet"
                compressed_path = version_dir / "data.parquet.gz"

                if data_path.exists() and not compressed_path.exists():
                    df = pd.read_parquet(data_path)
                    df.to_parquet(compressed_path, compression="gzip", index=True)
                    data_path.unlink()

                    metadata_path = version_dir / "metadata.json"
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    metadata["compressed"] = True
                    metadata["compression_date"] = datetime.now().isoformat()
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"📦 Compressed: {version_meta['version']}")

    def cleanup_all(self, ticker: str, keep_latest: int = 1) -> None:
        """Manually cleanup all versions except N latest."""
        versions = self.list_versions(ticker)
        ticker_clean = ticker.replace(".SA", "").replace(".", "_")

        if len(versions) > keep_latest:
            versions_sorted = sorted(versions, key=lambda x: x["timestamp"], reverse=True)
            to_remove = versions_sorted[keep_latest:]

            for version_meta in to_remove:
                version_dir = self.base_path / ticker_clean / version_meta["version"]
                if version_dir.exists():
                    shutil.rmtree(version_dir)


class ArtifactManager:
    """Manage ML artifacts (models, scalers, configs)."""

    def __init__(
        self,
        base_path: str = "artifacts",
        use_mlflow: bool = True,
        mlflow_tracking_uri: str = "file:./mlruns",
    ):
        """Initialize artifact manager."""
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.scalers_path = self.base_path / "scalers"
        self.configs_path = self.base_path / "configs"

        for path in [self.models_path, self.scalers_path, self.configs_path]:
            path.mkdir(parents=True, exist_ok=True)

        self.use_mlflow = use_mlflow
        if use_mlflow:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.mlflow_client = MlflowClient()
            logger.info("MLflow Model Registry: ENABLED")

        logger.info(f"Initialized ArtifactManager: {self.base_path}")

    def save_scaler(self, scaler: Any, name: str, metadata: Optional[Dict] = None) -> Path:
        """Save fitted scaler."""
        import joblib

        scaler_path = self.scalers_path / f"{name}.pkl"
        joblib.dump(scaler, scaler_path)

        if metadata:
            meta_path = self.scalers_path / f"{name}_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.success(f"💾 Saved scaler: {scaler_path}")
        return scaler_path

    def load_scaler(self, name: str) -> Any:
        """Load fitted scaler."""
        import joblib

        scaler_path = self.scalers_path / f"{name}.pkl"
        scaler = joblib.load(scaler_path)
        logger.info(f"📂 Loaded scaler: {name}")
        return scaler

    def save_config(self, config: Dict[str, Any], name: str) -> Path:
        """Save configuration."""
        config_path = self.configs_path / f"{name}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.success(f"💾 Saved config: {config_path}")
        return config_path

    def load_config(self, name: str) -> Dict[str, Any]:
        """Load configuration."""
        config_path = self.configs_path / f"{name}.json"
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"📂 Loaded config: {name}")
        return config

    def register_model_mlflow(
        self,
        model_name: str,
        run_id: str,
        artifact_path: str = "model",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register model in MLflow Model Registry."""
        if not self.use_mlflow:
            logger.warning("MLflow not enabled")
            return "N/A"

        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_version = mlflow.register_model(model_uri, model_name)

        if tags:
            for key, value in tags.items():
                self.mlflow_client.set_model_version_tag(
                    model_name, model_version.version, key, value
                )

        logger.success(f"📝 Registered model '{model_name}' v{model_version.version}")
        return model_version.version

    def get_latest_model_version(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """Get latest model version from MLflow Registry."""
        if not self.use_mlflow:
            return None

        try:
            versions = self.mlflow_client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
        except Exception as e:
            logger.warning(f"Could not get latest version: {e}")

        return None

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
        """Transition model to different stage."""
        if not self.use_mlflow:
            return

        self.mlflow_client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )
        logger.success(f"✓ Transitioned '{model_name}' v{version} to {stage}")
