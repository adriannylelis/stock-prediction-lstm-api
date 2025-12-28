"""MLflow experiment tracking integration.

This module provides integration with MLflow for tracking experiments,
logging parameters, metrics, models, and artifacts.
"""

from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch
from loguru import logger


class ExperimentTracker:
    """MLflow experiment tracker for logging training runs.

    This class provides a clean interface to MLflow tracking functionality,
    making it easy to log parameters, metrics, models, and artifacts.

    Attributes:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the current run.
        run: Active MLflow run object.

    Example:
        >>> tracker = ExperimentTracker(experiment_name="stock-lstm")
        >>> tracker.start_run(run_name="baseline-v1")
        >>> tracker.log_params({"lr": 0.001, "hidden_size": 50})
        >>> for epoch in range(100):
        ...     tracker.log_metrics({"train_loss": 0.5}, step=epoch)
        >>> tracker.log_model(model, "model")
        >>> tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "file:./mlruns",
        artifact_location: Optional[str] = None,
    ) -> None:
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: URI of the MLflow tracking server.
            artifact_location: Location to store artifacts. If None, uses default.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name, artifact_location=artifact_location
                )
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id
        except Exception as e:
            logger.error(f"Failed to set up MLflow experiment: {e}")
            raise

        self.run = None
        self.run_name = None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Start a new MLflow run.

        Args:
            run_name: Name for the run. If None, MLflow generates one.
            tags: Dictionary of tags to add to the run.
            description: Description of the run.
        """
        if self.run is not None:
            logger.warning("Run already active. Ending previous run.")
            self.end_run()

        self.run = mlflow.start_run(run_name=run_name)
        self.run_name = run_name

        # Set tags
        if tags:
            mlflow.set_tags(tags)

        # Set description
        if description:
            mlflow.set_tag("mlflow.note.content", description)

        logger.info(f"Started MLflow run: {self.run.info.run_id} (name: {run_name})")

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        try:
            mlflow.log_param(key, value)
            logger.debug(f"Logged param: {key}={value}")
        except Exception as e:
            logger.error(f"Failed to log param {key}: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters.

        Args:
            params: Dictionary of parameters to log.
        """
        try:
            mlflow.log_params(params)
            logger.info(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log params: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: Step number (e.g., epoch, batch).
        """
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metrics to log.
            step: Step number (e.g., epoch, batch).
        """
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_model(self, model: torch.nn.Module, artifact_path: str = "model", **kwargs) -> None:
        """Log a PyTorch model.

        Args:
            model: PyTorch model to log.
            artifact_path: Path within artifacts to save model.
            **kwargs: Additional arguments for mlflow.pytorch.log_model.
        """
        try:
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            logger.info(f"Logged model to {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a file as an artifact.

        Args:
            local_path: Path to local file.
            artifact_path: Path within artifacts to save file.
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact {local_path}: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log all files in a directory as artifacts.

        Args:
            local_dir: Path to local directory.
            artifact_path: Path within artifacts to save files.
        """
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"Logged artifacts from {local_dir}")
        except Exception as e:
            logger.error(f"Failed to log artifacts from {local_dir}: {e}")

    def log_figure(self, figure, artifact_file: str) -> None:
        """Log a matplotlib figure.

        Args:
            figure: Matplotlib figure object.
            artifact_file: Filename for the saved figure.
        """
        try:
            mlflow.log_figure(figure, artifact_file)
            logger.debug(f"Logged figure: {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log figure: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag for the current run.

        Args:
            key: Tag name.
            value: Tag value.
        """
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.error(f"Failed to set tag {key}: {e}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags.

        Args:
            tags: Dictionary of tags.
        """
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run.

        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED').
        """
        if self.run is None:
            logger.warning("No active run to end")
            return

        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.run.info.run_id} (status: {status})")
            self.run = None
            self.run_name = None
        except Exception as e:
            logger.error(f"Failed to end run: {e}")

    def get_run_id(self) -> Optional[str]:
        """Get the current run ID.

        Returns:
            Run ID if run is active, None otherwise.
        """
        if self.run:
            return self.run.info.run_id
        return None

    def register_model(
        self, model_uri: str, model_name: str, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Register a model in MLflow Model Registry.

        Args:
            model_uri: URI of the model (e.g., "runs:/<run_id>/model").
            model_name: Name for the registered model.
            tags: Optional tags for the model version.
        """
        try:
            result = mlflow.register_model(model_uri, model_name, tags=tags)
            logger.info(
                f"Registered model '{model_name}' (version {result.version}) from {model_uri}"
            )
        except Exception as e:
            logger.error(f"Failed to register model: {e}")

    @staticmethod
    def load_model(model_uri: str) -> torch.nn.Module:
        """Load a model from MLflow.

        Args:
            model_uri: URI of the model to load.

        Returns:
            Loaded PyTorch model.
        """
        try:
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_uri}: {e}")
            raise
