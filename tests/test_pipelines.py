"""Tests for pipeline classes."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.pipeline.predict_pipeline import PredictPipeline
from src.ml.pipeline.train_pipeline import TrainPipeline


def test_train_pipeline_initialization():
    """Test TrainPipeline initialization."""
    pipeline = TrainPipeline(
        ticker="PETR4.SA",
        epochs=5,  # Small for testing
        seed=42,
    )

    assert pipeline.ticker == "PETR4.SA"
    assert pipeline.epochs == 5
    assert pipeline.seed == 42


def test_train_pipeline_run():
    """Test TrainPipeline.run() with minimal data."""
    pipeline = TrainPipeline(
        ticker="PETR4.SA",
        start_date="2023-01-01",
        end_date="2023-12-31",
        epochs=5,  # Very small for speed
        batch_size=16,
        lookback=30,
        seed=42,
    )

    results = pipeline.run()

    # Check results structure
    assert "model_path" in results
    assert "training_history" in results
    assert "test_metrics" in results
    assert "metadata" in results

    # Check test metrics (keys are in uppercase)
    assert "MAE" in results["test_metrics"]
    assert "RMSE" in results["test_metrics"]
    assert "MAPE" in results["test_metrics"]

    # Check metadata
    assert results["metadata"]["ticker"] == "PETR4.SA"
    assert results["metadata"]["epochs_trained"] <= 5


@pytest.mark.slow
def test_predict_pipeline_initialization(tmp_path):
    """Test PredictPipeline initialization."""
    # First create a model
    model_path = tmp_path / "best_model.pt"
    train_pipeline = TrainPipeline(
        ticker="PETR4.SA",
        start_date="2023-01-01",
        end_date="2023-06-01",
        lookback=60,
        epochs=2,
        model_save_path=str(model_path),
        experiment_name=None,
    )
    results = train_pipeline.run()

    # Now test predict pipeline with correct parameters
    predict_pipeline = PredictPipeline(
        model_path=results["model_path"], ticker="PETR4.SA", lookback=60
    )

    assert predict_pipeline.ticker == "PETR4.SA"
    assert predict_pipeline.lookback == 60
    assert predict_pipeline.model is not None


@pytest.mark.slow
def test_predict_pipeline_predict(tmp_path):
    """Test PredictPipeline.predict()."""
    # First create a model
    model_path = tmp_path / "best_model.pt"
    train_pipeline = TrainPipeline(
        ticker="PETR4.SA",
        start_date="2023-01-01",
        end_date="2023-06-01",
        lookback=60,
        epochs=2,
        model_save_path=str(model_path),
        experiment_name=None,
    )
    results = train_pipeline.run()

    # Use predict pipeline with correct parameters
    predict_pipeline = PredictPipeline(
        model_path=results["model_path"], ticker="PETR4.SA", lookback=60
    )

    predictions = predict_pipeline.predict(days_ahead=3)

    # Check DataFrame structure
    assert predictions is not None
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
