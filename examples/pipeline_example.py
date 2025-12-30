"""Example: Using TrainPipeline and PredictPipeline.

This script demonstrates the complete workflow using pipeline classes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.ml.pipeline.train_pipeline import TrainPipeline
from src.ml.pipeline.predict_pipeline import PredictPipeline


def main():
    """Run complete pipeline workflow."""
    
    # =================================================================
    # PART 1: TRAINING PIPELINE
    # =================================================================
    logger.info("="*60)
    logger.info("PART 1: TRAINING PIPELINE")
    logger.info("="*60)
    
    # Create training pipeline
    train_pipeline = TrainPipeline(
        ticker="PETR4.SA",
        start_date="2020-01-01",
        end_date="2024-01-01",
        lookback=60,
        hidden_size=50,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        experiment_name="pipeline-example",
        model_save_path="artifacts/models/pipeline_model.pt",
        seed=42
    )
    
    # Run training
    train_results = train_pipeline.run()
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("📊 TRAINING RESULTS")
    logger.info("="*60)
    logger.info(f"Model: {train_results['model_path']}")
    logger.info(f"Epochs trained: {train_results['metadata']['epochs_trained']}")
    logger.info("\nTest Metrics:")
    for metric, value in train_results['test_metrics'].items():
        logger.info(f"  {metric.upper()}: {value:.6f}")
    logger.info("="*60)
    
    # =================================================================
    # PART 2: PREDICTION PIPELINE
    # =================================================================
    logger.info("\n" + "="*60)
    logger.info("PART 2: PREDICTION PIPELINE")
    logger.info("="*60)
    
    # Create prediction pipeline
    predict_pipeline = PredictPipeline(
        model_path=train_results['model_path'],
        ticker="PETR4.SA",
        lookback=60
    )
    
    # Generate predictions
    predictions_df = predict_pipeline.predict(
        days_ahead=5,
        output_path="data/predictions/pipeline_predictions.csv"
    )
    
    # Display predictions
    logger.info("\n" + "="*60)
    logger.info("🔮 PREDICTIONS")
    logger.info("="*60)
    logger.info(f"\n{predictions_df.to_string(index=False)}")
    logger.info("="*60)
    
    logger.success("\n✅ Pipeline workflow complete!")


if __name__ == "__main__":
    main()
