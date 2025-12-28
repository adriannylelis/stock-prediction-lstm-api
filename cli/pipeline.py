"""Pipeline command - Run complete pipeline."""

import click
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.pipeline.train_pipeline import TrainPipeline
from src.ml.pipeline.predict_pipeline import PredictPipeline
from src.ml.utils.seed import set_seed
from loguru import logger


@click.command()
@click.option(
    '--mode',
    type=click.Choice(['train', 'predict', 'both']),
    default='both',
    help='Pipeline mode: train, predict, or both'
)
@click.option(
    '--ticker',
    type=str,
    required=True,
    help='Stock ticker symbol (e.g., PETR4.SA)'
)
@click.option(
    '--start-date',
    type=str,
    default='2020-01-01',
    help='Start date for data (YYYY-MM-DD)'
)
@click.option(
    '--lookback',
    type=int,
    default=60,
    help='Lookback period for sequences'
)
@click.option(
    '--hidden-size',
    type=int,
    default=50,
    help='LSTM hidden size'
)
@click.option(
    '--num-layers',
    type=int,
    default=2,
    help='Number of LSTM layers'
)
@click.option(
    '--epochs',
    type=int,
    default=100,
    help='Maximum training epochs'
)
@click.option(
    '--model-path',
    type=str,
    default='artifacts/models/pipeline_model.pt',
    help='Path to model (save for train, load for predict)'
)
@click.option(
    '--days-ahead',
    type=int,
    default=5,
    help='Days to predict (predict mode only)'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed'
)
def pipeline(
    mode: str,
    ticker: str,
    start_date: str,
    lookback: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    model_path: str,
    days_ahead: int,
    seed: int
):
    """🔄 Run complete ML pipeline.
    
    Modes:
        - train: Training pipeline only
        - predict: Prediction pipeline only (requires trained model)
        - both: Train then predict
    
    Example:
        # Train and predict
        stock-ml pipeline --mode both --ticker PETR4.SA
        
        # Just predict
        stock-ml pipeline --mode predict --ticker PETR4.SA --model-path artifacts/models/best_model.pt
    """
    set_seed(seed)
    
    logger.info(f"🔄 Pipeline Mode: {mode.upper()}")
    
    try:
        # Training
        if mode in ['train', 'both']:
            logger.info(f"\n{'='*60}")
            logger.info("🚂 TRAINING PIPELINE")
            logger.info(f"{'='*60}")
            
            train_pipeline = TrainPipeline(
                ticker=ticker,
                start_date=start_date,
                lookback=lookback,
                hidden_size=hidden_size,
                num_layers=num_layers,
                epochs=epochs,
                model_save_path=model_path,
                seed=seed
            )
            
            train_results = train_pipeline.run()
            
            logger.success("\n✅ Training complete!")
            logger.info(f"Test MAE: {train_results['test_metrics']['mae']:.4f}")
            logger.info(f"Test RMSE: {train_results['test_metrics']['rmse']:.4f}")
        
        # Prediction
        if mode in ['predict', 'both']:
            logger.info(f"\n{'='*60}")
            logger.info("🔮 PREDICTION PIPELINE")
            logger.info(f"{'='*60}")
            
            predict_pipeline = PredictPipeline(
                model_path=model_path,
                ticker=ticker,
                lookback=lookback
            )
            
            from datetime import datetime
            output_path = f"data/predictions/{ticker.replace('.SA', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            predictions_df = predict_pipeline.predict(
                days_ahead=days_ahead,
                output_path=output_path
            )
            
            logger.success("\n✅ Predictions complete!")
            logger.info(f"\n{predictions_df.to_string(index=False)}")
            logger.info(f"Saved to: {output_path}")
        
        logger.success(f"\n{'='*60}")
        logger.success("🎉 Pipeline workflow complete!")
        logger.success(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise click.ClickException(str(e))
