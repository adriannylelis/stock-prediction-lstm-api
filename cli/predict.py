"""Predict command - Batch predictions."""

import click
from pathlib import Path
import pandas as pd
import torch
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.data.ingestion import StockDataIngestion
from src.ml.data.feature_engineering import TechnicalIndicators
from src.ml.data.preprocessing import StockPreprocessor
from src.ml.models.lstm import StockLSTM
from src.ml.utils.device import get_device


@click.command()
@click.option(
    '--model-path',
    type=str,
    required=True,
    help='Path to trained model (.pt file)'
)
@click.option(
    '--ticker',
    type=str,
    required=True,
    help='Stock ticker symbol (e.g., PETR4.SA)'
)
@click.option(
    '--days-ahead',
    type=int,
    default=5,
    help='Number of days to predict'
)
@click.option(
    '--lookback',
    type=int,
    default=60,
    help='Lookback period (must match training)'
)
@click.option(
    '--output',
    type=str,
    default=None,
    help='Output CSV path (default: data/predictions/{ticker}_{date}.csv)'
)
def predict(
    model_path: str,
    ticker: str,
    days_ahead: int,
    lookback: int,
    output: str
):
    """🔮 Generate batch predictions.
    
    Load trained model and predict future prices.
    
    Example:
        stock-ml predict --model-path artifacts/models/best_model.pt --ticker PETR4.SA --days-ahead 5
    """
    device = get_device()
    
    logger.info(f"🔮 Batch Prediction: {ticker}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Days ahead: {days_ahead}")
    
    try:
        # 1. Load model
        logger.info("📦 Loading model...")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        model = StockLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("✓ Model loaded")
        
        # 2. Get latest data
        logger.info("📥 Fetching latest data...")
        ingestion = StockDataIngestion(ticker=ticker)
        df = ingestion.download_and_validate()
        
        # Feature engineering
        tech_ind = TechnicalIndicators(df)
        df = tech_ind.add_all_indicators()
        df = tech_ind.fill_missing_values()
        logger.info(f"✓ Fetched {len(df)} records")
        
        # 3. Preprocess
        logger.info("⚙️ Preprocessing...")
        preprocessor = StockPreprocessor(lookback_period=lookback)
        
        # Use only last lookback points for prediction
        last_sequence = df.tail(lookback)
        
        # Normalize
        normalized_df = preprocessor.fit_transform(last_sequence)
        
        # Create sequence
        features = normalized_df[['Close']].values
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 4. Predict
        logger.info(f"🔮 Generating {days_ahead}-day predictions...")
        predictions = []
        
        with torch.no_grad():
            current_sequence = X
            
            for day in range(days_ahead):
                # Predict next value
                pred = model(current_sequence)
                predictions.append(pred.cpu().item())
                
                # Update sequence (rolling window)
                # Append prediction and remove oldest
                new_point = pred.unsqueeze(1)  # Shape: [1, 1, 1]
                current_sequence = torch.cat([current_sequence[:, 1:, :], new_point], dim=1)
        
        # Denormalize predictions
        scaler = preprocessor.scaler
        predictions_denorm = scaler.inverse_transform([[p] for p in predictions])
        predictions_denorm = [p[0] for p in predictions_denorm]
        
        # 5. Format results
        last_date = df.index[-1]
        prediction_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        results_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Close': predictions_denorm
        })
        
        # 6. Save
        if output is None:
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            output = f"data/predictions/{ticker.replace('.SA', '')}_{date_str}.csv"
        
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        # Display
        logger.success(f"\n{'='*60}")
        logger.info("📊 Predictions:")
        logger.info(f"\n{results_df.to_string(index=False)}")
        logger.success(f"{'='*60}")
        logger.success(f"💾 Saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise click.ClickException(str(e))
