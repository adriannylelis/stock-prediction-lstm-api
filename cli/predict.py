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
    '--output',
    type=str,
    default=None,
    help='Output CSV path (optional)'
)
def predict(
    model_path: str,
    ticker: str,
    output: str
):
    """🔮 Predict next day's closing price.
    
    Uses last 60 days of data to predict tomorrow's close.
    
    Example:
        stock-predict predict --model-path models/petr4.pt --ticker PETR4.SA
    """
    # Fixed parameters
    lookback = 60  # Always use 60 days
    days_ahead = 1  # Always predict next day only
    device = get_device()
    
    # Get project root directory (2 levels up from cli/predict.py)
    project_root = Path(__file__).parent.parent.resolve()
    
    # Convert model_path to absolute path relative to project root
    model_path_obj = Path(model_path)
    if not model_path_obj.is_absolute():
        model_path = str(project_root / model_path)
    
    # Same for output path if provided
    if output:
        output_path_obj = Path(output)
        if not output_path_obj.is_absolute():
            output = str(project_root / output)
    
    logger.info(f"🔮 Next Day Prediction: {ticker}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Predicting: Tomorrow's close (using last 60 days)")
    
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
        # Get last 2 years of data for prediction
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # ~2 years
        
        ingestion = StockDataIngestion(
            ticker=ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        df = ingestion.download_and_validate()
        
        # Feature engineering
        tech_ind = TechnicalIndicators(df)
        df = tech_ind.add_all_indicators()
        df = tech_ind.fill_missing_values()
        logger.info(f"✓ Fetched {len(df)} records")
        
        # 3. Preprocess
        logger.info("⚙️ Preprocessing...")
        preprocessor = StockPreprocessor(lookback_period=lookback)
        
        # Select features and normalize
        features_df = df[['Close']].copy()
        features_array = features_df.values
        
        # Normalize using the same scaler from training (if available)
        # For prediction, we need to fit on available data
        normalized_data = preprocessor.normalize(features_array, fit=True)
        
        # Get last lookback points for prediction
        if len(normalized_data) < lookback:
            raise ValueError(f"Not enough data. Need at least {lookback} points, got {len(normalized_data)}")
        
        last_sequence = normalized_data[-lookback:]
        
        # Create tensor
        X = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)  # (1, lookback, features)
        
        # 4. Predict next day
        logger.info(f"🔮 Predicting tomorrow's close...")
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
        predictions_denorm = [round(p[0], 2) for p in predictions_denorm]  # Round to 2 decimals (yfinance standard)
        
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
        from datetime import datetime
        today = datetime.now().date()
        last_data_date = last_date.date()
        predicted_date = prediction_dates[0].date()
        
        # Calculate data lag
        data_lag_days = (today - last_data_date).days
        
        logger.success(f"\n{'='*60}")
        logger.info(f"📊 Prediction for {ticker}:")
        logger.info(f"   Today's Date: {today}")
        logger.info(f"   Last Available Data: {last_data_date} (${df['Close'].iloc[-1]:.2f})")
        if data_lag_days > 0:
            logger.warning(f"   Data Lag: {data_lag_days} day(s) - Market data not updated yet")
        logger.info(f"   Predicted for {predicted_date}: ${predictions_denorm[0]:.2f}")
        change_pct = ((predictions_denorm[0] / df['Close'].iloc[-1] - 1) * 100)
        logger.info(f"   Expected Change: {change_pct:+.2f}%")
        logger.success(f"{'='*60}")
        logger.success(f"💾 Saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise click.ClickException(str(e))
