"""Evaluate command - Model evaluation."""

import click
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.data.ingestion import StockDataIngestion
from src.ml.data.feature_engineering import TechnicalIndicators
from src.ml.data.preprocessing import StockPreprocessor
from src.ml.models.lstm import StockLSTM
from src.ml.training.metrics import calculate_all_metrics, print_metrics
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
    '--start-date',
    type=str,
    default='2020-01-01',
    help='Start date for data (YYYY-MM-DD)'
)
@click.option(
    '--lookback',
    type=int,
    default=60,
    help='Lookback period (must match training)'
)
def evaluate(
    model_path: str,
    ticker: str,
    start_date: str,
    lookback: int
):
    """📊 Evaluate model performance.
    
    Calculate metrics on test set: MAE, RMSE, MAPE, R², Directional Accuracy.
    
    Example:
        stock-ml evaluate --model-path artifacts/models/best_model.pt --ticker PETR4.SA
    """
    device = get_device()
    
    # Get project root directory (2 levels up from cli/evaluate.py)
    project_root = Path(__file__).parent.parent.resolve()
    
    # Convert model_path to absolute path relative to project root
    model_path_obj = Path(model_path)
    if not model_path_obj.is_absolute():
        model_path = str(project_root / model_path)
    
    logger.info(f"📊 Model Evaluation: {ticker}")
    logger.info(f"Model: {model_path}")
    
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
        
        # 2. Prepare data
        logger.info("📥 Preparing test data...")
        ingestion = StockDataIngestion(ticker=ticker, start_date=start_date)
        df = ingestion.download_and_validate()
        
        tech_ind = TechnicalIndicators(df)
        df = tech_ind.add_all_indicators()
        df = tech_ind.fill_missing_values()
        
        preprocessor = StockPreprocessor(
            lookback_period=lookback,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        data = preprocessor.prepare_data(df)
        
        test_dataset = TensorDataset(data['X_test'], data['y_test'])
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"✓ Test set: {len(data['X_test'])} samples")
        
        # 3. Generate predictions
        logger.info("🔮 Generating predictions...")
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                preds = model(X_batch)
                
                predictions.extend(preds.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())
        
        # 4. Calculate metrics
        logger.info("📈 Calculating metrics...")
        metrics = calculate_all_metrics(actuals, predictions)
        
        # 5. Display results
        logger.success(f"\n{'='*60}")
        logger.info("📊 EVALUATION RESULTS")
        logger.success(f"{'='*60}")
        print_metrics(metrics)
        logger.success(f"{'='*60}")
        
        # Save metrics
        import json
        metrics_path = Path(model_path).parent / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"💾 Metrics saved to: {metrics_path}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise click.ClickException(str(e))
