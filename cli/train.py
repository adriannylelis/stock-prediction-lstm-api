"""Train command - Full training pipeline."""

import click
from pathlib import Path
from loguru import logger
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.data.ingestion import StockDataIngestion
from src.ml.data.feature_engineering import TechnicalIndicators
from src.ml.data.preprocessing import StockPreprocessor
from src.ml.models.lstm import create_model
from src.ml.training.trainer import Trainer
from src.ml.utils.device import get_device
from src.ml.utils.seed import set_seed


@click.command()
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
    '--end-date',
    type=str,
    default=None,
    help='End date for data (YYYY-MM-DD, default: today)'
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
    '--dropout',
    type=float,
    default=0.2,
    help='Dropout rate'
)
@click.option(
    '--lr',
    type=float,
    default=0.001,
    help='Learning rate'
)
@click.option(
    '--epochs',
    type=int,
    default=100,
    help='Maximum training epochs'
)
@click.option(
    '--batch-size',
    type=int,
    default=32,
    help='Batch size'
)
@click.option(
    '--experiment-name',
    type=str,
    default=None,
    help='MLflow experiment name (default: lstm-{ticker})'
)
@click.option(
    '--model-path',
    type=str,
    default='artifacts/models/best_model.pt',
    help='Path to save best model'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility'
)
def train(
    ticker: str,
    start_date: str,
    end_date: str,
    lookback: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    experiment_name: str,
    model_path: str,
    seed: int
):
    """🚂 Train LSTM model on stock data.
    
    Full training pipeline: data ingestion → feature engineering →
    preprocessing → training → evaluation.
    
    Example:
        stock-ml train --ticker PETR4.SA --epochs 100 --lr 0.001
    """
    # Setup
    set_seed(seed)
    device = get_device()
    
    if experiment_name is None:
        experiment_name = f"lstm-{ticker.replace('.SA', '').lower()}"
    
    logger.info(f"🚀 Training Pipeline: {ticker}")
    logger.info(f"Device: {device}")
    logger.info(f"Experiment: {experiment_name}")
    
    try:
        # 1. Data Ingestion
        logger.info("📥 Step 1/5: Data Ingestion")
        ingestion = StockDataIngestion(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        df = ingestion.download_and_validate()
        logger.info(f"✓ Downloaded {len(df)} records")
        
        # 2. Feature Engineering
        logger.info("🔧 Step 2/5: Feature Engineering")
        tech_ind = TechnicalIndicators(df)
        df = tech_ind.add_all_indicators()
        df = tech_ind.fill_missing_values()
        logger.info(f"✓ Generated {df.shape[1]} features")
        
        # 3. Preprocessing
        logger.info("⚙️ Step 3/5: Preprocessing")
        preprocessor = StockPreprocessor(
            lookback_period=lookback,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        data = preprocessor.prepare_data(df)
        logger.info(
            f"✓ Train: {len(data['X_train'])}, "
            f"Val: {len(data['X_val'])}, "
            f"Test: {len(data['X_test'])}"
        )
        
        # 4. Model Creation
        logger.info("🧠 Step 4/5: Creating Model")
        input_size = data['X_train'].shape[2]
        model = create_model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        )
        logger.info(f"✓ Model created: {model.count_parameters():,} parameters")
        
        # 5. Training
        logger.info("🏋️ Step 5/5: Training")
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=lr,
            loss_function='MSE',
            early_stopping_patience=10,
            experiment_name=experiment_name,
            model_save_path=model_path
        )
        
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(data['X_train'], data['y_train'])
        val_dataset = TensorDataset(data['X_val'], data['y_val'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs
        )
        
        # Results
        best_val_loss = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val_loss) + 1
        
        logger.success(
            f"\n{'='*60}\n"
            f"✅ Training Complete!\n"
            f"{'='*60}\n"
            f"Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch})\n"
            f"Model saved: {model_path}\n"
            f"Experiment: {experiment_name}\n"
            f"{'='*60}"
        )
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise click.ClickException(str(e))
