"""
Example training script demonstrating the complete ML pipeline.

This script shows how to:
1. Load data from Yahoo Finance
2. Apply technical indicators
3. Preprocess data
4. Train LSTM model
5. Track experiments with MLflow
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import TensorDataset, DataLoader

from src.ml.data.ingestion import StockDataIngestion
from src.ml.data.feature_engineering import TechnicalIndicators
from src.ml.data.preprocessing import StockPreprocessor
from src.ml.models.lstm import create_model
from src.ml.training.trainer import Trainer
from src.ml.utils.device import get_device
from src.ml.utils.seed import set_seed
from loguru import logger


def main():
    """Run training pipeline."""
    # Setup
    set_seed(42)
    device = get_device()
    
    logger.info("🚀 Starting training pipeline example")
    
    # 1. Data Ingestion
    logger.info("📥 Step 1: Data Ingestion")
    ingestion = StockDataIngestion(
        ticker="PETR4.SA",
        start_date="2020-01-01",
        end_date="2024-01-01"
    )
    
    df = ingestion.download_and_validate()
    logger.info(f"Downloaded {len(df)} records")
    
    # 2. Feature Engineering
    logger.info("🔧 Step 2: Feature Engineering")
    tech_ind = TechnicalIndicators(df)
    df = tech_ind.add_all_indicators()
    df = tech_ind.fill_missing_values()
    logger.info(f"Added {len(df.columns)} features")
    
    # 3. Preprocessing
    logger.info("⚙️ Step 3: Preprocessing")
    preprocessor = StockPreprocessor(
        lookback_period=60,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    data = preprocessor.prepare_data(df)
    
    logger.info(f"Train size: {len(data['X_train'])}")
    logger.info(f"Val size: {len(data['X_val'])}")
    logger.info(f"Test size: {len(data['X_test'])}")
    
    # 4. Create DataLoaders
    logger.info("📦 Step 4: Creating DataLoaders")
    train_dataset = TensorDataset(data['X_train'], data['y_train'])
    val_dataset = TensorDataset(data['X_val'], data['y_val'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 5. Create Model
    logger.info("🧠 Step 5: Creating LSTM Model")
    model = create_model(
        input_size=1,
        hidden_size=50,
        num_layers=2,
        dropout=0.2,
        device=device
    )
    
    logger.info(f"\n{model.summary()}")
    logger.info(f"Total parameters: {model.get_num_parameters():,}")
    
    # 6. Setup Training Components
    logger.info("🎯 Step 6: Setup Training")
    
    # 7. Train Model
    logger.info("🏋️ Step 7: Training Model")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        loss_function='MSE',
        early_stopping_patience=10,
        early_stopping_min_delta=0.0001,
        experiment_name="stock_prediction_example",
        tracking_uri="file:./mlruns"
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50
    )
    
    # 8. Evaluate
    logger.info("📊 Step 8: Evaluation")
    test_dataset = TensorDataset(data['X_test'], data['y_test'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_metrics = trainer.evaluate(test_loader)
    
    logger.info("\n" + "="*50)
    logger.info("📈 FINAL TEST METRICS")
    logger.info("="*50)
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("="*50)
    
    # 9. Save Model
    logger.info("💾 Step 9: Saving Model")
    trainer.save_checkpoint(
        checkpoint_path="models/lstm_baseline.pt",
        epoch=len(history['train_loss']),
        metrics=test_metrics
    )
    
    logger.info("✅ Training pipeline completed successfully!")
    logger.info("📁 Model saved to: models/lstm_baseline.pt")
    logger.info("📊 MLflow tracking: mlruns/")
    
    return trainer, history, test_metrics


if __name__ == "__main__":
    trainer, history, metrics = main()
    
    print("\n🎉 Training completed!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Test R²: {metrics['R2']:.4f}")
