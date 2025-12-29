"""Example: Hyperparameter tuning with Optuna.

This script demonstrates how to use HyperparameterTuner to find optimal
hyperparameters for the LSTM model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.ml.data.ingestion import StockDataIngestion
from src.ml.data.feature_engineering import TechnicalIndicators
from src.ml.data.preprocessing import StockPreprocessor
from src.ml.training.hyperparameter_tuner import HyperparameterTuner
from src.ml.utils.device import get_device
from src.ml.utils.seed import set_seed


def main():
    """Run hyperparameter tuning."""
    # Setup
    set_seed(42)
    device = get_device()
    
    logger.info("🎯 Starting Hyperparameter Tuning")
    
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
    
    # 3. Preprocessing
    logger.info("⚙️ Step 3: Preprocessing")
    preprocessor = StockPreprocessor(
        lookback_period=60,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    data = preprocessor.prepare_data(df)
    
    logger.info(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
    
    # 4. Hyperparameter Tuning
    logger.info("🔍 Step 4: Hyperparameter Optimization with Optuna")
    logger.info("This will run 50 trials (may take 30-60 minutes)")
    
    tuner = HyperparameterTuner(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        n_trials=50,
        device=str(device),
        study_name='petr4_lstm_tuning',
        experiment_name='hyperparameter_tuning'
    )
    
    # Run optimization
    best_params = tuner.optimize(show_progress=True)
    
    # Get history
    history = tuner.get_optimization_history()
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("🏆 OPTIMIZATION RESULTS")
    logger.info("="*60)
    logger.info(f"Best Validation Loss: {history['best_value']:.6f}")
    logger.info(f"Best Trial: #{history['best_trial']}")
    logger.info(f"Total Trials: {history['n_trials']}")
    logger.info("\n📋 Best Hyperparameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info("="*60)
    
    # Save optimization plots
    logger.info("📊 Saving optimization plots...")
    tuner.plot_optimization_history('artifacts/plots/optuna_optimization.png')
    
    # Save best config
    import json
    best_config = tuner.get_best_model_config()
    config_path = 'artifacts/best_hyperparameters.json'
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"💾 Saved best config to: {config_path}")
    
    logger.success("✅ Hyperparameter tuning complete!")


if __name__ == "__main__":
    main()
