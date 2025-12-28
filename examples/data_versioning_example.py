"""Example: Data versioning and persistence.

Demonstrates how to use DataVersionManager and ArtifactManager.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.ml.data.ingestion import StockDataIngestion
from src.ml.data.feature_engineering import TechnicalIndicators
from src.ml.utils.persistence import DataVersionManager, ArtifactManager


def main():
    """Demonstrate data versioning."""
    
    logger.info("="*60)
    logger.info("DATA VERSIONING & PERSISTENCE DEMO")
    logger.info("="*60)
    
    # Initialize managers
    data_manager = DataVersionManager()
    artifact_manager = ArtifactManager()
    
    # 1. Ingest and save raw data
    logger.info("\n📥 Step 1: Ingest and version raw data")
    ingestion = StockDataIngestion(ticker="PETR4.SA", start_date="2023-01-01")
    df_raw = ingestion.download_and_validate()
    
    raw_version = data_manager.save(
        df_raw,
        ticker="PETR4.SA",
        metadata={'stage': 'raw', 'source': 'yfinance'}
    )
    
    # 2. Process and save processed data
    logger.info("\n🔧 Step 2: Process and version processed data")
    tech_ind = TechnicalIndicators(df_raw)
    df_processed = tech_ind.add_all_indicators()
    df_processed = tech_ind.fill_missing_values()
    
    processed_version = data_manager.save(
        df_processed,
        ticker="PETR4.SA",
        metadata={
            'stage': 'processed',
            'features': list(df_processed.columns),
            'raw_version': raw_version
        }
    )
    
    # 3. List all versions
    logger.info("\n📋 Step 3: List all versions")
    versions = data_manager.list_versions("PETR4.SA")
    for v in versions:
        logger.info(f"  Version: {v['version']}")
        logger.info(f"    Stage: {v['custom']['stage']}")
        logger.info(f"    Records: {v['n_records']}")
        logger.info(f"    Features: {v['n_features']}")
    
    # 4. Load specific version
    logger.info("\n📂 Step 4: Load specific version")
    df_loaded = data_manager.load(raw_version, "PETR4.SA")
    logger.info(f"Loaded {len(df_loaded)} records")
    
    # 5. Save and load config
    logger.info("\n⚙️ Step 5: Save configuration")
    config = {
        'ticker': 'PETR4.SA',
        'lookback': 60,
        'hidden_size': 50,
        'data_versions': {
            'raw': raw_version,
            'processed': processed_version
        }
    }
    artifact_manager.save_config(config, "petr4_config")
    
    # 6. Load config
    loaded_config = artifact_manager.load_config("petr4_config")
    logger.info(f"Loaded config: {loaded_config}")
    
    logger.success("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
