"""Example: Advanced persistence features.

Demonstrates auto-cleanup, compression, and MLflow integration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import time

from src.ml.utils.persistence import DataVersionManager, ArtifactManager


def main():
    """Demonstrate advanced persistence features."""
    
    logger.info("="*60)
    logger.info("ADVANCED PERSISTENCE DEMO")
    logger.info("="*60)
    
    # 1. Auto-cleanup demonstration
    logger.info("\n🗑️ Demo 1: Auto-cleanup of old versions")
    
    # Create manager with auto-cleanup
    manager_cleanup = DataVersionManager(
        base_path="data/versioned_demo",
        auto_cleanup=True,
        max_versions=3,
        compress_after_days=0  # Compress immediately for demo
    )
    
    # Create 5 versions (should auto-cleanup to keep only 3)
    logger.info("Creating 5 versions (max_versions=3)...")
    for i in range(5):
        df = pd.DataFrame({
            'Close': np.random.normal(30, 5, 100),
            'Volume': np.random.normal(1e6, 2e5, 100)
        })
        version = manager_cleanup.save(
            df,
            ticker="TEST.SA",
            metadata={'version_num': i+1}
        )
        time.sleep(0.1)  # Small delay to ensure different timestamps
    
    # Check remaining versions
    versions = manager_cleanup.list_versions("TEST.SA")
    logger.info(f"Remaining versions: {len(versions)} (expected: 3)")
    for v in versions:
        logger.info(f"  - {v['version']} (v{v['custom']['version_num']})")
    
    # 2. Compression demonstration
    logger.info("\n📦 Demo 2: Dataset compression")
    
    manager_compress = DataVersionManager(
        base_path="data/versioned_demo2",
        compress_after_days=0  # Compress immediately
    )
    
    # Create large dataset
    logger.info("Creating dataset for compression...")
    large_df = pd.DataFrame({
        'Close': np.random.normal(30, 5, 10000),
        'Volume': np.random.normal(1e6, 2e5, 10000),
        'RSI': np.random.normal(50, 10, 10000),
        'MACD': np.random.normal(0, 1, 10000)
    })
    
    version = manager_compress.save(
        large_df,
        ticker="COMPRESS.SA",
        metadata={'test': 'compression'}
    )
    
    # Check if compressed
    ticker_clean = "COMPRESS"
    version_dir = Path(f"data/versioned_demo2/{ticker_clean}/{version}")
    compressed_file = version_dir / "data.parquet.gz"
    uncompressed_file = version_dir / "data.parquet"
    
    if compressed_file.exists():
        logger.success(f"✓ Dataset compressed: {compressed_file.name}")
    elif uncompressed_file.exists():
        logger.info(f"Dataset not yet compressed (age < {manager_compress.compress_after_days} days)")
    
    # 3. MLflow Model Registry
    logger.info("\n📝 Demo 3: MLflow Model Registry integration")
    
    artifact_manager = ArtifactManager(use_mlflow=True)
    
    # Save config
    config = {
        'model_type': 'LSTM',
        'hidden_size': 50,
        'learning_rate': 0.001
    }
    artifact_manager.save_config(config, "demo_config")
    logger.success("✓ Config saved")
    
    # Note: Model registration requires an active MLflow run
    logger.info("\nMLflow Model Registry features:")
    logger.info("  - register_model_mlflow(name, run_id)")
    logger.info("  - get_latest_model_version(name, stage)")
    logger.info("  - transition_model_stage(name, version, stage)")
    
    # 4. Manual cleanup
    logger.info("\n🧹 Demo 4: Manual cleanup")
    
    logger.info("Cleaning up demo data...")
    import shutil
    if Path("data/versioned_demo").exists():
        shutil.rmtree("data/versioned_demo")
    if Path("data/versioned_demo2").exists():
        shutil.rmtree("data/versioned_demo2")
    logger.success("✓ Demo data cleaned up")
    
    logger.success("\n✅ Demo complete!")
    
    # Summary
    logger.info("\n📋 Summary of Features:")
    logger.info("  ✓ Auto-cleanup: Keeps only N latest versions")
    logger.info("  ✓ Compression: Automatically compresses old datasets")
    logger.info("  ✓ MLflow Registry: Version models in production")
    logger.info("  ✓ Manual cleanup: cleanup_all(ticker, keep_latest=N)")


if __name__ == "__main__":
    main()
