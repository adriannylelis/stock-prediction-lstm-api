"""Example: Data drift detection.

Demonstrates how to detect drift between datasets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger

from src.ml.monitoring.drift_detector import DriftDetector
from src.ml.utils.persistence import DataVersionManager


def main():
    """Demonstrate drift detection."""
    
    logger.info("="*60)
    logger.info("DATA DRIFT DETECTION DEMO")
    logger.info("="*60)
    
    # 1. Load reference and production data
    logger.info("\n📂 Step 1: Load datasets")
    data_manager = DataVersionManager()
    
    # Simulate: load two versions
    logger.info("Simulating: Creating reference and production datasets...")
    
    # Reference data (normal distribution)
    np.random.seed(42)
    ref_data = pd.DataFrame({
        'Close': np.random.normal(30, 5, 1000),
        'Volume': np.random.normal(1e6, 2e5, 1000),
        'RSI': np.random.normal(50, 10, 1000),
        'MACD': np.random.normal(0, 1, 1000)
    })
    
    # Production data (with drift - shifted distribution)
    prod_data = pd.DataFrame({
        'Close': np.random.normal(35, 5, 1000),  # Mean shifted
        'Volume': np.random.normal(1.2e6, 2e5, 1000),  # Mean shifted
        'RSI': np.random.normal(50, 10, 1000),  # No drift
        'MACD': np.random.normal(0.5, 1, 1000)  # Mean shifted
    })
    
    logger.info(f"Reference data: {ref_data.shape}")
    logger.info(f"Production data: {prod_data.shape}")
    
    # 2. Detect drift using KS test
    logger.info("\n📊 Step 2: Detect drift using KS test")
    detector = DriftDetector(ks_threshold=0.05)
    
    ks_report = detector.detect_drift(ref_data, prod_data)
    
    logger.info("\nKS Test Results:")
    logger.info(f"  Has drift: {ks_report['has_drift']}")
    logger.info(f"  Drifted features: {ks_report['drifted_features']}")
    logger.info(f"  Drift percentage: {ks_report['summary']['drift_percentage']:.1f}%")
    
    # Show feature scores
    logger.info("\nFeature KS Scores:")
    for feature, score in ks_report['feature_scores'].items():
        p_value = ks_report['feature_pvalues'][feature]
        status = "❌ DRIFT" if p_value < 0.05 else "✓ OK"
        logger.info(f"  {feature:15s}: {score:.4f} (p={p_value:.4f}) {status}")
    
    # 3. Detect drift using PSI
    logger.info("\n📊 Step 3: Detect drift using PSI")
    
    psi_report = detector.detect_drift_psi(ref_data, prod_data)
    
    logger.info("\nPSI Test Results:")
    logger.info(f"  Has drift: {psi_report['has_drift']}")
    logger.info(f"  Drifted features: {psi_report['drifted_features']}")
    logger.info(f"  Max PSI: {psi_report['summary']['max_psi']:.4f}")
    logger.info(f"  Avg PSI: {psi_report['summary']['avg_psi']:.4f}")
    
    # Show PSI scores
    logger.info("\nFeature PSI Scores:")
    for feature, psi in psi_report['feature_psi'].items():
        if psi < 0.1:
            status = "✓ OK"
        elif psi < 0.2:
            status = "⚠️  WARNING"
        else:
            status = "❌ CRITICAL"
        logger.info(f"  {feature:15s}: {psi:.4f} {status}")
    
    # 4. Save report
    logger.info("\n💾 Step 4: Save drift report")
    detector.save_report(
        ks_report,
        "artifacts/drift_reports/ks_report.json"
    )
    detector.save_report(
        psi_report,
        "artifacts/drift_reports/psi_report.json"
    )
    
    # 5. Recommendations
    logger.info("\n💡 Recommendations:")
    if ks_report['has_drift']:
        logger.warning("  🔴 Drift detected! Consider:")
        logger.warning("     - Retrain model with new data")
        logger.warning("     - Investigate feature changes")
        logger.warning("     - Monitor model performance")
    else:
        logger.success("  ✅ No drift detected - model is stable")
    
    logger.success("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
