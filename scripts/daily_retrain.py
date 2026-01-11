"""
Daily Automated Retrain Script.

Complete pipeline: Train → Compare → Promote → Deploy

Designed to run as cronjob for automated model updates.

Usage:
    python scripts/daily_retrain.py
    
Cronjob example:
    0 2 * * * cd /app && python scripts/daily_retrain.py >> logs/retrain.log 2>&1
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlops.pipelines.promotion_pipeline import AutoPromotionPipeline
from src.mlops.pipelines.training_pipeline import AutoTrainingPipeline


def main():
    """Run complete daily retrain pipeline."""
    start_time = datetime.now()

    logger.info("=" * 100)
    logger.info("🚀 DAILY AUTOMATED RETRAIN PIPELINE")
    logger.info("=" * 100)
    logger.info(f"Started: {start_time.isoformat()}")
    logger.info("=" * 100)

    try:
        # ==================================================
        # STEP 1: TRAINING
        # ==================================================
        logger.info("\n" + "=" * 100)
        logger.info("📊 STEP 1/2: TRAINING NEW MODEL")
        logger.info("=" * 100)

        trainer = AutoTrainingPipeline(
            tickers="blue_chips",              # Use top 10 B3 stocks
            min_data_quality=0.95,             # 95% data quality threshold
            min_samples=1000,                  # Minimum 1000 samples
            start_date="2020-01-01",           # 5 years of data
            epochs=100,                        # Full training
            hidden_size=100,
            num_layers=3,
            dropout=0.3,
            batch_size=64,
            learning_rate=0.001,
            early_stopping_patience=15,
            experiment_name="lstm-multi-ticker",
        )

        training_result = trainer.run()

        if not training_result.success:
            logger.error(f"❌ Training failed: {training_result.error}")
            logger.error("Pipeline aborted")
            sys.exit(1)

        logger.success("\n✅ Training completed successfully!")
        logger.info(f"Model URI: {training_result.model_uri}")
        logger.info(f"Version: {training_result.version}")
        logger.info(f"Metrics: {training_result.metrics}")

        # ==================================================
        # STEP 2: PROMOTION & DEPLOYMENT
        # ==================================================
        logger.info("\n" + "=" * 100)
        logger.info("🎯 STEP 2/2: PROMOTION & DEPLOYMENT")
        logger.info("=" * 100)

        # Only promote if we have a version number
        if training_result.version is None:
            logger.warning("⚠️ Model not registered in MLflow, skipping promotion")
            logger.info("To enable promotion, ensure model is registered during training")
            logger.success("\nPipeline completed (training only)")
            return

        promoter = AutoPromotionPipeline(
            new_model_version=training_result.version,
            model_name="stock-lstm-model",     # Correct model name
            production_model="auto",           # Auto-detect current production
            auto_promote_criteria={
                "r2_improvement": 0.05,        # 5% better R²
                "mae_improvement": 0.02,       # 2% better MAE
                "min_r2": 0.15,                # Minimum R² = 0.15
                "max_mae": 0.04,               # Maximum MAE = 0.04
            },
            auto_deploy=True,                  # Auto-deploy if promoted
        )

        promotion_result = promoter.run_with_deploy()

        # ==================================================
        # FINAL RESULTS
        # ==================================================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 100)
        logger.success("✅ DAILY RETRAIN PIPELINE COMPLETED")
        logger.info("=" * 100)
        logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}min)")
        logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 100)

        logger.info("\n📊 TRAINING RESULTS:")
        logger.info(f"  Model: {training_result.model_uri}")
        logger.info(f"  Version: {training_result.version}")
        logger.info(f"  MAE: {training_result.metrics['MAE']:.4f}")
        logger.info(f"  RMSE: {training_result.metrics['RMSE']:.4f}")
        logger.info(f"  R²: {training_result.metrics['R2']:.4f}")

        logger.info("\n🎯 PROMOTION RESULTS:")
        logger.info(f"  Promoted: {'✅ YES' if promotion_result.promoted else '❌ NO'}")
        logger.info(f"  Deployed: {'✅ YES' if promotion_result.deployed else '❌ NO'}")
        logger.info(f"  Reason: {promotion_result.reason}")

        if promotion_result.comparison and promotion_result.comparison.get("improvements"):
            improvements = promotion_result.comparison["improvements"]
            logger.info("\n📈 IMPROVEMENTS:")
            logger.info(f"  R²: {improvements.get('r2', 0):+.1f}%")
            logger.info(f"  MAE: {improvements.get('mae', 0):+.1f}%")

        logger.info("=" * 100)

        if promotion_result.promoted and promotion_result.deployed:
            logger.success("🎉 New model deployed to production!")
        elif promotion_result.promoted:
            logger.warning("⚠️ Model promoted but deployment failed")
        else:
            logger.info("ℹ️ Current production model still the best")

        logger.info("=" * 100)

    except Exception as e:
        logger.error("\n" + "=" * 100)
        logger.error("❌ PIPELINE FAILED")
        logger.error("=" * 100)
        logger.error(f"Error: {str(e)}")
        logger.exception(e)
        logger.error("=" * 100)
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging for cronjob
    logger.add(
        "logs/daily_retrain_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )

    main()
