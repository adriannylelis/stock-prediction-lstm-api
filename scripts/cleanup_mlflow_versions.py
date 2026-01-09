"""
Cleanup old MLflow model versions.

Remove old Staging versions and keep only Production + recent Archived.
"""

from pathlib import Path

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient


def cleanup_old_versions(
    model_name: str = "stock-lstm-model",
    keep_production: bool = True,
    keep_archived: int = 5,
    delete_staging: bool = True,
    dry_run: bool = True
):
    """Cleanup old model versions.
    
    Args:
        model_name: Name of the model
        keep_production: Keep all Production versions
        keep_archived: Number of recent Archived versions to keep
        delete_staging: Delete all Staging versions
        dry_run: If True, only show what would be deleted
    """
    tracking_uri = f"file:{Path.cwd()}/data/mlflow/tracking"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    logger.info(f"{'DRY RUN - ' if dry_run else ''}Cleaning up {model_name}")
    logger.info("=" * 80)

    # Get all versions
    versions = client.search_model_versions(f"name='{model_name}'")

    # Organize by stage
    by_stage = {
        "Production": [],
        "Staging": [],
        "Archived": [],
        "None": []
    }

    for v in versions:
        by_stage[v.current_stage].append(v)

    # Sort Archived by version (descending)
    by_stage["Archived"].sort(key=lambda v: int(v.version), reverse=True)

    # Summary
    logger.info("Current state:")
    logger.info(f"  Production: {len(by_stage['Production'])} versions")
    logger.info(f"  Archived: {len(by_stage['Archived'])} versions")
    logger.info(f"  Staging: {len(by_stage['Staging'])} versions")
    logger.info(f"  None: {len(by_stage['None'])} versions")
    logger.info("")

    deleted_count = 0

    # Delete old Staging versions
    if delete_staging:
        logger.info(f"Deleting {len(by_stage['Staging'])} Staging versions...")
        for v in by_stage['Staging']:
            logger.info(f"  DELETE v{v.version} (Staging)")
            if not dry_run:
                client.delete_model_version(model_name, v.version)
            deleted_count += 1

    # Delete old Archived versions (keep recent ones)
    old_archived = by_stage["Archived"][keep_archived:]
    if old_archived:
        logger.info(f"\nDeleting {len(old_archived)} old Archived versions (keeping {keep_archived} most recent)...")
        for v in old_archived:
            logger.info(f"  DELETE v{v.version} (Archived)")
            if not dry_run:
                client.delete_model_version(model_name, v.version)
            deleted_count += 1

    # Delete None stage
    if by_stage["None"]:
        logger.info(f"\nDeleting {len(by_stage['None'])} versions with stage=None...")
        for v in by_stage["None"]:
            logger.info(f"  DELETE v{v.version} (None)")
            if not dry_run:
                client.delete_model_version(model_name, v.version)
            deleted_count += 1

    logger.info("")
    logger.info("=" * 80)
    if dry_run:
        logger.warning(f"DRY RUN: Would delete {deleted_count} versions")
        logger.info("Run with --execute to actually delete")
    else:
        logger.success(f"Deleted {deleted_count} versions")

    # Show remaining
    remaining = client.search_model_versions(f"name='{model_name}'")
    logger.info(f"\nRemaining versions: {len(remaining)}")
    for v in sorted(remaining, key=lambda x: int(x.version), reverse=True):
        logger.info(f"  v{v.version}: {v.current_stage}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup old MLflow model versions")
    parser.add_argument("--model", default="stock-lstm-model", help="Model name")
    parser.add_argument("--keep-archived", type=int, default=5, help="Number of Archived versions to keep")
    parser.add_argument("--keep-staging", action="store_true", help="Keep Staging versions")
    parser.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")

    args = parser.parse_args()

    cleanup_old_versions(
        model_name=args.model,
        keep_archived=args.keep_archived,
        delete_staging=not args.keep_staging,
        dry_run=not args.execute
    )
