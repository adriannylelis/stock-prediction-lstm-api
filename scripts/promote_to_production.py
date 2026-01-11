"""Promote latest model version to Production."""
import mlflow

mlflow.set_tracking_uri("file:data/mlflow/tracking")
client = mlflow.tracking.MlflowClient()

MODEL_NAME = "stock-lstm-model"

print("\n" + "="*60)
print("Promoting Model to Production")
print("="*60)

try:
    # Get latest version
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda x: int(x.version))

    print(f"\n📦 Model: {MODEL_NAME}")
    print(f"🔧 Latest Version: {latest.version} (currently {latest.current_stage})")
    print(f"   Run ID: {latest.run_id}")

    # Archive current production
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    if prod_versions:
        for pv in prod_versions:
            print(f"\n📥 Archiving current Production version {pv.version}...")
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=pv.version,
                stage="Archived"
            )
            print(f"   ✅ Version {pv.version} archived")

    # Promote latest to production
    print(f"\n🚀 Promoting version {latest.version} to Production...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=False
    )

    print("\n✅ SUCCESS!")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Version: {latest.version}")
    print("   Stage: Production")
    print("   🎯 Ready for API predictions!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60 + "\n")
