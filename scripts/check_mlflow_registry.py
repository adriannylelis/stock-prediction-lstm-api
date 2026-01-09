"""Check MLflow model registry status."""
import mlflow

mlflow.set_tracking_uri("file:data/mlflow/tracking")
client = mlflow.tracking.MlflowClient()

print("\n" + "="*60)
print("MLflow Model Registry Status")
print("="*60)

try:
    models = client.search_registered_models()

    if not models:
        print("\n⚠️  No models registered yet!")
    else:
        for model in models:
            print(f"\n📦 Model: {model.name}")
            print(f"   Description: {model.description or 'N/A'}")

            versions = client.search_model_versions(f"name='{model.name}'")
            if versions:
                print("\n   Versions:")
                for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
                    stage_icon = "🚀" if v.current_stage == "Production" else "🔧" if v.current_stage == "Staging" else "📋"
                    print(f"   {stage_icon} Version {v.version}: {v.current_stage}")
                    print(f"      Run ID: {v.run_id[:8]}...")
                    if v.current_stage == "Production":
                        print("      ✅ ACTIVE IN PRODUCTION")
            else:
                print("   No versions found")

except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "="*60 + "\n")
