"""Check Production model configuration."""
import mlflow

mlflow.set_tracking_uri("file:data/mlflow/tracking")
client = mlflow.tracking.MlflowClient()

print("\n" + "="*60)
print("Production Model Configuration")
print("="*60)

# Get production model
versions = client.search_model_versions('name="stock-lstm-model"')
prod_versions = [v for v in versions if v.current_stage == "Production"]

if not prod_versions:
    print("\n❌ No Production model found!")
else:
    prod = prod_versions[0]
    print("\n📦 Production Model:")
    print(f"   Version: {prod.version}")
    print(f"   Run ID: {prod.run_id}")

    # Get run details
    run = client.get_run(prod.run_id)

    print("\n📊 Model Parameters:")
    for k, v in sorted(run.data.params.items()):
        if any(x in k.lower() for x in ['num_features', 'input', 'hidden', 'num_layers', 'dropout', 'num_tickers']):
            print(f"   {k}: {v}")

    print("\n📈 Test Metrics:")
    for k, v in sorted(run.data.metrics.items()):
        if 'test' in k:
            print(f"   {k}: {float(v):.4f}")

print("\n" + "="*60 + "\n")
