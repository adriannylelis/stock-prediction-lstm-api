"""Check local scaler configuration."""
from pathlib import Path

import joblib

scaler_path = Path("artifacts/models/scaler.pkl")

if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print("\n📊 Local Scaler Info:")
    print(f"   Path: {scaler_path}")
    print(f"   Features: {scaler.n_features_in_}")
    print(f"   Feature range: {scaler.feature_range}")
    print(f"   Min shape: {scaler.data_min_.shape}")
    print(f"   Max shape: {scaler.data_max_.shape}")
else:
    print(f"\n❌ Scaler not found at {scaler_path}")
