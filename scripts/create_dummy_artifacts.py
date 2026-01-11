#!/usr/bin/env python3
"""
Cria artifacts dummy para testar a API sem treinar.
Útil para validação rápida da estrutura da API.
"""

import json
from pathlib import Path
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Diretórios
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

print("🎨 Criando artifacts dummy para teste...")

# 1. Criar modelo dummy
print("   📦 Criando model.pt...")
from src.ml.models.lstm import StockLSTM

model = StockLSTM(
    input_size=18,  # 18 features (OHLCV + indicators)
    hidden_size=100,
    num_layers=3,
    dropout=0.3,
    output_size=1
)

# Salvar modelo
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': 18,
    'hidden_size': 100,
    'num_layers': 3,
    'dropout': 0.3,
    'output_size': 1,
    'best_val_loss': 0.001,
    'epoch': 1
}, artifacts_dir / "model.pt")

print("      ✅ model.pt criado")

# 2. Criar scaler dummy
print("   📦 Criando scaler.pkl...")
scaler = MinMaxScaler()
# Fit com dados dummy (18 features)
dummy_data = np.random.randn(100, 18)
scaler.fit(dummy_data)
joblib.dump(scaler, artifacts_dir / "scaler.pkl")
print("      ✅ scaler.pkl criado")

# 3. Criar config.json
print("   📦 Criando config.json...")
config = {
    "ticker": "PETR4.SA",
    "lookback": 60,
    "num_features": 18,
    "hidden_size": 100,
    "num_layers": 3,
    "dropout": 0.3,
    "model_type": "StockLSTM",
    "note": "Dummy artifacts for testing"
}

with open(artifacts_dir / "config.json", 'w') as f:
    json.dump(config, f, indent=2)
print("      ✅ config.json criado")

print("\n✅ Artifacts dummy criados com sucesso!")
print(f"\n📂 Arquivos em {artifacts_dir}:")
for file in artifacts_dir.iterdir():
    size = file.stat().st_size / 1024  # KB
    print(f"   - {file.name}: {size:.1f} KB")

print("\n⚠️  IMPORTANTE: Estes são artifacts dummy!")
print("   Para produção, treine um modelo real: ./scripts/local_train.sh")
