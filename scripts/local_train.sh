#!/bin/bash

###############################################################################
# Script: Local Training - PETR4.SA
# Description: Treina o modelo localmente para testes antes do GitHub Actions
# Usage: ./scripts/local_train.sh
###############################################################################

set -e  # Exit on error

echo "🚀 Starting local training for PETR4.SA..."
echo ""

# Verificar se está no diretório correto
if [ ! -f "cli/train.py" ]; then
    echo "❌ ERROR: Must run from project root directory"
    exit 1
fi

# Criar pasta artifacts se não existir
mkdir -p artifacts

# Limpar artifacts antigos (opcional)
read -p "🗑️  Clear old artifacts? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf artifacts/*
    echo "✅ Artifacts cleared"
fi

# Configurações de treino
TICKER="${TICKER:-PETR4.SA}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-100}"
NUM_LAYERS="${NUM_LAYERS:-3}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-0.001}"

echo "📊 Training Configuration:"
echo "   Ticker: $TICKER"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Hidden Size: $HIDDEN_SIZE"
echo "   Num Layers: $NUM_LAYERS"
echo "   Dropout: $DROPOUT"
echo "   Learning Rate: $LR"
echo ""

# Executar treino
echo "🎯 Training model..."
python3 -m cli train \
    --ticker "$TICKER" \
    --start-date 2020-01-01 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-size "$HIDDEN_SIZE" \
    --num-layers "$NUM_LAYERS" \
    --dropout "$DROPOUT" \
    --lr "$LR"

# Validar artifacts
echo ""
echo "✅ Training completed!"
echo ""
echo "📦 Generated artifacts:"
ls -lah artifacts/

# Validar arquivos obrigatórios
if [ ! -f "artifacts/model.pt" ]; then
    echo "❌ ERROR: model.pt not found!"
    exit 1
fi

if [ ! -f "artifacts/scaler.pkl" ]; then
    echo "❌ ERROR: scaler.pkl not found!"
    exit 1
fi

echo ""
echo "✅ All required artifacts present"
echo ""
echo "🚀 Next steps:"
echo "   1. Test API locally:"
echo "      docker-compose up backend"
echo ""
echo "   2. Test prediction:"
echo "      curl -X POST http://localhost:5001/predict -H 'Content-Type: application/json' -d '{\"ticker\": \"PETR4.SA\", \"periods\": 7}'"
echo ""
echo "   3. Push to GitHub and let Actions handle the rest!"
