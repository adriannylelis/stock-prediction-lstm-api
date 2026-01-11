#!/bin/bash

# Script completo de validação: Treina → Testa API

set -e

echo "========================================="
echo "🎯 VALIDAÇÃO COMPLETA"
echo "========================================="
echo ""

# 1. Treinar modelo (ou usar existente)
echo "1️⃣  Verificando modelo..."

if [ -f "artifacts/models/best_model.pt" ] && [ -f "artifacts/models/scalers/scaler.pkl" ]; then
    echo "   ✅ Artifacts encontrados, pulando treino"
    echo "   Use 'rm -rf artifacts/models/*' se quiser retreinar"
else
    echo "   🎯 Treinando modelo (treino rápido com 5 epochs para validação)..."
    EPOCHS=5 BATCH_SIZE=32 ./scripts/local_train.sh
fi

echo ""
echo "========================================="

# 2. Validar artifacts
echo "2️⃣  Validando artifacts gerados..."
./scripts/validate_artifacts.sh

echo ""
echo "========================================="

# 3. Rebuild Docker
echo "3️⃣  Rebuilding Docker container..."
docker stop test-backend 2>/dev/null || true
docker rm test-backend 2>/dev/null || true
docker build --build-arg DOWNLOAD_ARTIFACTS=false -t stock-api-backend .

echo ""
echo "========================================="

# 4. Rodar container
echo "4️⃣  Iniciando container..."
docker run -d -p 5001:5001 --name test-backend stock-api-backend

echo "   Aguardando API inicializar..."
sleep 10

echo ""
echo "========================================="

# 5. Testar endpoints
echo "5️⃣  Testando API..."
./scripts/test_backend_complete.sh

echo ""
echo "========================================="
echo "✅ VALIDAÇÃO COMPLETA!"
echo "========================================="
echo ""
echo "📝 Próximos passos:"
echo "   1. git add ."
echo "   2. git commit -m 'fix: add LSTM model and update API response format'"
echo "   3. git push origin feat/integration"
echo "   4. Merge para master quando estiver pronto"
