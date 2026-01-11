#!/bin/bash

# Script de validação rápida - SEM TREINO
# Usa o GitHub Actions para gerar artifacts

set -e

echo "========================================="
echo "🚀 VALIDAÇÃO RÁPIDA (SEM TREINO LOCAL)"
echo "========================================="
echo ""

echo "ℹ️  Este script valida a API sem treinar localmente."
echo "   Os artifacts virão do GitHub Release após o treino automatizado."
echo ""

# 1. Verificar estrutura
echo "1️⃣  Verificando estrutura do código..."
if [ ! -d "src/ml/models" ]; then
    echo "   ❌ src/ml/models/ não encontrado!"
    exit 1
fi

if [ ! -f "src/ml/models/lstm.py" ]; then
    echo "   ❌ src/ml/models/lstm.py não encontrado!"
    exit 1
fi

echo "   ✅ Estrutura OK"

# 2. Build Docker (simulando produção)
echo ""
echo "2️⃣  Building Docker container (dev mode - sem artifacts)..."
docker stop test-backend 2>/dev/null || true
docker rm test-backend 2>/dev/null || true

# Build em modo dev (sem download de artifacts)
docker build --build-arg DOWNLOAD_ARTIFACTS=false -t stock-api-backend .

echo ""
echo "3️⃣  Rodando container..."
docker run -d -p 5001:5001 --name test-backend stock-api-backend

echo "   Aguardando API inicializar..."
sleep 15

# 4. Testar endpoints básicos
echo ""
echo "4️⃣  Testando endpoints básicos..."

# Health
echo "   Testing /health..."
response=$(curl -s http://localhost:5001/health)
if echo "$response" | grep -q "healthy"; then
    echo "   ✅ Health check OK"
else
    echo "   ❌ Health check failed"
    docker logs test-backend
    exit 1
fi

# Model Info
echo "   Testing /model/info..."
response=$(curl -s http://localhost:5001/model/info)
if echo "$response" | grep -q "architecture"; then
    echo "   ✅ Model info OK"
else
    echo "   ❌ Model info failed"
fi

echo ""
echo "========================================="
echo "✅ VALIDAÇÃO BÁSICA CONCLUÍDA!"
echo "========================================="
echo ""
echo "📝 Próximos passos:"
echo ""
echo "   Para testar predições (requer modelo treinado):"
echo "   1. Execute GitHub Actions: Train Model Weekly"
echo "   2. Aguarde artifacts serem criados no Release"
echo "   3. O Docker baixará automaticamente no próximo build"
echo ""
echo "   Para treinar localmente (requer ambiente Python):"
echo "   1. Configure venv: python3 -m venv venv && source venv/bin/activate"
echo "   2. Instale deps: pip install -r requirements.txt"
echo "   3. Treine: EPOCHS=10 ./scripts/local_train.sh"
echo ""
echo "   Container rodando em: http://localhost:5001"
echo "   Ver logs: docker logs test-backend"
echo "   Parar: docker stop test-backend && docker rm test-backend"
