#!/bin/bash
set -e

echo "🧪 TESTE COMPLETO PRÉ-DEPLOY"
echo "=============================="
echo ""

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para printar status
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        exit 1
    fi
}

# ==================== 1. VERIFICAR ARTIFACTS ====================
echo "📦 1. Verificando artifacts..."
if [ -d "artifacts" ] && [ -f "artifacts/model.pt" ] && [ -f "artifacts/scaler.pkl" ]; then
    MODEL_SIZE=$(du -h artifacts/model.pt | cut -f1)
    echo -e "${GREEN}✓${NC} Artifacts encontrados (model.pt: $MODEL_SIZE)"
else
    echo -e "${YELLOW}⚠${NC}  Artifacts não encontrados. Executando treino..."
    ./scripts/local_train.sh
    print_status "Treino concluído"
fi

# Validar artifacts
./scripts/validate_artifacts.sh
print_status "Artifacts validados"
echo ""

# ==================== 2. TESTAR BUILD BACKEND ====================
echo "🐳 2. Testando build do Backend Docker..."
docker build --build-arg DOWNLOAD_ARTIFACTS=false -t stock-api-backend-test . > /dev/null 2>&1
print_status "Backend Docker build OK"
echo ""

# ==================== 3. TESTAR BACKEND ====================
echo "🔧 3. Testando Backend API..."

# Parar container anterior se existir
docker stop test-backend 2>/dev/null || true
docker rm test-backend 2>/dev/null || true

# Rodar container
docker run -d -p 5001:5001 --name test-backend stock-api-backend-test > /dev/null

# Aguardar API iniciar
echo "   Aguardando API iniciar..."
sleep 5

# Testar health
HEALTH_RESPONSE=$(curl -s http://localhost:5001/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓${NC} Health check OK"
else
    echo -e "${RED}✗${NC} Health check FALHOU"
    docker logs test-backend
    docker stop test-backend && docker rm test-backend
    exit 1
fi

# Testar model-info
MODEL_INFO=$(curl -s http://localhost:5001/model-info)
if echo "$MODEL_INFO" | grep -q "ticker"; then
    echo -e "${GREEN}✓${NC} Model info OK"
else
    echo -e "${RED}✗${NC} Model info FALHOU"
    docker logs test-backend
    docker stop test-backend && docker rm test-backend
    exit 1
fi

# Testar predição
PREDICT_RESPONSE=$(curl -s -X POST http://localhost:5001/predict \
    -H "Content-Type: application/json" \
    -d '{"ticker": "PETR4.SA", "periods": 7}')

if echo "$PREDICT_RESPONSE" | grep -q "predictions"; then
    echo -e "${GREEN}✓${NC} Predição OK"
else
    echo -e "${RED}✗${NC} Predição FALHOU"
    echo "Response: $PREDICT_RESPONSE"
    docker logs test-backend
    docker stop test-backend && docker rm test-backend
    exit 1
fi

# Limpar
docker stop test-backend > /dev/null 2>&1
docker rm test-backend > /dev/null 2>&1
echo ""

# ==================== 4. TESTAR BUILD FRONTEND ====================
echo "🎨 4. Testando build do Frontend Docker..."
cd frontend
docker build -t stock-api-frontend-test . > /dev/null 2>&1
print_status "Frontend Docker build OK"

# Testar se frontend roda
docker stop test-frontend 2>/dev/null || true
docker rm test-frontend 2>/dev/null || true

docker run -d -p 3000:80 --name test-frontend stock-api-frontend-test > /dev/null
sleep 3

FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "$FRONTEND_RESPONSE" = "200" ]; then
    echo -e "${GREEN}✓${NC} Frontend HTTP 200 OK"
else
    echo -e "${RED}✗${NC} Frontend retornou HTTP $FRONTEND_RESPONSE"
    docker logs test-frontend
    docker stop test-frontend && docker rm test-frontend
    cd ..
    exit 1
fi

docker stop test-frontend > /dev/null 2>&1
docker rm test-frontend > /dev/null 2>&1
cd ..
echo ""

# ==================== 5. VERIFICAR WORKFLOWS ====================
echo "⚙️  5. Verificando GitHub Actions workflows..."

if [ ! -f ".github/workflows/train-weekly.yml" ]; then
    echo -e "${RED}✗${NC} train-weekly.yml não encontrado"
    exit 1
fi

if [ ! -f ".github/workflows/deploy-gcloud.yml" ]; then
    echo -e "${RED}✗${NC} deploy-gcloud.yml não encontrado"
    exit 1
fi

echo -e "${GREEN}✓${NC} Workflows encontrados"
echo ""

# ==================== 6. VERIFICAR SECRETS (lembrete) ====================
echo "🔐 6. Verificando configuração..."
echo ""
echo "   ${YELLOW}⚠${NC}  LEMBRE-SE: Antes de mergear para master, verifique:"
echo "      □ GitHub Secret: GCP_PROJECT_ID configurado"
echo "      □ GitHub Secret: GCP_SA_KEY configurado"
echo "      □ GCloud setup executado (./scripts/setup_gcloud.sh)"
echo ""

# ==================== RESULTADO ====================
echo "=============================="
echo -e "${GREEN}✅ TODOS OS TESTES PASSARAM!${NC}"
echo ""
echo "Próximos passos:"
echo "  1. git add -A"
echo "  2. git commit -m 'feat: implementa deploy GCloud'"
echo "  3. git push origin feat/integration"
echo "  4. Criar Pull Request para master"
echo "  5. Após merge, o deploy automático será executado"
echo ""
echo "Ou execute treino/deploy manual via GitHub Actions UI"
echo "=============================="
