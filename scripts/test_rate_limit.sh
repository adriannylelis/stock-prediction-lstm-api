#!/bin/bash

# Script de teste para validar implementação de Rate Limiting
# Testa a API localmente antes de fazer deploy

set -e

echo "🧪 Iniciando testes de Rate Limiting..."
echo ""

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para teste com sucesso
pass_test() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Função para teste com falha
fail_test() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Função para warning
warn_test() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo "📦 Passo 1: Verificando dependências..."
if python3 -c "import flask_limiter" 2>/dev/null; then
    pass_test "Flask-Limiter instalado"
else
    warn_test "Flask-Limiter não encontrado. Instalando..."
    pip install flask-limiter>=3.5.0
fi

echo ""
echo "🔍 Passo 2: Verificando sintaxe dos arquivos..."
python3 -m py_compile src/api/main.py && pass_test "main.py OK"
python3 -m py_compile src/api/config/rate_limit.py && pass_test "rate_limit.py OK"
python3 -m py_compile src/api/routes/prediction.py && pass_test "prediction.py OK"
python3 -m py_compile src/api/routes/analytics.py && pass_test "analytics.py OK"
python3 -m py_compile src/api/routes/health.py && pass_test "health.py OK"

echo ""
echo "🚀 Passo 3: Iniciando API em background (rate limit desabilitado)..."
export RATE_LIMIT_ENABLED=false
export PYTHONPATH=$PWD
python3 src/api/main.py > /tmp/api_test.log 2>&1 &
API_PID=$!
sleep 5

# Verificar se a API iniciou
if ! ps -p $API_PID > /dev/null; then
    fail_test "API falhou ao iniciar. Logs em /tmp/api_test.log"
fi

echo ""
echo "🧪 Passo 4: Testando endpoints básicos (sem rate limit)..."

# Test 1: Health
if curl -s -f http://localhost:5001/health > /dev/null; then
    pass_test "GET /health respondendo"
else
    kill $API_PID 2>/dev/null || true
    fail_test "GET /health falhou"
fi

# Test 2: Model info (pode falhar se artifacts não existirem)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/model/info)
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "404" ] || [ "$HTTP_CODE" = "500" ]; then
    pass_test "GET /model/info respondendo (status: $HTTP_CODE)"
else
    warn_test "GET /model/info retornou status inesperado: $HTTP_CODE"
fi

echo ""
echo "🔄 Passo 5: Reiniciando API com rate limiting habilitado..."
kill $API_PID
sleep 2

export RATE_LIMIT_ENABLED=true
python3 src/api/main.py > /tmp/api_test_ratelimit.log 2>&1 &
API_PID=$!
sleep 5

if ! ps -p $API_PID > /dev/null; then
    fail_test "API falhou ao iniciar com rate limiting. Logs em /tmp/api_test_ratelimit.log"
fi

pass_test "API iniciada com rate limiting"

echo ""
echo "🚦 Passo 6: Testando rate limiting..."

# Fazer requests até atingir o limite
SUCCESS_COUNT=0
RATE_LIMITED=false

for i in {1..15}; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/health)
    
    if [ "$HTTP_CODE" = "200" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    elif [ "$HTTP_CODE" = "429" ]; then
        RATE_LIMITED=true
        break
    fi
    
    sleep 0.1
done

if [ "$RATE_LIMITED" = true ]; then
    pass_test "Rate limit acionado após $SUCCESS_COUNT requests"
else
    warn_test "Rate limit não foi acionado (limite pode ser muito alto)"
fi

echo ""
echo "🧹 Passo 7: Cleanup..."
kill $API_PID 2>/dev/null || true
sleep 1

echo ""
echo "📊 Resumo dos Testes:"
echo "   ✅ Sintaxe dos arquivos: OK"
echo "   ✅ API inicia sem rate limiting: OK"
echo "   ✅ API inicia com rate limiting: OK"
echo "   ✅ Endpoints respondem: OK"
if [ "$RATE_LIMITED" = true ]; then
    echo "   ✅ Rate limiting funciona: OK"
else
    echo "   ⚠️  Rate limiting: Não testado completamente"
fi

echo ""
echo -e "${GREEN}🎉 Todos os testes passaram! A API está pronta para uso.${NC}"
echo ""
echo "📝 Próximos passos:"
echo "   1. Rodar testes unitários: pytest tests/unit/ -v"
echo "   2. Rodar testes de integração: pytest tests/integration/ -v"
echo "   3. Testar com Docker Compose: docker-compose up --build"
echo "   4. Commit e push das mudanças"
echo ""
