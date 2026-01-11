#!/bin/bash

# Script de validação completa do Backend
# Testa todos os endpoints da API

set -e

echo "========================================="
echo "🧪 VALIDAÇÃO COMPLETA DO BACKEND"
echo "========================================="
echo ""

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_URL="${API_URL:-http://localhost:5001}"

# Função para testar endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4
    
    echo -n "  ${description}... "
    
    if [ -z "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X ${method} "${API_URL}${endpoint}")
    else
        response=$(curl -s -w "\n%{http_code}" -X ${method} "${API_URL}${endpoint}" \
            -H "Content-Type: application/json" \
            -d "${data}")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" == "200" ] || [ "$http_code" == "201" ]; then
        echo -e "${GREEN}✓ OK (${http_code})${NC}"
        return 0
    else
        echo -e "${RED}✗ FALHOU (${http_code})${NC}"
        echo "  Resposta: $body"
        return 1
    fi
}

# Contador
total=0
passed=0
failed=0

echo "🔍 Testando endpoints..."
echo ""

# 1. Health Check
echo "1️⃣  Health Check"
if test_endpoint "GET" "/health" "" "GET /health"; then
    ((passed++))
else
    ((failed++))
fi
((total++))
echo ""

# 2. Model Info
echo "2️⃣  Model Info"
if test_endpoint "GET" "/model/info" "" "GET /model/info"; then
    ((passed++))
else
    ((failed++))
fi
((total++))
echo ""

# 3. Prediction - PETR4.SA 7 dias
echo "3️⃣  Prediction (PETR4.SA - 7 dias)"
payload='{"ticker": "PETR4.SA", "periods": 7}'
if test_endpoint "POST" "/predict" "$payload" "POST /predict (7 dias)"; then
    ((passed++))
    
    # Mostrar resultado
    response=$(curl -s -X POST "${API_URL}/predict" \
        -H "Content-Type: application/json" \
        -d "$payload")
    echo "  📊 Resultado:"
    echo "$response" | python3 -m json.tool | head -20
else
    ((failed++))
fi
((total++))
echo ""

# 4. Prediction - PETR4.SA 1 dia
echo "4️⃣  Prediction (PETR4.SA - 1 dia)"
payload='{"ticker": "PETR4.SA", "periods": 1}'
if test_endpoint "POST" "/predict" "$payload" "POST /predict (1 dia)"; then
    ((passed++))
else
    ((failed++))
fi
((total++))
echo ""

# 5. Prediction - Ticker inválido (deve falhar gracefully)
echo "5️⃣  Validation - Ticker inválido"
payload='{"ticker": "INVALID.SA", "periods": 7}'
response=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}/predict" \
    -H "Content-Type: application/json" \
    -d "$payload")
http_code=$(echo "$response" | tail -n1)

if [ "$http_code" == "400" ] || [ "$http_code" == "404" ]; then
    echo -e "  ${GREEN}✓ Validação OK (retornou ${http_code} como esperado)${NC}"
    ((passed++))
else
    echo -e "  ${RED}✗ Validação falhou (esperava 400/404, recebeu ${http_code})${NC}"
    ((failed++))
fi
((total++))
echo ""

# Resumo
echo "========================================="
echo "📊 RESUMO DOS TESTES"
echo "========================================="
echo "Total:   $total testes"
echo -e "Passou:  ${GREEN}$passed ✓${NC}"
echo -e "Falhou:  ${RED}$failed ✗${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✅ TODOS OS TESTES PASSARAM!${NC}"
    echo ""
    echo "🚀 Backend está pronto para deploy!"
    exit 0
else
    echo -e "${RED}❌ ALGUNS TESTES FALHARAM${NC}"
    echo ""
    echo "⚠️  Verifique os logs do container:"
    echo "   docker logs <container-name>"
    exit 1
fi
