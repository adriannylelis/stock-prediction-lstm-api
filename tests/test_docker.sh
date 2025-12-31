#!/bin/bash

# Script para testar container Docker da API
# Uso: ./test_docker.sh

set -e

echo "========================================================================"
echo "  TESTE DO CONTAINER DOCKER - Stock Prediction LSTM API"
echo "========================================================================"
echo ""

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CONTAINER_NAME="stock-api-test"
IMAGE_NAME="stock-prediction-api:latest"
PORT=5001

# 1. Verificar se container já está rodando
echo -e "${BLUE}[1/6]${NC} Verificando containers existentes..."
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo -e "${YELLOW}Container existente encontrado. Removendo...${NC}"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# 2. Iniciar container
echo -e "${BLUE}[2/6]${NC} Iniciando container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:$PORT \
    $IMAGE_NAME

# 3. Aguardar inicialização
echo -e "${BLUE}[3/6]${NC} Aguardando inicialização da API (15s)..."
sleep 15

# 4. Verificar se container está rodando
echo -e "${BLUE}[4/6]${NC} Verificando status do container..."
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo -e "${GREEN}✓ Container rodando${NC}"
else
    echo -e "${RED}✗ Container não está rodando${NC}"
    echo "Logs do container:"
    docker logs $CONTAINER_NAME
    exit 1
fi

# 5. Testar endpoints
echo -e "${BLUE}[5/6]${NC} Testando endpoints..."
echo ""

# Test 1: Health check
echo -e "${YELLOW}Test 1: GET /health${NC}"
response=$(curl -s http://localhost:$PORT/health)
if echo "$response" | grep -q "healthy"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "$response"
fi
echo ""

# Test 2: Model info
echo -e "${YELLOW}Test 2: GET /model/info${NC}"
response=$(curl -s http://localhost:$PORT/model/info)
if echo "$response" | grep -q "architecture"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "$response" | python3 -m json.tool | head -15
    echo "..."
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "$response"
fi
echo ""

# Test 3: Prediction
echo -e "${YELLOW}Test 3: POST /predict (AAPL)${NC}"
response=$(curl -s -X POST http://localhost:$PORT/predict \
    -H "Content-Type: application/json" \
    -d '{"ticker":"AAPL"}')
if echo "$response" | grep -q "predicted_price"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "$response"
fi
echo ""

# 6. Verificar healthcheck do Docker
echo -e "${BLUE}[6/6]${NC} Verificando Docker healthcheck..."
health_status=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "none")
if [ "$health_status" = "healthy" ] || [ "$health_status" = "starting" ]; then
    echo -e "${GREEN}✓ Healthcheck: $health_status${NC}"
else
    echo -e "${YELLOW}⚠ Healthcheck: $health_status (pode levar até 30s)${NC}"
fi

# Resumo
echo ""
echo "========================================================================"
echo "  RESUMO"
echo "========================================================================"
echo -e "Container Name: ${BLUE}$CONTAINER_NAME${NC}"
echo -e "Image: ${BLUE}$IMAGE_NAME${NC}"
echo -e "Port: ${BLUE}$PORT${NC}"
echo -e "Status: ${GREEN}RUNNING${NC}"
echo ""
echo "Comandos úteis:"
echo "  Ver logs:    docker logs $CONTAINER_NAME"
echo "  Parar:       docker stop $CONTAINER_NAME"
echo "  Remover:     docker rm $CONTAINER_NAME"
echo "  Shell:       docker exec -it $CONTAINER_NAME /bin/bash"
echo ""
echo -e "${YELLOW}Para parar e remover o container:${NC}"
echo "  docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
echo ""
