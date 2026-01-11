#!/bin/bash

###############################################################################
# Script: Test API Locally
# Description: Testa a API localmente com Docker
# Usage: ./scripts/test_api_local.sh
###############################################################################

set -e

echo "🧪 Testing API locally..."
echo ""

# Verificar se tem artifacts
if [ ! -f "artifacts/model.pt" ]; then
    echo "⚠️  No artifacts found. Training first..."
    ./scripts/local_train.sh
fi

echo "🐳 Building Docker image..."
docker build -t stock-api:local --build-arg DOWNLOAD_ARTIFACTS=false .

echo ""
echo "🚀 Starting container..."
docker run -d \
    --name stock-api-test \
    -p 5001:5001 \
    -v "$(pwd)/artifacts:/app/artifacts:ro" \
    stock-api:local

echo ""
echo "⏳ Waiting for API to be ready..."
sleep 10

# Health check
echo ""
echo "🏥 Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:5001/health)
echo "Response: $HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    docker logs stock-api-test
    docker stop stock-api-test
    docker rm stock-api-test
    exit 1
fi

# Test prediction
echo ""
echo "🔮 Testing prediction endpoint..."
PRED_RESPONSE=$(curl -s -X POST http://localhost:5001/predict \
    -H "Content-Type: application/json" \
    -d '{"ticker": "PETR4.SA", "periods": 7}')

echo "Response:"
echo "$PRED_RESPONSE" | jq .

if echo "$PRED_RESPONSE" | grep -q "predictions"; then
    echo "✅ Prediction test passed!"
else
    echo "❌ Prediction test failed!"
    docker logs stock-api-test
    docker stop stock-api-test
    docker rm stock-api-test
    exit 1
fi

# Model info
echo ""
echo "ℹ️  Testing model-info endpoint..."
INFO_RESPONSE=$(curl -s http://localhost:5001/model-info)
echo "$INFO_RESPONSE" | jq .

echo ""
echo "✅ All tests passed!"
echo ""
echo "🎯 API is running at: http://localhost:5001"
echo ""
echo "📝 Available endpoints:"
echo "   - GET  http://localhost:5001/health"
echo "   - GET  http://localhost:5001/model-info"
echo "   - POST http://localhost:5001/predict"
echo ""
echo "🛑 To stop the container:"
echo "   docker stop stock-api-test && docker rm stock-api-test"
