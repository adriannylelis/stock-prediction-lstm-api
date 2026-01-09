#!/bin/bash
# Start MLflow UI with correct tracking URI
# Usage: ./scripts/init_mlflow.sh [port]

PORT=${1:-5001}
TRACKING_URI="file:data/mlflow/tracking"

echo ""
echo "🚀 Starting MLflow UI..."
echo "📂 Tracking URI: $TRACKING_URI"
echo "🌐 Server: http://127.0.0.1:$PORT"
echo ""
echo "Press CTRL+C to stop"
echo ""

# Set environment variable and start MLflow
export MLFLOW_TRACKING_URI="$TRACKING_URI"
mlflow ui --port "$PORT" --backend-store-uri "$TRACKING_URI"
