#!/bin/bash

###############################################################################
# Script: Validate Artifacts
# Description: Valida que todos os artifacts necessários estão presentes
# Usage: ./scripts/validate_artifacts.sh
###############################################################################

set -e

echo "🔍 Validating artifacts..."
echo ""

# Verificar se pasta existe
if [ ! -d "artifacts" ]; then
    echo "❌ ERROR: artifacts/ directory not found"
    exit 1
fi

# Lista de arquivos obrigatórios
REQUIRED_FILES=(
    "artifacts/model.pt"
    "artifacts/scaler.pkl"
)

# Lista de arquivos opcionais
OPTIONAL_FILES=(
    "artifacts/y_scaler.pkl"
    "artifacts/ticker_mapping.json"
    "artifacts/metrics.json"
    "artifacts/training_history.json"
)

# Validar arquivos obrigatórios
echo "📋 Checking required files..."
ALL_GOOD=true

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "   ✅ $file ($SIZE)"
    else
        echo "   ❌ $file - NOT FOUND"
        ALL_GOOD=false
    fi
done

echo ""
echo "📋 Checking optional files..."

for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "   ✅ $file ($SIZE)"
    else
        echo "   ⚪ $file - not present (optional)"
    fi
done

echo ""

if [ "$ALL_GOOD" = true ]; then
    echo "✅ All required artifacts are present!"
    echo ""
    echo "📊 Total size:"
    du -sh artifacts/
    echo ""
    echo "🚀 Ready to deploy!"
    exit 0
else
    echo "❌ Some required artifacts are missing!"
    echo ""
    echo "💡 To generate artifacts, run:"
    echo "   ./scripts/local_train.sh"
    exit 1
fi
