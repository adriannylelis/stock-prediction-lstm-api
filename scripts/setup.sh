#!/bin/bash
# Setup script for Stock Prediction LSTM API
# Linux/Mac version
# Usage: cd to project root, then run: ./scripts/setup.sh

set -e  # Exit on error

# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stock Prediction LSTM API - Setup Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📂 Diretório do projeto: $PROJECT_ROOT"
echo ""

# Check Python version
echo "🔍 Verificando versão do Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado! Instale Python 3.8+ primeiro."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python $PYTHON_VERSION detectado"
echo ""

# Create virtual environment
echo "📦 Criando ambiente virtual (.venv)..."
if [ -d ".venv" ]; then
    echo "✓ Ambiente virtual já existe. Pulando criação..."
else
    python3 -m venv .venv
    echo "✓ Ambiente virtual criado"
fi
echo ""

# Activate virtual environment
echo "🔌 Ativando ambiente virtual..."
source .venv/bin/activate
echo "✓ Ambiente ativado"
echo ""

# Upgrade pip
echo "⬆️  Atualizando pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip atualizado"
echo ""

# Install dependencies
echo "📥 Instalando dependências..."
echo "   (Isso pode levar alguns minutos...)"
pip install -e . > /dev/null 2>&1
echo "✓ Dependências instaladas"
echo ""

# Install dev dependencies
echo "🛠️  Instalando dependências de desenvolvimento..."
pip install pytest pytest-cov ruff > /dev/null 2>&1
echo "✓ Dependências de dev instaladas"
echo ""

# Create necessary directories
echo "📁 Criando diretórios necessários..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/versioned
mkdir -p models
mkdir -p artifacts
mkdir -p logs
echo "✓ Diretórios criados"
echo ""

# Verify installation
echo "✅ Verificando instalação..."
if command -v stock-predict &> /dev/null; then
    echo "✓ CLI instalado corretamente"
    stock-predict --help > /dev/null 2>&1
else
    echo "⚠️  Aviso: CLI pode não estar no PATH ainda"
    echo "   Execute: source .venv/bin/activate"
fi
echo ""

# Run quick test
echo "🧪 Executando teste rápido..."
if python -c "import torch, pandas, numpy, sklearn, yfinance; print('✓ Imports OK')" 2>/dev/null; then
    echo "✓ Todas as bibliotecas principais importadas com sucesso"
else
    echo "⚠️  Alguns imports falharam (pode ser normal se GPU não disponível)"
fi
echo ""

# Final instructions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Setup Completo!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📌 Próximos passos:"
echo ""
echo "1. Ative o ambiente virtual:"
echo "   $ source .venv/bin/activate"
echo ""
echo "2. Verifique a instalação:"
echo "   $ stock-predict --help"
echo ""
echo "3. Execute os testes:"
echo "   $ pytest tests/ -v"
echo ""
echo "4. Treine um modelo:"
echo "   $ stock-predict train --ticker PETR4.SA --start-date 2023-01-01 --end-date 2024-01-01"
echo ""
echo "5. Faça predições:"
echo "   $ stock-predict predict --model-path models/best_model.pt --ticker PETR4.SA --days-ahead 5"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📚 Documentação: docs/"
echo "🐛 Issues: https://github.com/adriannylelis/stock-prediction-lstm-api/issues"
echo ""
echo "💡 Dica: Execute este script a partir da raiz do projeto:"
echo "   $ ./scripts/setup.sh"
echo ""
