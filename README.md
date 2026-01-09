# Stock Prediction LSTM API 📈

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Flask 3.1+](https://img.shields.io/badge/Flask-3.1+-000000.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/tests-63%20total-blue.svg)](tests/)
[![Unit Tests](https://img.shields.io/badge/unit-53%2F53%20✓-brightgreen.svg)](tests/unit/)
[![Integration](https://img.shields.io/badge/integration-5%2F6%20✓-green.svg)](tests/integration/)
[![E2E](https://img.shields.io/badge/e2e-5%2F7%20✓-yellow.svg)](tests/e2e/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Sistema completo de **ML Engineering** para previsão de preços de ações usando LSTM (Long Short-Term Memory), com foco em boas práticas de engenharia, monitoramento, versionamento e qualidade de código.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Funcionalidades](#-funcionalidades)
- [Instalação](#-instalação)
- [Guia de Uso](#-guia-de-uso)
- [CLI Commands](#-cli-commands)
- [API REST](#-api-rest)
- [Testes](#-testes)
- [Documentação](#-documentação)
- [Tecnologias](#-tecnologias)

---

## 🚀 Quick Start

```bash
# 1. Treinar modelo de produção (36 tickers B3)
python -m cli train --use-all-tickers --epochs 25 --batch-size 64

# 2. Visualizar no MLflow UI
.\scripts\init_mlflow.ps1  # Windows
# ou
./scripts/init_mlflow.sh   # Linux/Mac
# Acesse: http://127.0.0.1:5001

# 3. Promover para Production
python promote_to_production.py

# 4. Testar predição
python -m src.api.main  # Inicia API na porta 5000
# Em outro terminal:
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'
```

**📊 Para análise detalhada**, veja [RELATORIO_PROJETO.md](RELATORIO_PROJETO.md)

---

## 🎯 Visão Geral

Este projeto implementa um sistema completo de **previsão de preços de ações B3** com arquitetura **MLOps moderna**:

- ✅ **MLflow-First**: Modelos, scalers, configs e métricas no MLflow (source of truth)
- ✅ **Multi-Ticker**: Suporte a **36 tickers B3 válidos** com ticker embeddings de 8 dimensões
- ✅ **Pipeline de Dados**: Ingestão automática (Yahoo Finance), **19 features técnicas** (OHLCV + 14 indicadores)
- ✅ **Modelo LSTM**: PyTorch com **ticker embeddings**, 3 camadas, dropout 0.3, early stopping
- ✅ **Treinamento**: MLflow tracking, model registry (Staging/Production), salvamento de artifacts
- ✅ **CLI Simplificado**: Comando `train` com suporte a multi-ticker (`--use-all-tickers`)
- ✅ **API REST**: Flask com 3 endpoints (health, model/info, predict), carregamento MLflow automático
- ✅ **Qualidade**: **63 testes** (58 passing - 92.06%), shapes validados, correções críticas aplicadas

**📊 Relatório Técnico Completo**: Veja [RELATORIO_PROJETO.md](RELATORIO_PROJETO.md) para análise detalhada de:
- Arquitetura e shapes (19 features + 8 embedding = 27 input total)
- Como funciona o ticker embedding
- Tickers suportados (36 válidos, 7 removidos por falta de dados)
- Correções críticas (Y normalization, batch consistency, deduplication)
- Dimensionalidade da rede (~342k parâmetros)

---

## 🏗️ Arquitetura

### **MLflow-First Architecture** 🎯

O projeto segue uma arquitetura **MLflow-first**, onde **MLflow é a fonte da verdade** para:
- ✅ Modelos (versionados e registrados)
- ✅ Scalers (artifacts)
- ✅ Configurações (params)
- ✅ Métricas (tracking)
- ✅ Experimentos (runs)

```
stock-prediction-lstm-api/
├── src/                         # Código fonte principal
│   ├── ml/                      # Core ML components
│   │   ├── data/                # Data pipeline
│   │   │   ├── ingestion.py         # Yahoo Finance integration
│   │   │   ├── feature_engineering.py  # 19 technical indicators
│   │   │   └── preprocessing.py     # Normalization & sequences (PyTorch)
│   │   ├── models/
│   │   │   └── lstm.py              # PyTorch LSTM with Ticker Embeddings
│   │   ├── training/
│   │   │   ├── trainer.py           # Training loop + MLflow tracking
│   │   │   ├── early_stopping.py    # Callback with min_delta
│   │   │   ├── metrics.py           # MAE, RMSE, MAPE, R², DA
│   │   │   ├── hyperparameter_tuner.py  # Optuna integration
│   │   │   └── experiment_tracker.py    # MLflow wrapper
│   │   ├── pipeline/
│   │   │   ├── train_pipeline.py    # Single + Multi-ticker training
│   │   │   └── predict_pipeline.py  # MLflow-based predictions
│   │   └── utils/
│   │       ├── persistence.py       # Data versioning (DataVersionManager)
│   │       ├── device.py            # CPU/CUDA/MPS detection
│   │       ├── logging.py           # Loguru structured logging
│   │       └── seed.py              # Reproducibility (torch + numpy)
│   ├── api/                     # REST API (Flask)
│   │   ├── main.py                  # Application factory
│   │   ├── routes/                  # API endpoints
│   │   │   ├── health.py            # GET /health
│   │   │   ├── model_info.py        # GET /model/info
│   │   │   └── prediction.py        # POST /predict
│   │   └── services/                # Business logic
│   │       ├── model_service.py     # MLflow model loader (singleton)
│   │       ├── data_service.py      # yfinance integration
│   │       └── predict_service.py   # Prediction orchestration
│   └── mlops/                   # MLOps automation
│       ├── pipelines/               # Automation pipelines
│       │   ├── training_pipeline.py     # Auto-training (43 tickers)
│       │   └── promotion_pipeline.py    # Staging → Production
│       ├── monitoring/
│       │   └── model_comparator.py      # Metric-based comparison
│       └── deployment/
│           └── model_deployer.py        # Production deployment
├── cli/
│   ├── main.py                  # CLI entry point
│   └── train.py                 # Train command (130 lines)
├── tests/                       # 📊 63 testes automatizados
│   ├── unit/                    # 53 testes (100% passando)
│   │   ├── test_metrics.py          # 7 testes
│   │   ├── test_model.py            # 5 testes
│   │   ├── test_model_deployer.py   # 10 testes
│   │   ├── test_model_service.py    # 11 testes
│   │   ├── test_persistence.py      # 15 testes
│   │   └── test_preprocessing.py    # 7 testes
│   ├── integration/             # 6 testes (5 passando, 1 skipped)
│   │   ├── test_full_pipeline.py    # 3 testes
│   │   └── test_pipelines.py        # 3 testes
│   └── e2e/                     # 7 testes (5 passando, 1 failed, 1 skipped)
│       └── test_mlops_complete.py   # Complete MLOps workflow
├── configs/                     # Configurações
│   └── production_model.yaml        # Production model config
├── data/                        # MLflow-first data structure
│   ├── mlflow/
│   │   └── tracking/                # 📦 MLflow tracking store
│   └── versioned/                   # Data versioning (tests only)
├── artifacts/                   # Local artifacts (fallback)
│   └── models/                      # Checkpoints locais (keep 3)
├── notebooks/
│   └── eda.ipynb                # Exploratory analysis
├── docs/                        # Documentação
│   ├── TESTING_PLAN.md              # Plano de testes
│   └── QUICK_START.md               # Guia rápido
├── requirements.txt             # PyTorch, MLflow, Flask, yfinance
├── requirements-dev.txt         # pytest, ruff, ipython
└── pyproject.toml               # Project configuration
```

---

## ⚡ Funcionalidades

### **Pipeline de Dados**
- Download automático de dados históricos (Yahoo Finance)
- 14 indicadores técnicos: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, etc.
- Normalização MinMaxScaler com persistência
- Criação de sequências temporais configuráveis

### **Modelo & Treinamento**
- LSTM multi-camadas com dropout (PyTorch)
- Early stopping para prevenir overfitting
- MLflow para tracking de experimentos
- Optuna para hyperparameter tuning
- Checkpoints completos (arquitetura + pesos)

### **Pipelines End-to-End**

**TrainPipeline** (5 etapas):
1. Data Ingestion → Download from Yahoo Finance
2. Feature Engineering → Add technical indicators
3. Preprocessing → Normalize & create sequences
4. Training → Train LSTM with validation
5. Evaluation → Calculate test metrics

**PredictPipeline** (4 etapas):
1. Data Ingestion → Download latest 2 years
2. Feature Engineering → Add indicators
3. Preprocessing → Prepare last sequence
4. Prediction → Multi-step forecasting

### **Monitoramento & Versionamento**
- Drift detection (Kolmogorov-Smirnov test, PSI)
- Data versioning com timestamps
- Artifact management (models, scalers, configs)
- Auto-cleanup de versões antigas

### **API REST**
- Flask Application Factory Pattern
- CORS habilitado para integração frontend
- 3 endpoints: health check, model info, predictions
- Singleton pattern para carregamento de modelo
- Validação de entrada e tratamento de erros
- Logging estruturado

---

## 🚀 Instalação

### **Pré-requisitos**
- Python 3.13+
- pip ou uv

### **Setup Rápido (Recomendado)**

Use o script automatizado para configurar o ambiente:

#### **Linux/Mac/Git Bash**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

O script irá:
- ✅ Detectar Python 3.13
- ✅ Criar ambiente virtual (.venv)
- ✅ Instalar PyTorch (escolha CPU ou GPU CUDA 12.4)
- ✅ Instalar dependências (requirements.txt + requirements-dev.txt)
- ✅ Instalar projeto em modo editable (pip install -e .)
- ✅ Criar diretórios necessários (data/, models/, artifacts/, logs/)
- ✅ Verificar instalação (torch, mlflow, flask)

---

### **Setup Manual (Alternativo)**

Se preferir configurar manualmente:

#### **1. Clone o Repositório**
```bash
git clone https://github.com/adriannylelis/stock-prediction-lstm-api.git
cd stock-prediction-lstm-api
```

#### **2. Crie o Ambiente Virtual**
```bash
# Com venv
python -m venv .venv

# Ativar no Windows
.venv\Scripts\activate

# Ativar no Linux/Mac
source .venv/bin/activate

# Ou com uv (recomendado)
uv venv
uv sync
```

#### **3. Instale as Dependências**
```bash
# PyTorch (escolha CPU ou GPU)
# CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 12.4):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Dependências principais
pip install -r requirements.txt

# Ferramentas de desenvolvimento (pytest, ruff, ipython)
pip install -r requirements-dev.txt

# Instalar projeto em modo editable
pip install -e .
```

#### **4. Crie Diretórios Necessários**
```bash
mkdir -p data/raw data/processed data/versioned
mkdir -p models artifacts logs
```

#### **5. Verifique a Instalação**
```bash
stock-predict --help
```

---

## � MLflow Tracking & Model Registry

O projeto usa **MLflow** como fonte da verdade para modelos, métricas e experimentos.

### **Iniciar MLflow UI**

```powershell
# Windows PowerShell (recomendado)
.\scripts\init_mlflow.ps1

# Linux/Mac
chmod +x scripts/init_mlflow.sh
./scripts/init_mlflow.sh

# Python (alternativa)
python -m mlflow_config

# Comando direto (funciona em qualquer plataforma)
$env:MLFLOW_TRACKING_URI="file:data/mlflow/tracking"; mlflow ui --port 5001 --backend-store-uri "file:data/mlflow/tracking"
```

Acesse: **http://127.0.0.1:5001**

### **Estrutura de Dados MLflow**

```
data/
├── mlflow/
│   └── tracking/           # 📦 MLflow tracking store (SQLITE)
│       ├── 0/              # Experimento Default
│       ├── 941569.../      # Experimento lstm-multi-ticker
│       │   ├── meta.yaml
│       │   ├── 9116f4c9.../   # Run 1
│       │   │   ├── artifacts/
│       │   │   │   ├── model/
│       │   │   │   ├── scaler.pkl
│       │   │   │   └── config.yaml
│       │   │   ├── metrics/
│       │   │   ├── params/
│       │   │   └── tags/
│       │   └── f7133de2.../   # Run 2
│       └── models/            # Model Registry
│           └── stock-lstm-model/
│               ├── version-1/
│               └── version-2/
```

### **Tracking URI**

O projeto está configurado para usar:
```
MLFLOW_TRACKING_URI = file:data/mlflow/tracking
```

Todos os componentes (CLI, API, pipelines) usam automaticamente esse URI.

---

## �📖 Guia de Uso

### **Quick Start: Treinar e Prever**

```bash
# 1. Treinar modelo para PETR4.SA (Petrobras)
stock-predict train \
  --ticker PETR4.SA \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --lookback 60 \
  --hidden-size 64 \
  --epochs 50 \
  --model-path models/petr4_model.pt

# 2. Fazer previsões para os próximos 5 dias
stock-predict predict \
  --model-path models/best_model.pt \
  --ticker PETR4.SA \
  --lookback 60 \
  --days-ahead 5

# 3. Detectar drift
stock-predict drift \
  --ticker PETR4.SA \
  --reference-version 20240101_120000 \
  --production-version 20240201_120000
```

---

## 🖥️ CLI Commands

### **1. Train - Treinar Modelo**

```bash
stock-predict train [OPTIONS]

Opções:
  --ticker TEXT          Ticker da ação (ex: PETR4.SA, VALE3.SA)  [required]
  --start-date TEXT      Data de início (YYYY-MM-DD)              [required]
  --end-date TEXT        Data de fim (YYYY-MM-DD)                 [required]
  --lookback INTEGER     Janela temporal (padrão: 60)
  --hidden-size INTEGER  Tamanho da camada LSTM (padrão: 64)
  --num-layers INTEGER   Número de camadas LSTM (padrão: 2)
  --dropout FLOAT        Dropout rate (padrão: 0.2)
  --epochs INTEGER       Número de épocas (padrão: 50)
  --batch-size INTEGER   Tamanho do batch (padrão: 32)
  --learning-rate FLOAT  Learning rate (padrão: 0.001)
  --model-path TEXT      Caminho para salvar modelo
  --experiment-name TEXT Nome do experimento MLflow

Exemplo:
  stock-predict train --ticker PETR4.SA --start-date 2023-01-01 \
    --end-date 2024-01-01 --lookback 60 --epochs 50
```

**Output:**
```
✓ Data ingestion complete: 248 records
✓ Features engineered: 19 features
✓ Data preprocessed: 218 sequences
✓ Training complete: 50 epochs
✓ Test Metrics:
  - RMSE: 0.1234
  - MAE: 0.0987
  - MAPE: 3.45%
  - R²: 0.8765
  - Directional Accuracy: 65.43%
✓ Model saved: models/best_model.pt
```

---

### **2. Predict - Fazer Previsões**

```bash
stock-predict predict [OPTIONS]

Opções:
  --model-path TEXT      Caminho do modelo treinado  [required]
  --ticker TEXT          Ticker da ação              [required]
  --lookback INTEGER     Janela temporal (padrão: 60)
  --days-ahead INTEGER   Dias para prever (padrão: 5)

Exemplo:
  stock-predict predict --model-path models/best_model.pt \
    --ticker PETR4.SA --days-ahead 5
```

**Output:**
```
✓ Model loaded successfully
✓ Data ingested: 499 records (last 2 years)
✓ Predictions generated:

        Date  Predicted_Close
0 2025-12-27        29.95
1 2025-12-28        29.94
2 2025-12-29        29.94
3 2025-12-30        29.93
4 2025-12-31        29.93
```

---

### **3. Tune - Otimizar Hiperparâmetros**

```bash
stock-predict tune [OPTIONS]

Opções:
  --ticker TEXT         Ticker da ação              [required]
  --start-date TEXT     Data de início              [required]
  --end-date TEXT       Data de fim                 [required]
  --n-trials INTEGER    Número de trials Optuna (padrão: 20)
  --timeout INTEGER     Timeout em segundos (padrão: 3600)

Exemplo:
  stock-predict tune --ticker PETR4.SA --start-date 2023-01-01 \
    --end-date 2024-01-01 --n-trials 30
```

**Output:**
```
[I 2025-12-28 10:00:00,000] Trial 1: RMSE=0.1456
[I 2025-12-28 10:05:00,000] Trial 2: RMSE=0.1234  ← Best
[I 2025-12-28 10:10:00,000] Trial 3: RMSE=0.1389
...
✓ Best hyperparameters:
  - lookback: 60
  - hidden_size: 128
  - num_layers: 3
  - dropout: 0.3
  - learning_rate: 0.0005
✓ Best RMSE: 0.1234
```

---

### **4. Drift - Detectar Drift**

```bash
stock-predict drift [OPTIONS]

Opções:
  --ticker TEXT              Ticker da ação  [required]
  --reference-version TEXT   Versão de referência (timestamp)  [required]
  --production-version TEXT  Versão de produção (timestamp)    [required]

Exemplo:
  stock-predict drift --ticker PETR4.SA \
    --reference-version 20240101_120000 \
    --production-version 20240201_120000
```

**Output:**
```
✓ Drift Detection Report:
  - Has Drift: True
  - Drifted Features: ['Close', 'Volume']
  - Drift Scores:
    * Close: KS=0.1234, p-value=0.0012
    * Volume: KS=0.0987, p-value=0.0456
  - Recommendation: Retrain model
```

---

### **5. Pipeline - Executar Pipeline Completo**

```bash
stock-predict pipeline [OPTIONS]

Opções:
  --ticker TEXT     Ticker da ação              [required]
  --start-date TEXT Data de início              [required]
  --end-date TEXT   Data de fim                 [required]
  --days-ahead INT  Dias para prever (padrão: 5)

Exemplo:
  stock-predict pipeline --ticker PETR4.SA --start-date 2023-01-01 \
    --end-date 2024-01-01 --days-ahead 5
```

**Output:**
```
=== Training Pipeline ===
✓ Data ingestion: 248 records
✓ Training complete: RMSE=0.1234
✓ Model saved: models/best_model.pt

=== Prediction Pipeline ===
✓ Predictions generated for 5 days

        Date  Predicted_Close
0 2025-12-27        29.95
1 2025-12-28        29.94
...
```

---

## 🌐 API REST

### **Iniciar o Servidor**

```bash
# Usando script automatizado (recomendado)
./scripts/init_backend.sh

# Ou manualmente
export FLASK_APP=src.api.main:create_app
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000 --reload

# Servidor roda em http://localhost:5000
```

---

### **Endpoints Disponíveis**

#### **1. GET /health - Health Check**

Verifica se a API está funcionando.

```bash
curl http://localhost:5001/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-30T03:24:59.083016",
  "service": "stock-prediction-lstm-api"
}
```

---

#### **2. GET /model/info - Informações do Modelo**

Retorna metadados do modelo LSTM treinado.

```bash
curl http://localhost:5001/model/info
```

**Response (200 OK):**
```json
{
  "architecture": "LSTM-1x16",
  "input_size": 1,
  "hidden_size": 16,
  "num_layers": 1,
  "dropout": 0.0,
  "lookback": 60,
  "features": ["Close"],
  "metrics": {
    "mape": 1.21,
    "mae": 0.38,
    "rmse": 0.53,
    "r2": 0.90
  },
  "training": {
    "params": 1233,
    "train_samples": 996,
    "test_samples": 215
  }
}
```

---

#### **3. POST /predict - Fazer Previsão**

Realiza previsão de preço para um ticker.

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

**Request Body:**
```json
{
  "ticker": "AAPL"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "ticker": "AAPL",
    "predicted_price": 88.59,
    "current_price": 273.76,
    "change_percent": -67.64,
    "change_direction": "down",
    "prediction_date": "2025-12-31",
    "confidence": "low",
    "timestamp": "2025-12-30T04:18:19.245103"
  }
}
```

**Campos do Response:**
- `success`: Indica se a operação foi bem-sucedida
- `data`: Objeto com os dados da previsão
  - `ticker`: Ticker da ação
  - `predicted_price`: Preço previsto para o próximo dia
  - `current_price`: Último preço conhecido
  - `change_percent`: Variação percentual esperada
  - `change_direction`: Direção da mudança (up/down/neutral)
  - `prediction_date`: Data da previsão (T+1)
  - `confidence`: Nível de confiança (high/medium/low)
  - `timestamp`: Timestamp UTC da previsão

**Níveis de Confiança:**
- `high`: Mudança < 2%
- `medium`: Mudança entre 2% e 5%
- `low`: Mudança > 5%

---

### **Tratamento de Erros**

#### **400 - Bad Request (Ticker Inválido)**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "A"}'
```

**Response:**
```json
{
  "error": "InvalidTicker",
  "message": "Ticker 'A' é inválido ou não encontrado",
  "status": 400,
  "details": {
    "ticker": "A",
    "suggestion": "Ticker deve ter entre 2 e 10 caracteres"
  }
}
```

#### **400 - Missing Field**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response:**
```json
{
  "error": "Missing Field",
  "message": "Campo 'ticker' é obrigatório",
  "status": 400
}
```

#### **404 - Ticker Not Found**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "INVALID"}'
```

**Response:**
```json
{
  "error": "TickerNotFound",
  "message": "Ticker 'INVALID' não encontrado",
  "status": 404,
  "details": {
    "ticker": "INVALID",
    "suggestion": "Verifique se o ticker está correto. Exemplos: AAPL, PETR4.SA, VALE3.SA"
  }
}
```

#### **500 - Internal Server Error**
```json
{
  "error": "Internal Server Error",
  "message": "Erro interno do servidor",
  "status": 500
}
```

#### **503 - Service Unavailable**
```json
{
  "error": "ServiceUnavailable",
  "message": "Yahoo Finance indisponível no momento",
  "status": 503
}
```

---

### **Arquitetura da API**

```
src/api/
├── main.py                    # Application Factory
│   ├── create_app()           # Factory function
│   ├── register_blueprints()  # Route registration
│   └── register_error_handlers()  # Global error handling
├── routes/                    # Endpoints (Blueprints)
│   ├── health.py              # Health check
│   ├── model_info.py          # Model metadata
│   └── prediction.py          # Predictions
├── services/                  # Business Logic
│   ├── model_service.py       # Singleton: Model + Scaler loading
│   ├── data_service.py        # Yahoo Finance integration
│   └── predict_service.py     # Prediction pipeline orchestration
├── models/
│   └── lstm_model.py          # StockLSTM PyTorch model
└── utils/
    ├── validators.py          # Input validation (ticker format)
    └── exceptions.py          # Custom exceptions hierarchy
```

**Design Patterns:**
- **Application Factory**: Criação flexível da app Flask
- **Singleton**: ModelService carrega modelo apenas 1x
- **Blueprint**: Modularização de rotas
- **Service Layer**: Separação de lógica de negócio
- **Custom Exceptions**: Hierarquia de exceções com status codes HTTP apropriados

**Custom Exceptions:**
- `InvalidTickerError` (400): Formato de ticker inválido
- `InsufficientDataError` (400): Menos de 60 dias de dados disponíveis
- `TickerNotFoundError` (404): Ticker não existe no Yahoo Finance
- `ModelInferenceError` (500): Erro na inferência do modelo
- `ServiceUnavailableError` (503): Yahoo Finance indisponível

---

### **Exemplo de Uso Completo**

```bash
# 1. Iniciar API
PYTHONPATH=$PWD python src/api/main.py &

# 2. Verificar health
curl http://localhost:5001/health

# 3. Ver informações do modelo
curl http://localhost:5001/model/info | python -m json.tool

# 4. Fazer previsão para AAPL
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}' | python -m json.tool

# 5. Fazer previsão para ação brasileira
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}' | python -m json.tool
```

---

## 🧪 Testes

### **Executar Todos os Testes**

```bash
# Executar suite completa
pytest tests/

# Com cobertura
pytest tests/ --cov=src --cov-report=html

# Apenas integration tests
pytest tests/integration/

# Apenas unit tests
pytest tests/unit/

# Testes específicos
pytest tests/test_pipelines.py -v

# Quiet mode (apenas resumo)
pytest tests/ -q
```

### **Estatísticas de Testes**

```bash
# Última execução:
83 passed in 91.58s
Coverage: 72.79%
```

### **Estrutura de Testes**

| Categoria | Quantidade | Descrição |
|-----------|------------|-----------|
| **Integration** | 8 | Testes end-to-end (train, predict, drift) |
| **Unit** | 75+ | Testes unitários de componentes |
| **Pipeline** | 4 | Testes de pipelines completos |
| **Monitoring** | 31 | Testes de drift detection & versioning |

### **Testes End-to-End Principais**

- `test_train_pipeline_end_to_end`: Valida pipeline completo de treino
- `test_predict_pipeline_end_to_end`: Valida pipeline completo de predição
- `test_train_and_predict_integration`: Valida integração treino → predição
- `test_full_retraining_workflow`: Valida workflow com drift detection + retreino

**Ver detalhes**: [docs/TESTING_REPORT.md](docs/TESTING_REPORT.md)

---

## 📚 Documentação

| Documento | Descrição |
|-----------|-----------|
| [PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Relatório completo do projeto, arquitetura, funcionalidades e integração com API |
| [TESTING_REPORT.md](docs/TESTING_REPORT.md) | Relatório detalhado de testes end-to-end (treino, retreino, predição) |
| [MODEL_DOCUMENTATION.md](docs/MODEL_DOCUMENTATION.md) | Documentação do modelo LSTM, arquitetura e métricas |
| [API_SCHEMA_COMPLETO.md](docs/API_SCHEMA_COMPLETO.md) | Schema completo da API REST (proposta) |

---

## 🛠️ Tecnologias

| Categoria | Tecnologia | Versão |
|-----------|-----------|--------|
| **ML Framework** | PyTorch | 2.1+ |
| **Data Processing** | pandas, numpy | latest |
| **Data Source** | yfinance | latest |
| **Experiment Tracking** | MLflow | 2.9+ |
| **Hyperparameter Tuning** | Optuna | 3.5+ |
| **Testing** | pytest, pytest-cov | 8.0+, 7.0+ |
| **Code Quality** | Ruff | 0.1+ |
| **CLI** | Click | 8.1+ |
| **API Framework** | Flask, Flask-CORS | 3.0+, 6.0+ |
| **Logging** | Loguru | 0.7+ |

---

## 🔄 Workflow de Desenvolvimento

### **1. Desenvolvimento Local**

```bash
# Criar branch
git checkout -b feature/nova-funcionalidade

# Desenvolver
# ... código ...

# Rodar testes
pytest tests/ -v

# Formatar código
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Commit
git add .
git commit -m "feat: nova funcionalidade"
git push origin feature/nova-funcionalidade
```

### **2. Quality Checks**

```bash
# Ruff (substitui black, isort, flake8, mypy)
ruff check src/ tests/ --fix
ruff format src/ tests/

# Testes com coverage
pytest tests/ --cov=src --cov-report=term-missing

# Verificar tipos (opcional)
mypy src/ --ignore-missing-imports
```

---

## 📊 Métricas de Qualidade

| Métrica | Valor | Status |
|---------|-------|--------|
| **Testes** | 83/83 passing | ✅ |
| **Coverage** | 72.79% | ✅ |
| **Ruff Issues** | 0 | ✅ |
| **Type Hints** | ~80% | ⚠️ |

---

**Licença**: MIT  
