# Stock Prediction LSTM API 📈

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Flask 3.1+](https://img.shields.io/badge/Flask-3.1+-000000.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/tests-92%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-72.79%25-yellow.svg)](tests/)
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

## 🎯 Visão Geral

Este projeto implementa uma solução end-to-end para previsão de preços de ações:

- ✅ **Pipeline de Dados**: Ingestão automática (Yahoo Finance), feature engineering (14 indicadores técnicos)
- ✅ **Modelo LSTM**: PyTorch multi-camadas com dropout, early stopping
- ✅ **Treinamento**: MLflow tracking, Optuna tuning, métricas completas (MAE, RMSE, MAPE, R², Directional Accuracy)
- ✅ **Monitoramento**: Drift detection (KS-test, PSI), data versioning, artifact management
- ✅ **CLI**: 5 comandos (train, predict, tune, drift, pipeline)
- ✅ **API REST**: Flask API com 3 endpoints (health, model/info, predict), tratamento de erros HTTP robusto
- ✅ **Qualidade**: 92 testes (100% passando), 72.79% coverage, Ruff linter

---

## 🏗️ Arquitetura

```
stock-prediction-lstm-api/
├── src/ml/                      # Core ML components
│   ├── data/                    # Data pipeline
│   │   ├── ingestion.py         # Yahoo Finance integration
│   │   ├── feature_engineering.py  # Technical indicators (SMA, RSI, MACD, etc.)
│   │   └── preprocessing.py     # Normalization & sequences
│   ├── models/
│   │   └── lstm.py              # PyTorch LSTM model
│   ├── training/
│   │   ├── trainer.py           # Training loop & checkpoints
│   │   ├── early_stopping.py    # Callback
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── hyperparameter_tuner.py  # Optuna
│   │   └── experiment_tracker.py    # MLflow
│   ├── pipeline/
│   │   ├── train_pipeline.py    # End-to-end training
│   │   └── predict_pipeline.py  # End-to-end prediction
│   ├── monitoring/
│   │   └── drift_detector.py    # Drift detection (KS-test, PSI)
│   └── utils/
│       ├── persistence.py       # Data versioning & artifacts
│       ├── device.py            # CPU/GPU management
│       ├── logging.py           # Structured logging
│       └── seed.py              # Reproducibility
├── cli/
│   └── main.py                  # CLI interface (5 commands)
├── src/api/                     # REST API (Flask)
│   ├── main.py                  # Application factory
│   ├── routes/                  # API endpoints
│   │   ├── health.py            # GET /health
│   │   ├── model_info.py        # GET /model/info
│   │   └── prediction.py        # POST /predict
│   ├── services/                # Business logic
│   │   ├── model_service.py     # Model loading (singleton)
│   │   ├── data_service.py      # Data fetching (yfinance)
│   │   └── predict_service.py   # Prediction pipeline
│   ├── models/
│   │   └── lstm_model.py        # LSTM model class
│   └── utils/
│       └── validators.py        # Input validation
├── tests/                       # 83 tests (100% passing)
│   ├── integration/             # 8 integration tests
│   ├── unit/                    # 75+ unit tests
│   └── test_*.py
├── notebooks/
│   └── eda.ipynb                # Exploratory analysis
├── artifacts/                   # Models, scalers, configs
├── data/                        # Raw & processed data
├── models/                      # Trained models
└── docs/                        # Documentation
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

## 🚀 Quick Start

### **🐳 Opção 1: Docker (Recomendado)**

A forma mais rápida de iniciar o projeto completo (backend + frontend):

```bash
# 1. Clone o repositório
git clone https://github.com/adriannylelis/stock-prediction-lstm-api.git
cd stock-prediction-lstm-api

# 2. Inicie os containers
docker-compose up --build

# 3. Acesse as aplicações
# Frontend: http://localhost:3000
# Backend API: http://localhost:5001/health
```

**O que está rodando:**
- ✅ **Frontend**: React dashboard em http://localhost:3000
- ✅ **Backend**: Flask API em http://localhost:5001
- ✅ **Healthchecks**: Automáticos para ambos containers
- ✅ **Network**: Comunicação interna entre serviços

**Comandos úteis:**
```bash
# Ver logs em tempo real
docker-compose logs -f

# Ver logs só do frontend
docker-compose logs -f frontend

# Ver logs só do backend
docker-compose logs -f backend

# Parar containers
docker-compose down

# Parar e remover volumes
docker-compose down -v
```

---

### **💻 Opção 2: Desenvolvimento Local**

#### **Backend (API)**

**Pré-requisitos:**
- Python 3.11+
- pip ou uv

**Setup:**
```bash
# 1. Clone o repositório
git clone https://github.com/adriannylelis/stock-prediction-lstm-api.git
cd stock-prediction-lstm-api

# 2. Crie ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Instale dependências
pip install -r requirements.txt

# 4. Execute a API
python -m src.api.main

# API disponível em http://localhost:5001
```

**Scripts automatizados (alternativa):**

Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

Windows:
```powershell
.\setup.ps1
```

#### **Frontend (React Dashboard)**

**Pré-requisitos:**
- Node.js 18+
- npm ou yarn

**Setup:**
```bash
# 1. Entre na pasta do frontend
cd frontend

# 2. Instale dependências
npm install

# 3. Configure variável de ambiente (opcional)
# Crie arquivo .env com:
VITE_API_URL=http://localhost:5001

# 4. Inicie o servidor de desenvolvimento
npm run dev

# Frontend disponível em http://localhost:5173
```

**Build para produção:**
```bash
npm run build
npm run preview  # Testa build localmente
```

---

## 📦 Instalação Completa (Manual)

Se preferir configurar tudo manualmente:

#### **1. Clone o Repositório**
```bash
git clone https://github.com/adriannylelis/stock-prediction-lstm-api.git
cd stock-prediction-lstm-api
```

#### **2. Crie o Ambiente Virtual (Backend)**
```bash
# Com venv
python -m venv venv

# Ativar no Windows
venv\Scripts\activate

# Ativar no Linux/Mac
source venv/bin/activate

# Ou com uv (recomendado)
uv venv
uv sync
```

#### **3. Instale as Dependências**
```bash
# Com pip
pip install -e .

# Ou com uv
uv pip install -e .

# Dependências de desenvolvimento (opcional)
pip install -e ".[dev]"

# API REST (opcional)
pip install -e ".[api]"
```

#### **4. Setup Frontend**
```bash
cd frontend
npm install
cd ..
```

#### **5. Crie Diretórios Necessários**
```bash
mkdir -p data/raw data/processed data/versioned
mkdir -p models artifacts logs
```

#### **6. Verifique a Instalação**
```bash
stock-predict --help
```

---

## 📖 Guia de Uso

### **🖥️ Dashboard Web (Interface Gráfica)**

A forma mais fácil de usar o sistema:

```bash
# Com Docker
docker-compose up

# Acesse http://localhost:3000
```

**Recursos do Dashboard:**
- 📊 **Seleção de ações**: 10 tickers disponíveis (7 US + 3 BR)
- 📈 **Gráfico interativo**: Recharts com previsões de 7 dias
- 💰 **Métricas**: Preço atual, previsão D+7, % variação, tendência
- ⚡ **Real-time**: Loading states e atualização automática
- 🎨 **Design moderno**: shadcn/ui + Tailwind CSS
- 📱 **Responsivo**: Mobile, tablet e desktop

**Ações disponíveis:**
- 🇺🇸 US Stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META
- 🇧🇷 BR Stocks: PETR4.SA, VALE3.SA, ITUB4.SA

---

### **🔌 API REST**

Integre com suas aplicações via HTTP:

#### **Endpoints**

**1. Health Check**
```bash
curl http://localhost:5001/health

# Response:
{
  "status": "healthy",
  "timestamp": "2026-01-06T19:00:00Z"
}
```

**2. Model Info**
```bash
curl http://localhost:5001/model/info

# Response:
{
  "model_version": "1.0.0",
  "architecture": "LSTM-1x16",
  "input_size": 1,
  "hidden_size": 16,
  "num_layers": 1,
  "dropout": 0.2
}
```

**3. Stock Prediction**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Response:
{
  "ticker": "AAPL",
  "current_price": 227.48,
  "predictions": [228.15, 228.82, 229.49, 230.16, 230.83, 231.50, 232.17],
  "dates": ["2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13"],
  "prediction_date": "2026-01-06",
  "days_ahead": 7,
  "trend": "bullish",
  "confidence": "high",
  "model_version": "1.0.0"
}
```

**Códigos de erro:**
- `400`: Ticker inválido ou ausente
- `404`: Ticker não encontrado no Yahoo Finance
- `500`: Erro interno do modelo
- `503`: Serviço indisponível (dados insuficientes)

---

### **⌨️ CLI Commands**

Para uso avançado e treinamento de modelos:

#### **Quick Start: Treinar e Prever**

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
# Desenvolvimento (com hot reload)
PYTHONPATH=$PWD python src/api/main.py

# Servidor roda em http://localhost:5001
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
