# 🚀 Stock Prediction LSTM API - PETR4.SA

**Previsão de preços de ações usando LSTM + Deploy automatizado na Google Cloud Platform**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Platform-yellow.svg)](https://cloud.google.com/)

API REST completa para previsão de preços de ações brasileiras usando **LSTM**, com foco em **PETR4.SA (Petrobras)**. Inclui frontend web, treino automatizado via GitHub Actions e deploy na Google Cloud Platform.

---

---

## 📋 Índice

- [⚡ Quick Start](#-quick-start)
- [🎯 Visão Geral](#-visão-geral)
- [🏗️ Arquitetura](#️-arquitetura)
- [💰 Custos](#-custos)
- [📚 Documentação](#-documentação)
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

**Ver detalhes**: [docs/RUN_TESTS.md](docs/RUN_TESTS.md)

---

## 📚 Documentação

| Documento | Descrição |
|-----------|-----------|
| [PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Relatório completo do projeto, arquitetura, funcionalidades e integração com API |
| [ML_DOCUMENTATION.md](docs/ML_DOCUMENTATION.md) | Documentação do modelo LSTM, pipelines de treino/predição e métricas |
| [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | Especificação técnica da API REST (endpoints, fluxo, exceções, validações) |
| [RUN_TESTS.md](docs/RUN_TESTS.md) | Como rodar a suíte de testes (unit, integration, e2e) |

### 🔁 Pipelines Principais

- **Pipeline de ML (dados → treino → predição):** descrito em [docs/ML_DOCUMENTATION.md](docs/ML_DOCUMENTATION.md); cobre `TrainPipeline`, `PredictPipeline`, métricas e artefatos (`model.pt`, scalers, configs) usados pela API.
- **Pipeline MLOps/CI/CD (treino automático → release → deploy):** detalhado em [docs/ARCHITECTURE_MLOPS.md](docs/ARCHITECTURE_MLOPS.md) e [docs/GCLOUD_DEPLOY.md](docs/GCLOUD_DEPLOY.md); cobre GitHub Actions, GitHub Releases, Docker/Cloud Build e Cloud Run.

### 🗺️ Mapa da Documentação por Perfil

- **Dev Backend / API:** [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md), [docs/DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md), [frontend/TESTING_GUIDE.md](frontend/TESTING_GUIDE.md).
- **Engenharia de ML:** [docs/ML_DOCUMENTATION.md](docs/ML_DOCUMENTATION.md), [notebooks/eda.ipynb](notebooks/eda.ipynb), [docs/TICKER_ENCODING_STRATEGY.md](docs/TICKER_ENCODING_STRATEGY.md), [cli](cli).
- **MLOps / DevOps:** [docs/ARCHITECTURE_MLOPS.md](docs/ARCHITECTURE_MLOPS.md), [docs/GCLOUD_DEPLOY.md](docs/GCLOUD_DEPLOY.md), [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md), [docs/DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md), [docs/alternatives/DEPLOY_FREE_TIER.md](docs/alternatives/DEPLOY_FREE_TIER.md).

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
