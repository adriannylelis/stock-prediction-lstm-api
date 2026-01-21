# 🚀 Stock Prediction LSTM API - PETR4.SA

**Previsão de preços de ações usando LSTM + Deploy automatizado na Google Cloud Platform**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Platform-yellow.svg)](https://cloud.google.com/)
[![Firestore](https://img.shields.io/badge/Firestore-Native-orange.svg)](https://firebase.google.com/docs/firestore)

API REST completa para previsão de preços de ações brasileiras usando **LSTM**, com foco em **PETR4.SA (Petrobras)**. Inclui frontend web, treino automatizado via GitHub Actions, histórico de predições no Firestore e deploy na Google Cloud Platform.

---

## ✨ Funcionalidades Principais

- 📈 **Predições LSTM**: Modelo PyTorch treinado para prever preços de ações
- 💾 **Histórico Persistente**: Todas as predições são salvas no Google Cloud Firestore
- 📊 **Analytics**: Endpoints para acompanhar acurácia ao longo do tempo (MAE, MAPE, RMSE)
- 🔄 **Auto-Update**: Predições passadas são atualizadas automaticamente com preços reais
- 🚫 **UPSERT Inteligente**: Múltiplas predições no mesmo dia atualizam o registro existente
- 🎯 **API REST Completa**: Health check, model info, predictions, analytics
- 🌐 **Frontend React**: Interface web para visualização de predições
- 🔁 **CI/CD Automatizado**: Treino semanal + deploy via GitHub Actions
- 🐳 **Docker**: Containerização completa (backend, frontend, Firestore emulator)

---

## 📋 Índice

- [⚡ Quick Start](#-quick-start)
- [🎯 Visão Geral](#-visão-geral)
- [🏗️ Arquitetura](#️-arquitetura)
- [🗄️ Firestore - Histórico de Predições](#️-firestore---histórico-de-predições)
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

**Persistência Automática:**
- Todas as predições são salvas automaticamente no Firestore
- Se houver predição existente para o mesmo ticker e data, ela é atualizada (UPSERT)
- Predições passadas são atualizadas automaticamente com o preço real quando disponível

---

#### **4. GET /analytics/<ticker> - Histórico e Métricas**

Retorna histórico completo de predições e métricas de acurácia para um ticker.

```bash
curl http://localhost:5001/analytics/AAPL
```

**Response (200 OK):**
```json
{
  "success": true,
  "ticker": "AAPL",
  "predictions": [
    {
      "prediction_date": "2026-01-20",
      "predicted_price": 88.59,
      "current_price": 273.76,
      "actual_price": 275.12,
      "error": 186.53,
      "error_percent": 67.81,
      "model_version": "1.0",
      "predicted_at": "2026-01-19T04:18:19"
    }
  ],
  "metrics": {
    "mae": 186.53,
    "mape": 67.81,
    "rmse": 186.53,
    "total_predictions": 1,
    "predictions_with_actual": 1
  }
}
```

---

#### **5. GET /analytics/<ticker>/pending - Predições Pendentes**

Retorna predições que ainda não possuem preço real atualizado.

```bash
curl http://localhost:5001/analytics/AAPL/pending
```

**Response (200 OK):**
```json
{
  "success": true,
  "ticker": "AAPL",
  "pending_predictions": [
    {
      "prediction_date": "2026-01-20",
      "predicted_price": 88.59,
      "current_price": 273.76,
      "model_version": "1.0",
      "predicted_at": "2026-01-19T04:18:19"
    }
  ],
  "total_pending": 1
}
```

---

#### **6. GET /analytics/<ticker>/accuracy - Apenas Métricas**

Retorna apenas as métricas de acurácia sem o histórico.

```bash
curl http://localhost:5001/analytics/AAPL/accuracy
```

**Response (200 OK):**
```json
{
  "success": true,
  "ticker": "AAPL",
  "metrics": {
    "mae": 186.53,
    "mape": 67.81,
    "rmse": 186.53,
    "total_predictions": 1,
    "predictions_with_actual": 1
  }
}
```

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
│   ├── prediction.py          # Predictions (com salvamento Firestore)
│   └── analytics.py           # Analytics e histórico
├── services/                  # Business Logic
│   ├── model_service.py       # Singleton: Model + Scaler loading
│   ├── data_service.py        # Yahoo Finance integration
│   ├── predict_service.py     # Prediction pipeline orchestration
│   └── firestore_service.py   # Firestore CRUD + Analytics
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

## 🗄️ Firestore - Histórico de Predições

### **Por Que Firestore?**

O projeto utiliza **Google Cloud Firestore** para persistir histórico de predições, oferecendo:

- ✅ **Serverless**: Sem necessidade de gerenciar infraestrutura de banco de dados
- ✅ **Free Tier Generoso**: 50k leituras/20k escritas/20k deletes por dia gratuitamente
- ✅ **Escalabilidade Automática**: Cresce conforme a demanda sem configuração
- ✅ **Integração Nativa**: Já está no ecossistema Google Cloud (Cloud Run, Cloud Build)
- ✅ **Baixa Latência**: Perfeito para aplicações de tempo real
- ✅ **NoSQL Flexível**: Schema-less, ideal para armazenar predições

### **Schema de Dados**

Cada predição é salva como um documento na coleção `predictions`:

```json
{
  "ticker": "AAPL",
  "prediction_date": "2026-01-20",
  "predicted_price": 88.59,
  "current_price": 273.76,
  "actual_price": 275.12,
  "error": 186.53,
  "error_percent": 67.81,
  "model_version": "1.0",
  "predicted_at": "2026-01-19T04:18:19"
}
```

**Campos:**
- `ticker`: Símbolo da ação (ex: AAPL, PETR4.SA)
- `prediction_date`: Data para qual a predição foi feita (T+1)
- `predicted_price`: Preço previsto pelo modelo
- `current_price`: Último preço conhecido no momento da predição
- `actual_price`: Preço real observado (preenchido automaticamente depois)
- `error`: Diferença absoluta entre previsto e real
- `error_percent`: Erro percentual
- `model_version`: Versão do modelo que fez a predição
- `predicted_at`: Timestamp UTC da predição

### **Funcionalidades Implementadas**

#### **1. UPSERT Automático**
Múltiplas predições para o mesmo ticker e data **atualizam** o registro existente ao invés de criar duplicatas:

```python
# Primeira predição do dia para AAPL
POST /predict {"ticker": "AAPL"}  # ✅ Cria novo documento

# Segunda predição do dia para AAPL
POST /predict {"ticker": "AAPL"}  # ✅ Atualiza o documento existente
```

**Lógica:**
- Query por `(ticker, prediction_date)`
- Se encontrar: **UPDATE** (preserva histórico, atualiza valores)
- Se não encontrar: **CREATE** (novo documento)

#### **2. Auto-Update de Preços Reais**
Predições passadas são atualizadas automaticamente com o preço real quando você faz uma nova predição:

```python
# Dia 1: Fazer predição para amanhã
POST /predict {"ticker": "AAPL"}
# Salvo: { "prediction_date": "2026-01-20", "predicted_price": 88.59, "actual_price": null }

# Dia 2: Fazer nova predição
POST /predict {"ticker": "AAPL"}
# ✅ Auto-atualiza predição do Dia 1 com preço real
# Atualizado: { "prediction_date": "2026-01-20", "predicted_price": 88.59, "actual_price": 275.12, "error": 186.53 }
# ✅ Cria nova predição para amanhã
```

**Benefícios:**
- Não precisa rodar scripts separados para atualizar preços reais
- Cálculo automático de erro (MAE, MAPE, RMSE)
- Histórico sempre atualizado

#### **3. Analytics e Métricas**
Endpoints dedicados para acompanhar performance do modelo:

```bash
# Ver histórico completo + métricas
GET /analytics/AAPL

# Ver apenas predições pendentes (sem preço real)
GET /analytics/AAPL/pending

# Ver apenas métricas de acurácia
GET /analytics/AAPL/accuracy
```

**Métricas calculadas:**
- **MAE** (Mean Absolute Error): Erro médio absoluto
- **MAPE** (Mean Absolute Percentage Error): Erro percentual médio
- **RMSE** (Root Mean Squared Error): Raiz do erro quadrático médio
- **Total de predições**: Quantidade de predições feitas
- **Predições com preço real**: Quantidade de predições já validadas

### **Setup Local (Emulator)**

Para desenvolvimento local, use o Firestore Emulator via Docker Compose:

```bash
# Iniciar emulator
docker-compose up -d firestore

# Verificar se está rodando
curl http://localhost:8080

# Rodar backend conectado ao emulator
FIRESTORE_EMULATOR_HOST=localhost:8080 \
GOOGLE_CLOUD_PROJECT=stock-prediction-local \
python src/api/main.py
```

### **Setup em Produção (GCP)**

Execute o script de setup para configurar Firestore no Google Cloud:

```bash
# Rodar no Cloud Shell ou localmente com gcloud CLI
bash scripts/setup_firestore.sh
```

**O script:**
1. Habilita a API do Firestore
2. Cria o banco de dados Firestore Native em `us-central1`
3. Configura permissões IAM para Cloud Run acessar Firestore
4. Valida a configuração

**Ver detalhes completos:** [docs/FIRESTORE_SETUP_GUIDE.md](docs/FIRESTORE_SETUP_GUIDE.md)

### **Monitoramento**

```bash
# Ver predições salvas
gcloud firestore databases list

# Consultar coleção via gcloud
gcloud firestore operations list

# Usar console web
https://console.firebase.google.com/project/stock-prediction-prod/firestore
```

### **Custos**

**Free Tier (sempre gratuito):**
- 50,000 reads/dia
- 20,000 writes/dia
- 20,000 deletes/dia
- 1 GB storage

**Estimativa de uso:**
- ~10 predições/dia × 30 dias = 300 writes/mês
- ~100 leituras analytics/dia = 3,000 reads/mês
- **Custo: R$ 0,00** (bem dentro do free tier)

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

# 6. Ver histórico e métricas de AAPL
curl http://localhost:5001/analytics/AAPL | python -m json.tool

# 7. Ver predições pendentes (sem preço real)
curl http://localhost:5001/analytics/AAPL/pending | python -m json.tool

# 8. Ver apenas métricas de acurácia
curl http://localhost:5001/analytics/AAPL/accuracy | python -m json.tool
```

---

## 🚦 Rate Limiting

A API implementa limites de requisições por minuto para proteger contra abuso e controlar custos:

| Endpoint | Limite | Descrição |
|----------|--------|-----------|
| `GET /health` | 100/min | Health checks |
| `GET /model/info` | 30/min | Informações do modelo |
| `POST /predict` | **10/min** | Predições (custoso) |
| `GET /analytics/*` | 30/min | Analytics e histórico |

### Resposta quando limite é excedido

```bash
# 11ª requisição em 1 minuto
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d '{"ticker": "AAPL"}'
```

```json
{
  "error": "RateLimitExceeded",
  "message": "Limite de requisições excedido. Tente novamente em alguns instantes.",
  "status": 429,
  "retry_after": "1 per 1 minute"
}
```

### Configuração

```bash
# Desabilitar rate limiting (desenvolvimento)
export RATE_LIMIT_ENABLED=false

# Usar Redis (produção com múltiplas instâncias)
export RATE_LIMIT_STORAGE_URI=redis://redis:6379/0
```

**Ver detalhes:** [docs/API_DOCUMENTATION.md#rate-limiting](docs/API_DOCUMENTATION.md#rate-limiting)

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
| [FIRESTORE_SETUP_GUIDE.md](docs/FIRESTORE_SETUP_GUIDE.md) | Como configurar e usar o Firestore para histórico de predições |
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
| **Database** | Google Cloud Firestore | latest |
| **Rate Limiting** | Flask-Limiter | 3.5+ |
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
