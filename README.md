# Stock Prediction LSTM API 📈

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-83%20passing-brightgreen.svg)](tests/)
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
- ✅ **Qualidade**: 83 testes (100% passando), 72.79% coverage, Ruff linter

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

---

## 🚀 Instalação

### **Pré-requisitos**
- Python 3.13+
- pip ou uv

### **Setup Rápido (Recomendado)**

Use os scripts automatizados para configurar o ambiente:

#### **Linux/Mac**
```bash
chmod +x setup.sh
./setup.sh
```

#### **Windows**
```powershell
.\setup.ps1
```

Os scripts irão:
- ✅ Criar ambiente virtual (.venv)
- ✅ Instalar todas as dependências
- ✅ Criar diretórios necessários (data/, models/, artifacts/, logs/)
- ✅ Verificar instalação do CLI
- ✅ Testar imports principais

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
# Com pip
pip install -e .

# Ou com uv
uv pip install -e .

# Dependências de desenvolvimento (opcional)
pip install -e ".[dev]"

# API REST (opcional)
pip install -e ".[api]"
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

## 📖 Guia de Uso

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
