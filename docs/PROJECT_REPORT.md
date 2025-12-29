# Relatório do Projeto - Stock Prediction LSTM API

## 📋 Visão Geral

Este projeto implementa um sistema completo de **ML Engineering** para previsão de preços de ações usando LSTM (Long Short-Term Memory), com foco em boas práticas de engenharia, monitoramento, versionamento e qualidade de código.

---

## 🏗️ Arquitetura do Sistema

### **Componentes Principais**

```
stock-prediction-lstm-api/
├── src/ml/                      # Core ML components
│   ├── data/                    # Data pipeline
│   │   ├── ingestion.py         # Yahoo Finance integration
│   │   ├── feature_engineering.py  # Technical indicators
│   │   └── preprocessing.py     # Normalization & sequences
│   ├── models/                  # Neural network models
│   │   └── lstm.py              # PyTorch LSTM implementation
│   ├── training/                # Training infrastructure
│   │   ├── trainer.py           # Training loop & checkpoints
│   │   ├── early_stopping.py    # Early stopping callback
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── hyperparameter_tuner.py  # Optuna integration
│   │   └── experiment_tracker.py    # MLflow tracking
│   ├── pipeline/                # End-to-end pipelines
│   │   ├── train_pipeline.py    # Training orchestration
│   │   └── predict_pipeline.py  # Prediction workflow
│   ├── monitoring/              # Production monitoring
│   │   └── drift_detector.py    # Data/concept drift detection
│   └── utils/                   # Utilities
│       ├── persistence.py       # Data versioning & artifacts
│       ├── device.py            # CPU/GPU management
│       ├── logging.py           # Structured logging
│       └── seed.py              # Reproducibility
├── cli/                         # Command-line interface
│   └── main.py                  # 5 CLI commands
├── tests/                       # Comprehensive test suite
│   ├── integration/             # Integration tests
│   ├── unit/                    # Unit tests
│   └── test_*.py                # Test modules
└── docs/                        # Documentation
```

---

## 🎯 Funcionalidades Implementadas

### **1. Pipeline de Dados**
- ✅ **Ingestão**: Download automático de dados do Yahoo Finance
- ✅ **Feature Engineering**: 14 indicadores técnicos (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- ✅ **Preprocessing**: Normalização MinMaxScaler, criação de sequências temporais
- ✅ **Validação**: Verificação de qualidade dos dados

### **2. Modelo LSTM**
- ✅ **Arquitetura**: PyTorch LSTM multi-camadas com dropout
- ✅ **Flexibilidade**: Configurável (hidden_size, num_layers, dropout, lookback)
- ✅ **Checkpoint**: Salvamento completo com arquitetura e pesos

### **3. Treinamento**
- ✅ **Trainer**: Loop de treinamento com validação e logging
- ✅ **Early Stopping**: Previne overfitting com patience configurável
- ✅ **MLflow**: Rastreamento de experimentos, métricas e modelos
- ✅ **Otimização**: Optuna para hyperparameter tuning
- ✅ **Métricas**: MAE, RMSE, MAPE, R², Directional Accuracy

### **4. Pipelines Orquestrados**

#### **TrainPipeline** (5 etapas)
```python
1. Data Ingestion      → Download from Yahoo Finance
2. Feature Engineering → Add technical indicators
3. Preprocessing       → Normalize & create sequences
4. Training           → Train LSTM with validation
5. Evaluation         → Calculate test metrics
```

#### **PredictPipeline** (4 etapas)
```python
1. Data Ingestion      → Download latest 2 years
2. Feature Engineering → Add indicators
3. Preprocessing       → Prepare last sequence
4. Prediction         → Multi-step forecasting
```

### **5. Monitoramento & Versionamento**
- ✅ **Drift Detection**: KS-test e PSI para detectar drift
- ✅ **Data Versioning**: Controle de versões de datasets
- ✅ **Artifact Management**: Salvamento de scalers, configs, modelos
- ✅ **Auto Cleanup**: Limpeza automática de versões antigas

### **6. Interface CLI**
```bash
# 5 comandos disponíveis:
stock-predict train      # Treinar modelo
stock-predict predict    # Fazer previsões
stock-predict tune       # Otimizar hiperparâmetros
stock-predict drift      # Detectar drift
stock-predict pipeline   # Executar pipeline completo
```

### **7. Qualidade & Testes**
- ✅ **Ruff**: Linter/formatter único (substitui black+isort+flake8+mypy)
- ✅ **83 testes**: 100% passando
- ✅ **Coverage**: 72.79%
- ✅ **Testes End-to-End**: Treino, retreino, predição
- ✅ **Testes de Integração**: 8 testes
- ✅ **Testes Unitários**: 75+ testes

---

## 🔌 Integração com API REST

### **Arquitetura Proposta**

```
┌─────────────────┐
│   Frontend      │
│   (React/Vue)   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   Flask         │  ← API REST Layer
│   + Pydantic    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Pipelines   │  ← Existing Code
│  (TrainPipeline)│
│  (PredictPipeline)│
└─────────────────┘
```

### **Implementação Recomendada**

#### **1. Estrutura de Diretórios**
```
stock-prediction-lstm-api/
├── api/                         # NEW: API layer
│   ├── __init__.py
│   ├── main.py                  # FastAPI app
│   ├── routers/                 # API endpoints
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── monitoring.py
│   │   └── health.py
│   ├── schemas/                 # Pydantic models
│   │   ├── train_request.py
│   │   ├── predict_request.py
│   │   └── responses.py
│   ├── dependencies.py          # Dependency injection
│   └── background_tasks.py      # Async training
└── src/ml/                      # Existing ML code
```

#### **2. Endpoints Principais**

```python
# api/routers/train.py
@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks
):
    """
    POST /api/v1/train
    {
        "ticker": "PETR4.SA",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "lookback": 60,
        "hidden_size": 64,
        "epochs": 50
    }
    """
    task_id = str(uuid.uuid4())
    background_tasks.add_task(
        run_training_pipeline,
        task_id=task_id,
        params=request
    )
    return {"task_id": task_id, "status": "queued"}

# api/routers/predict.py
@router.post("/predict", response_model=PredictResponse)
async def predict_prices(request: PredictRequest):
    """
    POST /api/v1/predict
    {
        "ticker": "PETR4.SA",
        "model_path": "models/best_model.pt",
        "days_ahead": 5
    }
    """
    pipeline = PredictPipeline(
        model_path=request.model_path,
        ticker=request.ticker,
        lookback=request.lookback
    )
    predictions = pipeline.predict(days_ahead=request.days_ahead)
    
    return {
        "ticker": request.ticker,
        "predictions": predictions.to_dict(orient="records"),
        "generated_at": datetime.now().isoformat()
    }

# api/routers/monitoring.py
@router.post("/drift/detect")
async def detect_drift(request: DriftRequest):
    """
    POST /api/v1/drift/detect
    {
        "ticker": "PETR4.SA",
        "reference_version": "20240101_120000",
        "production_version": "20240201_120000"
    }
    """
    detector = DriftDetector()
    ref_data = load_data_version(request.reference_version)
    prod_data = load_data_version(request.production_version)
    
    report = detector.detect_drift(ref_data, prod_data)
    
    return {
        "has_drift": report["has_drift"],
        "drifted_features": report["drifted_features"],
        "drift_scores": report["drift_scores"],
        "recommendation": "retrain" if report["has_drift"] else "ok"
    }

# api/routers/health.py
@router.get("/health")
async def health_check():
    """GET /api/v1/health"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
```

#### **3. Schemas Pydantic**

```python
# api/schemas/train_request.py
class TrainRequest(BaseModel):
    ticker: str = Field(..., example="PETR4.SA")
    start_date: str = Field(..., example="2023-01-01")
    end_date: str = Field(..., example="2024-01-01")
    lookback: int = Field(60, ge=5, le=200)
    hidden_size: int = Field(64, ge=16, le=512)
    num_layers: int = Field(2, ge=1, le=5)
    epochs: int = Field(50, ge=1, le=1000)
    learning_rate: float = Field(0.001, gt=0, lt=1)
    
    class Config:
        schema_extra = {
            "example": {
                "ticker": "PETR4.SA",
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "lookback": 60,
                "hidden_size": 64,
                "epochs": 50
            }
        }

# api/schemas/predict_request.py
class PredictRequest(BaseModel):
    ticker: str
    model_path: str
    lookback: int = 60
    days_ahead: int = Field(5, ge=1, le=30)

# api/schemas/responses.py
class TrainResponse(BaseModel):
    task_id: str
    status: str
    message: str = "Training started"

class PredictResponse(BaseModel):
    ticker: str
    predictions: List[Dict[str, Any]]
    generated_at: str
    model_version: Optional[str]
```
 
 

## 🎓 Tecnologias Utilizadas

| Categoria | Tecnologia | Versão |
|-----------|-----------|--------|
| **ML Framework** | PyTorch | 2.1+ |
| **Data** | pandas, numpy | latest |
| **Tracking** | MLflow | 2.9+ |
| **Tuning** | Optuna | 3.5+ |
| **Testing** | pytest, pytest-cov | 8.0+, 7.0+ |
| **Quality** | Ruff | 0.1+ |
| **CLI** | Click | 8.1+ |
| **Logging** | Loguru | 0.7+ |

---
  