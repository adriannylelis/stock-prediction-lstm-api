# Documentação do Modelo - Stock Prediction LSTM

## 📋 Visão Geral

Documentação completa das classes, métodos e pipelines do sistema de ML Engineering para previsão de ações.

---

## 🏗️ Arquitetura de Componentes

```
┌─────────────────────────────────────────────────────┐
│                  CLI Layer                          │
│  (train, predict, tune, drift, pipeline)            │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────┐
│              Pipeline Layer                         │
│  ┌──────────────────┐   ┌──────────────────────┐  │
│  │  TrainPipeline   │   │  PredictPipeline     │  │
│  └──────────────────┘   └──────────────────────┘  │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────┐
│              ML Core Layer                          │
│  ┌────────────┐ ┌─────────┐ ┌─────────────────┐   │
│  │   Data     │ │ Models  │ │   Training      │   │
│  │ Pipeline   │ │  LSTM   │ │   Trainer       │   │
│  └────────────┘ └─────────┘ └─────────────────┘   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────┐
│            Monitoring & Utils                       │
│  ┌──────────────┐  ┌─────────────┐ ┌────────────┐ │
│  │DriftDetector │  │Persistence  │ │  Logging   │ │
│  └──────────────┘  └─────────────┘ └────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

## 📦 Pipeline Layer

### **TrainPipeline**

**Propósito**: Orquestrar pipeline completo de treinamento end-to-end.

**Localização**: `src/ml/pipeline/train_pipeline.py`

#### **Inicialização**

```python
from src.ml.pipeline.train_pipeline import TrainPipeline

pipeline = TrainPipeline(
    ticker: str,                    # Ticker da ação (ex: "PETR4.SA")
    start_date: str,                # Data início (ex: "2023-01-01")
    end_date: str,                  # Data fim (ex: "2024-01-01")
    lookback: int = 60,             # Janela temporal
    hidden_size: int = 64,          # Tamanho da camada LSTM
    num_layers: int = 2,            # Número de camadas LSTM
    dropout: float = 0.2,           # Dropout rate
    epochs: int = 50,               # Número de épocas
    batch_size: int = 32,           # Tamanho do batch
    learning_rate: float = 0.001,   # Learning rate
    early_stop_patience: int = 10,  # Patience do early stopping
    model_save_path: str = "models/best_model.pt",  # Path do modelo
    experiment_name: str = None,    # Nome experimento MLflow (opcional)
    device: str = None              # "cpu", "cuda", "mps" (auto-detect)
)
```

#### **Métodos Públicos**

##### **`run() -> Dict[str, Any]`**

Executa pipeline completo de treinamento.

**Retorno**:
```python
{
    'model_path': str,              # Path do modelo salvo (best_model.pt)
    'training_history': {
        'train_loss': List[float],  # Loss de treino por época
        'val_loss': List[float],    # Loss de validação por época
        'epochs_trained': int       # Número de épocas treinadas
    },
    'test_metrics': {
        'MAE': float,               # Mean Absolute Error
        'RMSE': float,              # Root Mean Squared Error
        'MAPE': float,              # Mean Absolute Percentage Error (%)
        'R2': float,                # R² Score
        'Directional_Accuracy': float  # Acurácia direcional (%)
    },
    'metadata': {
        'ticker': str,
        'start_date': str,
        'end_date': str,
        'lookback': int,
        'model_config': {
            'hidden_size': int,
            'num_layers': int,
            'dropout': float
        },
        'training_time': float,     # Tempo de treino (segundos)
        'timestamp': str            # ISO format timestamp
    }
}
```

**Exemplo de Uso**:
```python
# 1. Criar pipeline
pipeline = TrainPipeline(
    ticker="PETR4.SA",
    start_date="2023-01-01",
    end_date="2024-01-01",
    lookback=60,
    hidden_size=64,
    epochs=50
)

# 2. Executar
results = pipeline.run()

# 3. Acessar resultados
print(f"Modelo salvo em: {results['model_path']}")
print(f"RMSE: {results['test_metrics']['RMSE']:.4f}")
print(f"MAE: {results['test_metrics']['MAE']:.4f}")
print(f"R²: {results['test_metrics']['R2']:.4f}")
```

#### **Fluxo Interno (5 Etapas)**

```python
def run(self):
    # 1. Data Ingestion
    self._ingest_data()  # Download de dados do Yahoo Finance
    
    # 2. Feature Engineering
    self._engineer_features()  # Adiciona 14 indicadores técnicos
    
    # 3. Preprocessing
    self._preprocess_data()  # Normaliza e cria sequências
    
    # 4. Training
    self._train_model()  # Treina LSTM com validation
    
    # 5. Evaluation
    self._evaluate_model()  # Calcula métricas no test set
    
    # 6. Save Results
    return self._save_results()  # Retorna dicionário de resultados
```

#### **Artefatos Salvos**

Após `pipeline.run()`, os seguintes arquivos são criados:

```
models/
├── best_model.pt      # Checkpoint completo do modelo
│   └── Contém:
│       - model_state_dict (pesos)
│       - optimizer_state_dict
│       - best_val_loss
│       - history (train/val loss)
│       - input_size, hidden_size, num_layers, dropout
│
└── scaler.pkl         # MinMaxScaler treinado
    └── Usado para normalizar/desnormalizar predições
```

---

### **PredictPipeline**

**Propósito**: Fazer previsões multi-step usando modelo treinado.

**Localização**: `src/ml/pipeline/predict_pipeline.py`

#### **Inicialização**

```python
from src.ml.pipeline.predict_pipeline import PredictPipeline

pipeline = PredictPipeline(
    model_path: str,       # Path do modelo treinado (best_model.pt)
    ticker: str,           # Ticker da ação (ex: "PETR4.SA")
    lookback: int = 60,    # Janela temporal (deve ser igual ao treino)
    device: str = None     # "cpu", "cuda", "mps" (auto-detect)
)
```

#### **Métodos Públicos**

##### **`predict(days_ahead: int = 5) -> pd.DataFrame`**

Gera previsões multi-step para os próximos N dias.

**Parâmetros**:
- `days_ahead` (int): Número de dias para prever (padrão: 5)

**Retorno**:
```python
pd.DataFrame([
    {'Date': '2025-12-27', 'Predicted_Close': 29.95},
    {'Date': '2025-12-28', 'Predicted_Close': 29.94},
    {'Date': '2025-12-29', 'Predicted_Close': 29.94},
    ...
])
```

**Exemplo de Uso**:
```python
# 1. Criar pipeline
pipeline = PredictPipeline(
    model_path="models/best_model.pt",
    ticker="PETR4.SA",
    lookback=60
)

# 2. Gerar previsões
predictions = pipeline.predict(days_ahead=5)

# 3. Visualizar
print(predictions)
#         Date  Predicted_Close
# 0 2025-12-27        29.95
# 1 2025-12-28        29.94
# ...

# 4. Salvar
predictions.to_csv("predictions.csv", index=False)
```

#### **Fluxo Interno (4 Etapas)**

```python
def predict(self, days_ahead: int = 5):
    # 1. Load Model
    self._load_model()  # Carrega checkpoint + arquitetura
    
    # 2. Ingest Latest Data
    df = self._ingest_latest_data()  # Download últimos 2 anos
    
    # 3. Preprocess
    sequence = self._preprocess_latest(df)  # Normaliza + cria sequência
    
    # 4. Generate Predictions
    return self._generate_predictions(sequence, days_ahead)
```

#### **Estratégia Multi-Step**

O pipeline usa **auto-regression** para gerar múltiplas previsões:

```python
# Pseudo-código
predictions = []
current_sequence = last_60_days  # Sequência inicial

for i in range(days_ahead):
    # Prever próximo dia
    next_pred = model(current_sequence)
    predictions.append(next_pred)
    
    # Atualizar sequência (sliding window)
    current_sequence = append(current_sequence[1:], next_pred)

return predictions
```

---

## 🧠 ML Core Layer

### **LSTM Model**

**Localização**: `src/ml/models/lstm.py`

#### **Classe: `LSTMModel`**

```python
import torch.nn as nn
from src.ml.models.lstm import LSTMModel

model = LSTMModel(
    input_size: int = 1,        # Número de features de entrada
    hidden_size: int = 64,      # Tamanho da camada oculta
    num_layers: int = 2,        # Número de camadas LSTM
    dropout_prob: float = 0.2,  # Dropout rate
    output_size: int = 1        # Número de saídas (1 = preço)
)
```

**Métodos**:
- `forward(x: Tensor) -> Tensor`: Forward pass
- `init_hidden(batch_size: int) -> Tuple[Tensor, Tensor]`: Inicializa hidden states

**Exemplo**:
```python
import torch

# Criar modelo
model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)

# Input: (batch_size, seq_len, input_size)
x = torch.randn(32, 60, 1)  # 32 samples, 60 timesteps, 1 feature

# Forward
output = model(x)  # Shape: (32, 1)
```

---

### **Trainer**

**Localização**: `src/ml/training/trainer.py`

#### **Classe: `Trainer`**

```python
from src.ml.training.trainer import Trainer

trainer = Trainer(
    model: nn.Module,                   # Modelo PyTorch
    train_loader: DataLoader,           # DataLoader de treino
    val_loader: DataLoader,             # DataLoader de validação
    criterion: nn.Module,               # Loss function (ex: MSELoss)
    optimizer: torch.optim.Optimizer,   # Optimizer (ex: Adam)
    device: str = "cpu",                # Device
    early_stopping_patience: int = 10,  # Patience
    checkpoint_path: str = None         # Path para salvar checkpoints
)
```

**Métodos**:

##### **`train(epochs: int) -> Dict[str, List[float]]`**

Treina modelo por N épocas com early stopping.

**Retorno**:
```python
{
    'train_loss': [0.123, 0.098, 0.087, ...],  # Loss por época
    'val_loss': [0.145, 0.112, 0.095, ...]     # Val loss por época
}
```

##### **`save_checkpoint(epoch: int, path: str)`**

Salva checkpoint completo do modelo.

**Formato do Checkpoint**:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,      # Pesos do modelo
    'optimizer_state_dict': dict,         # Estado do optimizer
    'best_val_loss': float,               # Melhor val loss
    'history': dict,                      # Histórico de treino
    'input_size': int,                    # Arquitetura
    'hidden_size': int,
    'num_layers': int,
    'dropout': float
}
```

**Exemplo**:
```python
import torch.nn as nn
import torch.optim as optim

# Setup
model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Criar trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    early_stopping_patience=10,
    checkpoint_path="models/best_model.pt"
)

# Treinar
history = trainer.train(epochs=50)

# Verificar se early stopping foi acionado
if len(history['train_loss']) < 50:
    print("Early stopping acionado!")
```

---

### **Metrics**

**Localização**: `src/ml/training/metrics.py`

#### **Funções Disponíveis**

```python
from src.ml.training.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_r2,
    calculate_directional_accuracy,
    evaluate_model
)
```

##### **`calculate_mae(y_true, y_pred) -> float`**
Mean Absolute Error: $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

##### **`calculate_rmse(y_true, y_pred) -> float`**
Root Mean Squared Error: $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

##### **`calculate_mape(y_true, y_pred) -> float`**
Mean Absolute Percentage Error: $MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$

##### **`calculate_r2(y_true, y_pred) -> float`**
R² Score: $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

##### **`calculate_directional_accuracy(y_true, y_pred) -> float`**
Acurácia da direção (up/down): % de vezes que o modelo acertou a direção do movimento.

##### **`evaluate_model(model, dataloader, criterion, device) -> Dict`**

Avalia modelo em um dataset completo.

**Retorno**:
```python
{
    'MAE': 0.0987,
    'RMSE': 0.1234,
    'MAPE': 3.45,
    'R2': 0.8765,
    'Directional_Accuracy': 65.43
}
```

**Exemplo**:
```python
import numpy as np

y_true = np.array([10.0, 11.0, 12.0, 13.0])
y_pred = np.array([9.8, 11.2, 11.9, 13.1])

# Calcular métricas individuais
mae = calculate_mae(y_true, y_pred)     # 0.15
rmse = calculate_rmse(y_true, y_pred)   # 0.158
mape = calculate_mape(y_true, y_pred)   # 1.39%
r2 = calculate_r2(y_true, y_pred)       # 0.98

# Ou usar evaluate_model para todas de uma vez
metrics = evaluate_model(model, test_loader, criterion, device)
print(metrics)
# {'MAE': 0.15, 'RMSE': 0.158, 'MAPE': 1.39, 'R2': 0.98, ...}
```

---

### **Hyperparameter Tuner**

**Localização**: `src/ml/training/hyperparameter_tuner.py`

#### **Classe: `HyperparameterTuner`**

```python
from src.ml.training.hyperparameter_tuner import HyperparameterTuner

tuner = HyperparameterTuner(
    ticker: str,              # Ticker da ação
    start_date: str,          # Data de início
    end_date: str,            # Data de fim
    n_trials: int = 20,       # Número de trials Optuna
    timeout: int = 3600,      # Timeout em segundos
    device: str = None        # Device
)
```

**Métodos**:

##### **`tune() -> Dict[str, Any]`**

Executa otimização de hiperparâmetros usando Optuna.

**Parâmetros Otimizados**:
- `lookback`: [10, 20, 30, 60, 90]
- `hidden_size`: [16, 32, 64, 128, 256]
- `num_layers`: [1, 2, 3, 4]
- `dropout`: [0.1, 0.5]
- `learning_rate`: [1e-4, 1e-2] (log scale)

**Retorno**:
```python
{
    'best_params': {
        'lookback': 60,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'learning_rate': 0.0005
    },
    'best_value': 0.1234,  # Melhor RMSE
    'study': optuna.study.Study  # Objeto Study do Optuna
}
```

**Exemplo**:
```python
# Criar tuner
tuner = HyperparameterTuner(
    ticker="PETR4.SA",
    start_date="2023-01-01",
    end_date="2024-01-01",
    n_trials=30,
    timeout=7200  # 2 horas
)

# Executar tuning
results = tuner.tune()

# Melhores hiperparâmetros
print(results['best_params'])
# {'lookback': 60, 'hidden_size': 128, ...}

# Treinar modelo final com melhores params
pipeline = TrainPipeline(
    ticker="PETR4.SA",
    start_date="2023-01-01",
    end_date="2024-01-01",
    **results['best_params']  # Unpack best params
)
final_model = pipeline.run()
```

---

## 📊 Data Layer

### **StockDataIngestion**

**Localização**: `src/ml/data/ingestion.py`

#### **Classe: `StockDataIngestion`**

```python
from src.ml.data.ingestion import StockDataIngestion

ingestion = StockDataIngestion(
    ticker: str,              # Ticker (ex: "PETR4.SA")
    start_date: str,          # Data início (YYYY-MM-DD)
    end_date: str             # Data fim (YYYY-MM-DD)
)
```

**Métodos**:

##### **`download() -> pd.DataFrame`**

Baixa dados históricos do Yahoo Finance.

**Retorno**:
```python
pd.DataFrame([
    {'Date': '2023-01-02', 'Open': 30.0, 'High': 30.5, 'Low': 29.8, 'Close': 30.2, 'Volume': 1000000},
    ...
])
```

**Exemplo**:
```python
# Baixar dados
ingestion = StockDataIngestion(
    ticker="PETR4.SA",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

df = ingestion.download()
print(f"Downloaded {len(df)} records")
# Downloaded 248 records
```

---

### **StockFeatureEngineering**

**Localização**: `src/ml/data/feature_engineering.py`

#### **Classe: `StockFeatureEngineering`**

```python
from src.ml.data.feature_engineering import StockFeatureEngineering

engineer = StockFeatureEngineering()
```

**Métodos**:

##### **`add_features(df: pd.DataFrame) -> pd.DataFrame`**

Adiciona 14 indicadores técnicos aos dados.

**Indicadores Adicionados**:
1. **SMA_20**: Simple Moving Average (20 dias)
2. **SMA_50**: Simple Moving Average (50 dias)
3. **EMA_12**: Exponential Moving Average (12 dias)
4. **RSI**: Relative Strength Index
5. **MACD**: Moving Average Convergence Divergence
6. **MACD_Signal**: Linha de sinal do MACD
7. **MACD_Hist**: Histograma MACD
8. **BB_Upper**: Bollinger Band superior
9. **BB_Middle**: Bollinger Band média
10. **BB_Lower**: Bollinger Band inferior
11. **ATR**: Average True Range
12. **OBV**: On-Balance Volume
13. **ROC**: Rate of Change
14. **Stochastic**: Stochastic Oscillator

**Retorno**:
```python
pd.DataFrame([
    {
        'Date': '2023-01-02',
        'Close': 30.2,
        'Volume': 1000000,
        'SMA_20': 29.8,
        'RSI': 55.3,
        'MACD': 0.12,
        ...  # + 11 outros indicadores
    },
    ...
])
```

**Exemplo**:
```python
# Adicionar features
engineer = StockFeatureEngineering()
df_with_features = engineer.add_features(df)

print(f"Features: {df_with_features.columns.tolist()}")
# ['Date', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'RSI', ...]
```

---

### **StockPreprocessor**

**Localização**: `src/ml/data/preprocessing.py`

#### **Classe: `StockPreprocessor`**

```python
from src.ml.data.preprocessing import StockPreprocessor

preprocessor = StockPreprocessor(
    lookback_period: int = 60,    # Janela temporal
    train_split: float = 0.7,     # % treino
    val_split: float = 0.15       # % validação (resto = test)
)
```

**Métodos**:

##### **`normalize(data: np.ndarray, fit: bool = True) -> np.ndarray`**

Normaliza dados usando MinMaxScaler.

**Parâmetros**:
- `data`: Dados a normalizar
- `fit`: Se True, treina scaler; se False, usa scaler já treinado

##### **`inverse_transform(data: np.ndarray) -> np.ndarray`**

Desnormaliza dados (volta para escala original).

##### **`create_sequences(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`**

Cria sequências temporais para LSTM.

**Retorno**:
- `X`: Shape (n_samples, lookback_period, n_features)
- `y`: Shape (n_samples, 1)

##### **`prepare_data(df: pd.DataFrame, target_col: str = 'Close') -> Dict`**

Pipeline completo de preprocessing.

**Retorno**:
```python
{
    'train': (X_train, y_train),  # Tupla de arrays
    'val': (X_val, y_val),
    'test': (X_test, y_test),
    'scaler': MinMaxScaler,       # Scaler treinado
    'metadata': {
        'n_samples': 218,
        'n_features': 19,
        'lookback': 60,
        'train_size': 152,
        'val_size': 32,
        'test_size': 34
    }
}
```

**Exemplo**:
```python
# Preparar dados
preprocessor = StockPreprocessor(lookback_period=60)
data_dict = preprocessor.prepare_data(df, target_col='Close')

# Acessar dados
X_train, y_train = data_dict['train']
X_val, y_val = data_dict['val']
X_test, y_test = data_dict['test']

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
# Train: (152, 60, 19), Val: (32, 60, 19), Test: (34, 60, 19)
```

---

## 🔍 Monitoring Layer

### **DriftDetector**

**Localização**: `src/ml/monitoring/drift_detector.py`

#### **Classe: `DriftDetector`**

```python
from src.ml.monitoring.drift_detector import DriftDetector

detector = DriftDetector(
    ks_threshold: float = 0.05,   # Threshold KS-test (p-value)
    psi_threshold: float = 0.1    # Threshold PSI
)
```

**Métodos**:

##### **`detect_drift(ref_data: pd.DataFrame, prod_data: pd.DataFrame) -> Dict`**

Detecta drift usando Kolmogorov-Smirnov test.

**Retorno**:
```python
{
    'has_drift': bool,                # True se drift detectado
    'drifted_features': List[str],    # Features com drift
    'drift_scores': {
        'Close': {'ks_stat': 0.12, 'p_value': 0.001},
        'Volume': {'ks_stat': 0.08, 'p_value': 0.05}
    },
    'recommendation': str             # "retrain" ou "ok"
}
```

##### **`detect_drift_psi(ref_data: pd.DataFrame, prod_data: pd.DataFrame) -> Dict`**

Detecta drift usando Population Stability Index (PSI).

**Retorno**:
```python
{
    'has_drift': bool,
    'feature_psi': {
        'Close': 0.15,      # PSI > 0.1 = drift
        'Volume': 0.08      # PSI < 0.1 = ok
    },
    'recommendation': str
}
```

**Exemplo**:
```python
# Dados de referência (treinamento)
ref_data = pd.DataFrame({
    'Close': np.random.normal(30, 5, 1000),
    'Volume': np.random.normal(1e6, 2e5, 1000)
})

# Dados de produção (com drift)
prod_data = pd.DataFrame({
    'Close': np.random.normal(33, 5, 1000),  # Mean shifted +3
    'Volume': np.random.normal(1e6, 2e5, 1000)
})

# Detectar drift
detector = DriftDetector()
report = detector.detect_drift(ref_data, prod_data)

if report['has_drift']:
    print(f"Drift detectado em: {report['drifted_features']}")
    print("Recomendação: Retreinar modelo")
```

---

### **DataVersionManager**

**Localização**: `src/ml/utils/persistence.py`

#### **Classe: `DataVersionManager`**

```python
from src.ml.utils.persistence import DataVersionManager

manager = DataVersionManager(
    base_path: str = "data/versioned",  # Diretório base
    auto_cleanup: bool = True,          # Auto-limpeza
    max_versions: int = 10              # Máx versões a manter
)
```

**Métodos**:

##### **`save(df: pd.DataFrame, ticker: str, metadata: Dict = None) -> str`**

Salva versão de dados com timestamp.

**Retorno**: Version ID (ex: "20240128_143022_456")

##### **`load(ticker: str, version: str) -> pd.DataFrame`**

Carrega versão específica de dados.

##### **`load_latest(ticker: str) -> pd.DataFrame`**

Carrega versão mais recente de dados.

##### **`list_versions(ticker: str) -> List[str]`**

Lista todas as versões disponíveis.

**Exemplo**:
```python
# Criar manager
manager = DataVersionManager(
    base_path="data/versioned",
    auto_cleanup=True,
    max_versions=5
)

# Salvar dados
version = manager.save(
    df=stock_data,
    ticker="PETR4.SA",
    metadata={'source': 'yahoo', 'features': 19}
)
print(f"Saved version: {version}")
# Saved version: 20240128_143022_456

# Listar versões
versions = manager.list_versions("PETR4.SA")
print(f"Available versions: {versions}")
# Available versions: ['20240128_143022_456', '20240127_100000_123', ...]

# Carregar versão específica
df = manager.load("PETR4.SA", "20240128_143022_456")

# Carregar versão mais recente
latest_df = manager.load_latest("PETR4.SA")
```

---

## 🛠️ Utils Layer

### **Device Manager**

**Localização**: `src/ml/utils/device.py`

```python
from src.ml.utils.device import get_device

# Auto-detecta melhor device (cuda > mps > cpu)
device = get_device()
print(device)  # "cuda", "mps", ou "cpu"

# Usar device específico
device = get_device(device="cpu")  # Força CPU
```

### **Logging**

**Localização**: `src/ml/utils/logging.py`

```python
from src.ml.utils.logging import get_logger

logger = get_logger(__name__)

logger.info("Training started")
logger.warning("Early stopping triggered")
logger.error("Failed to load model", exc_info=True)
```

### **Seed**

**Localização**: `src/ml/utils/seed.py`

```python
from src.ml.utils.seed import set_seed

# Garante reprodutibilidade
set_seed(42)

# Agora todos os random processes são determinísticos
```

---

## 📞 Exemplos de Uso Completos

### **Exemplo 1: Pipeline Completo de Treino e Predição**

```python
from src.ml.pipeline.train_pipeline import TrainPipeline
from src.ml.pipeline.predict_pipeline import PredictPipeline

# 1. TREINAR MODELO
train_pipeline = TrainPipeline(
    ticker="PETR4.SA",
    start_date="2023-01-01",
    end_date="2024-01-01",
    lookback=60,
    hidden_size=64,
    num_layers=2,
    epochs=50
)

results = train_pipeline.run()
print(f"✓ Modelo treinado: RMSE={results['test_metrics']['RMSE']:.4f}")

# 2. FAZER PREDIÇÕES
predict_pipeline = PredictPipeline(
    model_path=results['model_path'],
    ticker="PETR4.SA",
    lookback=60
)

predictions = predict_pipeline.predict(days_ahead=5)
print(predictions)
```

### **Exemplo 2: Workflow com Drift Detection**

```python
from src.ml.pipeline.train_pipeline import TrainPipeline
from src.ml.monitoring.drift_detector import DriftDetector
from src.ml.utils.persistence import DataVersionManager

# Setup
detector = DriftDetector()
manager = DataVersionManager()

# 1. Treinar modelo inicial
pipeline_v1 = TrainPipeline(
    ticker="PETR4.SA",
    start_date="2023-01-01",
    end_date="2023-06-30",
    lookback=60
)
results_v1 = pipeline_v1.run()

# Salvar dados de referência
ref_version = manager.save(pipeline_v1.data, ticker="PETR4.SA")

# 2. Simular produção (1 mês depois)
# ... tempo passa ...

# 3. Baixar novos dados
new_pipeline = TrainPipeline(
    ticker="PETR4.SA",
    start_date="2023-07-01",
    end_date="2023-12-31",
    lookback=60
)
new_pipeline._ingest_data()
new_pipeline._engineer_features()

# Salvar dados de produção
prod_version = manager.save(new_pipeline.data, ticker="PETR4.SA")

# 4. Detectar drift
ref_data = manager.load("PETR4.SA", ref_version)
prod_data = manager.load("PETR4.SA", prod_version)

drift_report = detector.detect_drift(ref_data, prod_data)

# 5. Retreinar se drift detectado
if drift_report['has_drift']:
    print(f"⚠️ Drift detectado em: {drift_report['drifted_features']}")
    
    pipeline_v2 = TrainPipeline(
        ticker="PETR4.SA",
        start_date="2023-06-01",  # Dados atualizados
        end_date="2023-12-31",
        lookback=60
    )
    results_v2 = pipeline_v2.run()
    
    print(f"✓ Modelo retreinado: RMSE={results_v2['test_metrics']['RMSE']:.4f}")
else:
    print("✓ Sem drift, modelo atual OK")
```

### **Exemplo 3: Hyperparameter Tuning + Training**

```python
from src.ml.training.hyperparameter_tuner import HyperparameterTuner
from src.ml.pipeline.train_pipeline import TrainPipeline

# 1. Tunning
tuner = HyperparameterTuner(
    ticker="PETR4.SA",
    start_date="2023-01-01",
    end_date="2024-01-01",
    n_trials=30
)

results = tuner.tune()
print(f"✓ Melhores params: {results['best_params']}")

# 2. Treinar com melhores params
pipeline = TrainPipeline(
    ticker="PETR4.SA",
    start_date="2023-01-01",
    end_date="2024-01-01",
    **results['best_params']  # Unpack
)

final_model = pipeline.run()
print(f"✓ Modelo final: RMSE={final_model['test_metrics']['RMSE']:.4f}")
```

---

**Versão**: 1.0.0  
**Última Atualização**: 28/12/2025  
**Autor**: FIAP Tech Challenge Team
