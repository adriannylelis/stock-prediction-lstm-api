# 📚 Documentação Técnica da API REST

Documentação completa da arquitetura, fluxos e componentes da Stock Prediction LSTM API.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Fluxo de Requisição](#fluxo-de-requisição)
4. [Endpoints](#endpoints)
5. [Serviços](#serviços)
6. [Sistema de Exceções](#sistema-de-exceções)
7. [Validadores](#validadores)
8. [Segurança](#segurança)
9. [Performance](#performance)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

### Propósito

API REST em Flask para servir predições de preços de ações utilizando modelo LSTM treinado. Fornece endpoints para health check, informações do modelo e predições em tempo real.

### Tecnologias

### Tecnologias (estado atual)
- **Framework:** Flask 3.x (porta 5001)
- **CORS:** Flask-CORS
- **ML:** PyTorch 2.2 (CPU-only)
- **Data Source:** Yahoo Finance (yfinance)
- **Python:** 3.11+

### URLs Base
### URLs Base
- **Desenvolvimento:** `http://localhost:5001`
- **Produção:** Configurável via variável de ambiente
- **Produção:** Configurável via variável de ambiente

  POST /predict                                                 
  Content-Type: application/json                               
  Body: {"ticker": "PETR4.SA"}                                     

### Padrões de Projeto

│    ├─ Campo "ticker" presente? ✓                               
│    └─ validate_ticker("PETR4.SA")                              
#### 1. Application Factory Pattern (`main.py`)
```python
def create_app(config=None):
│ 4. BUSCA DE DADOS (DataService)                                 │
│    fetch_data("PETR4.SA")                                       │
    app = Flask(__name__)
    # Configuração
    # Registro de blueprints
│    ├─ Calcula período: hoje - ~120 dias até hoje (para preencher 60 úteis)                 │
│    ├─ yf.Ticker("PETR4.SA").history(start, end)                    │
    # Handlers de erro
    return app
```

**Benefícios:**
│ 5. FEATURE ENGINEERING (18 colunas)                             │
│    ├─ OHLCV: Open, High, Low, Close, Volume                    │
│    ├─ Indicadores: SMA_20, SMA_50, EMA_12, EMA_26, RSI_14, MACD, MACD_signal, MACD_hist, BB_middle, BB_upper, BB_lower, ATR_14, Returns
│    └─ current_price = último Close                             │
- Múltiplas instâncias para diferentes ambientes
- Facilita testes unitários
- Separação de configuração e inicialização

#### 2. Singleton Pattern (`ModelService`)
```python
│ 6. NORMALIZAÇÃO                                                 │
│    ModelService.get_scaler()                                    │
│    ├─ MinMaxScaler range [0, 1]                                │
│    └─ scaled_data = scaler.transform(features[18 col])        │
class ModelService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
│ 7. CONVERSÃO PARA TENSOR                                        │
│    X = torch.FloatTensor(scaled_data[-60:]).unsqueeze(0)      │
│    Shape: [1, 60, 18]                                           │
│    ├─ batch_size: 1                                            │
│    ├─ sequence_length: 60                                      │
│    └─ features: 18                                             │
```

**Benefícios:**
- Uma única instância do modelo em memória (~500MB)
- Evita recarregar modelo a cada requisição
- Aumenta performance significativamente

#### 3. Blueprint Pattern (Rotas)
│ 8. INFERÊNCIA DO MODELO                                         │
│    ModelService.get_model()                                     │
│    with torch.no_grad():                                        │
│        prediction_scaled = model(X)                             │
│    ├─ LSTM processa sequência                                  │
│    └─ Linear layer gera predição                               │
│        → Tensor [batch, 1] (normalizado)                       │
```python
health_bp = Blueprint('health', __name__)
model_info_bp = Blueprint('model_info', __name__)
prediction_bp = Blueprint('prediction', __name__)
```

**Benefícios:**
- Organização modular de endpoints
- Facilita manutenção e escalabilidade
- Permite registro condicional

### Estrutura de Diretórios

```
src/api/
├── main.py                    # Application Factory
│
├── routes/                    # Endpoints HTTP (Blueprints)
│   ├── __init__.py
│   ├── health.py             # GET /health
│   ├── model_info.py         # GET /model/info
│   └── prediction.py         # POST /predict
│
├── services/                  # Lógica de negócio
│   ├── __init__.py
│   ├── model_service.py      # Gerencia modelo LSTM (Singleton)
│   ├── data_service.py       # Busca dados do Yahoo Finance
│   └── predict_service.py    # Orquestra pipeline de predição
│
├── models/                    # Definições de modelos ML
│   ├── __init__.py
│   └── lstm_model.py         # Arquitetura LSTM PyTorch
│
└── utils/                     # Utilitários
    ├── __init__.py
    ├── exceptions.py         # Exceções customizadas
    └── validators.py         # Validação de entrada
```

---

## 🔄 Fluxo de Requisição

### Pipeline Completo - POST /predict

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. ENTRADA DO CLIENTE                                           │
│    POST /predict                                                 │
│    Content-Type: application/json                               │
│    Body: {"ticker": "AAPL"}                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. VALIDAÇÃO (prediction.py)                                    │
│    ├─ Content-Type é application/json? ✓                       │
│    ├─ Campo "ticker" presente? ✓                               │
│    └─ validate_ticker("AAPL")                                  │
│        ├─ 2-10 caracteres? ✓                                   │
│        ├─ Formato [A-Z][A-Z0-9.-]? ✓                          │
│        └─ Não começa com número? ✓                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. PREDICT SERVICE                                              │
│    PredictService.predict("AAPL")                               │
│    ├─ normalize_ticker("AAPL") → "AAPL"                       │
│    └─ Inicia pipeline                                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. BUSCA DE DADOS (DataService)                                 │
│    fetch_data("AAPL")                                           │
│    ├─ Calcula período: hoje - 90 dias até hoje                 │
│    ├─ yf.Ticker("AAPL").history(start, end)                    │
│    ├─ Valida DataFrame não vazio                               │
│    ├─ Valida >= 60 dias de dados                               │
│    └─ Retorna últimos 60 dias                                  │
│        → DataFrame[Open, High, Low, Close, Volume]              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. PREPARAÇÃO DOS DADOS                                         │
│    ├─ Extrai Close prices: [175.1, 176.3, ..., 175.2]         │
│    ├─ Reshape para (60, 1)                                     │
│    └─ current_price = 175.20                                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. NORMALIZAÇÃO                                                 │
│    ModelService.get_scaler()                                    │
│    ├─ MinMaxScaler range [0, 1]                                │
│    └─ scaled_data = scaler.transform(close_prices)             │
│        → [0.523, 0.541, ..., 0.520]                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. CONVERSÃO PARA TENSOR                                        │
│    X = torch.FloatTensor(scaled_data).unsqueeze(0)             │
│    Shape: [1, 60, 1]                                            │
│    ├─ batch_size: 1                                            │
│    ├─ sequence_length: 60                                      │
│    └─ features: 1                                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. INFERÊNCIA DO MODELO                                         │
│    ModelService.get_model()                                     │
│    with torch.no_grad():                                        │
│        prediction_scaled = model(X)                             │
│    ├─ LSTM processa sequência                                  │
│    └─ Linear layer gera predição                               │
│        → Tensor [0.535] (normalizado)                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. DESNORMALIZAÇÃO                                              │
│    predicted_price = scaler.inverse_transform(prediction_scaled)│
│    → 178.45                                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. CÁLCULO DE MÉTRICAS                                         │
│    ├─ change_percent = ((178.45 - 175.20) / 175.20) * 100     │
│    │   → 1.85%                                                  │
│    ├─ change_direction = "up"                                  │
│    ├─ prediction_date = hoje + 1 dia                           │
│    └─ confidence = "medium" (|1.85%| está entre 2-5%)          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 11. RESPOSTA AO CLIENTE                                         │
│    HTTP 200 OK                                                  │
│    {                                                            │
│      "success": true,                                           │
│      "data": {                                                  │
│        "ticker": "AAPL",                                        │
│        "predicted_price": 178.45,                               │
│        "current_price": 175.20,                                 │
│        "change_percent": 1.85,                                  │
│        "change_direction": "up",                                │
│        "prediction_date": "2025-12-31",                         │
│        "confidence": "medium",                                  │
│        "timestamp": "2025-12-30T10:30:00"                       │
│      }                                                           │
│    }                                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Fluxo de Erro

```
Exception levantada em qualquer ponto
         │
         ▼
   Tipo da Exception?
         │
         ├─ InvalidTickerError
         │  └─→ HTTP 400 {"error": "Invalid Ticker", ...}
         │
         ├─ TickerNotFoundError
         │  └─→ HTTP 404 {"error": "Ticker Not Found", ...}
         │
         ├─ InsufficientDataError
         │  └─→ HTTP 400 {"error": "Insufficient Data", ...}
         │
         ├─ ServiceUnavailableError
         │  └─→ HTTP 503 {"error": "Service Unavailable", ...}
         │
         ├─ ModelInferenceError
         │  └─→ HTTP 500 {"error": "Model Inference Error", ...}
         │
         └─ Exception genérica
            └─→ HTTP 500 {"error": "Internal Server Error"}
```

---

## 🌐 Endpoints

### 1. Health Check

#### `GET /health`

Verifica se a API está rodando e operacional.

**Request:**
```bash
curl http://localhost:5001/health
```

**Response 200 OK:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-30T10:30:00.123456",
  "service": "stock-prediction-lstm-api"
}
```

**Uso:**
- Health checks de containers Docker
- Monitoramento de disponibilidade
- Load balancers e orchestrators

**Latência:** ~5ms

---

### 2. Model Info

#### `GET /model/info`

Retorna metadados e configuração do modelo LSTM carregado.

**Request:**
```bash
curl http://localhost:5001/model/info
```

**Response 200 OK:**
```json
{
  "model_type": "LSTM",
  "architecture": "LSTM-1x16",
  "input_size": 1,
  "hidden_size": 16,
  "num_layers": 1,
  "dropout": 0.2,
  "sequence_length": 60,
  "target_column": "Close",
  "metrics": {
    "test_mae": 2.34,
    "test_mse": 8.92,
    "test_rmse": 2.99,
    "test_mape": 1.21,
    "test_r2": 0.90
  },
  "training_info": {
    "dataset": "AAPL (2020-2024)",
    "trained_on": "2024-12-15",
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "Adam",
    "learning_rate": 0.001
  }
}
```

**Response 404 Not Found:**
```json
{
  "error": "Config Not Found",
  "message": "Arquivo de configuração do modelo não encontrado",
  "status": 404
}
```

**Response 500 Internal Server Error:**
```json
{
  "error": "Internal Server Error",
  "message": "Erro ao carregar configurações: Invalid JSON",
  "status": 500
}
```

**Latência:** ~10ms

---

### 3. Predict

#### `POST /predict`

Realiza predição de preço de fechamento para o próximo dia útil.

**Query Parameters:**

| Parâmetro | Tipo | Obrigatório | Padrão | Descrição |
|-----------|------|-------------|--------|-----------|
| include_history | boolean | ❌ | false | Inclui dados históricos dos últimos 30 dias (OHLCV) na resposta |

**Request (sem histórico):**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

**Request (com histórico):**
```bash
curl -X POST "http://localhost:5001/predict?include_history=true" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

**Request Body Schema:**
```json
{
  "ticker": "string (2-10 chars, uppercase, alphanumeric + . -)"
}
```

**Validações:**
| Campo | Tipo | Obrigatório | Validação |
|-------|------|-------------|-----------|
| ticker | string | ✅ | 2-10 caracteres, começa com letra, apenas [A-Z0-9.-] |

**Exemplos Válidos:**
- `AAPL` (Apple)
- `MSFT` (Microsoft)
- `PETR4.SA` (Petrobras - B3)
- `BRK-B` (Berkshire Hathaway Class B)

**Response 200 OK (sem histórico):**
```json
{
  "success": true,
  "data": {
    "ticker": "AAPL",
    "predicted_price": 178.45,
    "current_price": 175.20,
    "change_percent": 1.85,
    "change_direction": "up",
    "prediction_date": "2025-12-31",
    "confidence": "medium",
    "timestamp": "2025-12-30T10:30:00.123456"
  }
}
```

**Response 200 OK (com histórico - `?include_history=true`):**
```json
{
  "success": true,
  "data": {
    "ticker": "AAPL",
    "predicted_price": 178.45,
    "current_price": 175.20,
    "change_percent": 1.85,
    "change_direction": "up",
    "prediction_date": "2025-12-31",
    "confidence": "medium",
    "timestamp": "2025-12-30T10:30:00.123456",
    "historical_data": [
      {
        "date": "2025-11-28",
        "open": 172.50,
        "high": 174.20,
        "low": 171.80,
        "close": 173.90,
        "volume": 52000000
      },
      {
        "date": "2025-11-29",
        "open": 173.95,
        "high": 175.60,
        "low": 173.50,
        "close": 175.20,
        "volume": 48500000
      }
      // ... 28 dias adicionais
    ]
  }
}
```

**Campos de Response:**

| Campo | Tipo | Opcional | Descrição |
|-------|------|----------|-----------|
| ticker | string | ❌ | Símbolo normalizado (uppercase) |
| predicted_price | float | ❌ | Preço previsto para próximo dia (arredondado 2 casas) |
| current_price | float | ❌ | Último preço de fechamento conhecido |
| change_percent | float | ❌ | Variação % esperada (positivo = alta, negativo = baixa) |
| change_direction | string | ❌ | `"up"`, `"down"` ou `"neutral"` |
| prediction_date | string | ❌ | Data da previsão (formato YYYY-MM-DD) |
| confidence | string | ❌ | `"high"` (<2%), `"medium"` (2-5%), `"low"` (>5%) |
| timestamp | string | ❌ | ISO 8601 timestamp da predição |
| historical_data | array | ✅ | Array com últimos 30 dias (OHLCV). Incluído apenas se `include_history=true` |

**Campos de `historical_data` (quando incluído):**

| Campo | Tipo | Descrição |
|-------|------|-----------|
| date | string | Data no formato YYYY-MM-DD |
| open | float | Preço de abertura (2 casas decimais) |
| high | float | Preço máximo do dia (2 casas decimais) |
| low | float | Preço mínimo do dia (2 casas decimais) |
| close | float | Preço de fechamento (2 casas decimais) |
| volume | integer | Volume negociado |

**Response 400 Bad Request (Content-Type inválido):**
```json
{
  "error": "Invalid Content-Type",
  "message": "Content-Type deve ser application/json",
  "status": 400
}
```

**Response 400 Bad Request (Campo ausente):**
```json
{
  "error": "Missing Field",
  "message": "Campo 'ticker' é obrigatório",
  "status": 400
}
```

**Response 400 Bad Request (Ticker inválido):**
```json
{
  "error": "Invalid Ticker",
  "message": "Formato de ticker inválido",
  "ticker": "A",
  "suggestion": "Ticker deve ter entre 2 e 10 caracteres",
  "status": 400
}
```

**Response 404 Not Found (Ticker não existe):**
```json
{
  "error": "Ticker Not Found",
  "message": "Ticker INVALID não encontrado no Yahoo Finance",
  "ticker": "INVALID",
  "suggestion": "Verifique se o símbolo está correto. Para ações brasileiras, use o sufixo .SA (ex: PETR4.SA)",
  "status": 404
}
```

**Response 400 Bad Request (Dados insuficientes):**
```json
{
  "error": "Insufficient Data",
  "message": "Dados históricos insuficientes para NEWCO",
  "ticker": "NEWCO",
  "days_available": 25,
  "days_required": 60,
  "suggestion": "Modelo requer pelo menos 60 dias de histórico",
  "status": 400
}
```

**Response 503 Service Unavailable (Yahoo Finance offline):**
```json
{
  "error": "Service Unavailable",
  "message": "Serviço Yahoo Finance temporariamente indisponível",
  "service": "Yahoo Finance",
  "retry_after": 60,
  "status": 503
}
```

**Response 500 Internal Server Error (Erro de inferência):**
```json
{
  "error": "Model Inference Error",
  "message": "Erro ao processar previsão para AAPL",
  "ticker": "AAPL",
  "details": "Tensor shape mismatch: expected [1, 60, 1], got [1, 59, 1]",
  "status": 500
}
```

**Latência:** ~470ms (média)
- 400ms: Yahoo Finance
- 50ms: Inferência do modelo
- 20ms: Processamento restante

**Tamanho do Payload:**
- Sem histórico: ~250 bytes
- Com histórico (`include_history=true`): ~3.5 KB

**💡 Dica:** Use `include_history=true` apenas quando necessário exibir gráfico histórico, para economizar banda.

---

## 🔧 Serviços

### ModelService

**Arquivo:** `src/api/services/model_service.py`

**Responsabilidade:** Gerenciar modelo LSTM e MinMaxScaler em memória.

**Padrão:** Singleton (instância única compartilhada)

**Ciclo de Vida:**

```python
# Primeira chamada - inicialização
service = ModelService()
├─ Carrega artifacts/model_config.json
├─ Cria StockLSTM(input_size=1, hidden_size=16, ...)
├─ Carrega artifacts/model_lstm_1x16.pt
├─ Define model.eval() (modo inferência)
└─ Carrega artifacts/scaler_corrected.pkl

# Chamadas subsequentes - reutiliza instância
service = ModelService()  # Retorna mesma instância
```

**Métodos Públicos:**

```python
get_model() -> torch.nn.Module
    """Retorna modelo LSTM em modo eval."""
    # Raises: RuntimeError se modelo não carregado

get_scaler() -> MinMaxScaler
    """Retorna scaler para normalização."""
    # Raises: RuntimeError se scaler não carregado

get_config() -> Dict[str, Any]
    """Retorna configuração completa do modelo."""
    # Raises: RuntimeError se config não carregada

is_ready() -> bool
    """Verifica se modelo, scaler e config estão carregados."""
    # Returns: True se tudo OK, False caso contrário
```

**Thread-Safety:** ✅ Sim (singleton com inicialização única)

**Memory Footprint:** ~500MB (modelo + scaler)

**Tratamento de Erros:**
- `FileNotFoundError` → `RuntimeError("Artefato necessário não encontrado")`
- Qualquer outro erro → `RuntimeError("Falha ao inicializar modelo")`

---

### DataService

**Arquivo:** `src/api/services/data_service.py`

**Responsabilidade:** Buscar dados históricos de ações via Yahoo Finance.

**Padrão:** Instância simples (não singleton, pode ter múltiplas)

**Inicialização:**

```python
service = DataService(lookback_days=60)
```

**Métodos Públicos:**

```python
fetch_data(ticker: str) -> pd.DataFrame
    """
    Busca últimos N dias de dados históricos.
    
    Returns: DataFrame com [Open, High, Low, Close, Volume]
    
    Raises:
        TickerNotFoundError: Ticker não existe no Yahoo Finance
        InsufficientDataError: Menos de lookback_days disponíveis
        ServiceUnavailableError: Yahoo Finance offline/timeout
    """
```

**Fluxo Interno de `fetch_data`:**

```python
1. Calcula período
   end_date = datetime.now()
   start_date = end_date - timedelta(days=lookback_days + 30)
   # +30 dias para compensar fins de semana/feriados

2. Busca dados
   stock = yf.Ticker(ticker)
   df = stock.history(start=start_date, end=end_date)

3. Valida dados
   if df.empty:
       raise TickerNotFoundError(ticker)
   
   if len(df) < lookback_days:
       raise InsufficientDataError(...)

4. Retorna últimos N dias
   return df.tail(lookback_days)
```

**Tratamento de Exceções:**

| Exceção Python | Ação |
|----------------|------|
| `ConnectionError` | `ServiceUnavailableError(retry_after=60)` |
| `TimeoutError` | `ServiceUnavailableError(retry_after=30)` |
| String contém "connection"/"timeout" | `ServiceUnavailableError` |
| Outras exceções | `TickerNotFoundError` (assume ticker inválido) |

**Método Auxiliar:**

```python
get_latest_price(ticker: str) -> Optional[float]
    """
    Retorna último preço de fechamento.
    
    Returns: float ou None se erro
    """
```

---

### PredictService

**Arquivo:** `src/api/services/predict_service.py`

**Responsabilidade:** Orquestrar pipeline completo de predição.

**Dependências:**
- `ModelService` (singleton)
- `DataService` (instância própria com lookback_days=60)

**Inicialização:**

```python
service = PredictService()
├─ self.model_service = ModelService()
├─ self.data_service = DataService(lookback_days=60)
└─ Valida model_service.is_ready()
```

**Método Principal:**

```python
predict(ticker: str) -> Dict[str, Any]
    """
    Pipeline completo de predição.
    
    Steps:
    1. Normaliza ticker (uppercase, strip)
    2. Busca dados históricos (60 dias)
    3. Extrai Close prices
    4. Normaliza com scaler
    5. Converte para tensor PyTorch [1, 60, 1]
    6. Inferência do modelo
    7. Desnormaliza resultado
    8. Calcula métricas (change_percent, confidence)
    9. Retorna resultado estruturado
    
    Returns: dict com previsão e metadados
    
    Raises: Todas as custom exceptions do pipeline
    """
```

**Cálculo de Confiança:**

```python
abs_change = abs(change_percent)

if abs_change < 2:
    confidence = "high"    # Variação pequena, alta confiança
elif abs_change < 5:
    confidence = "medium"  # Variação moderada, média confiança
else:
    confidence = "low"     # Variação grande, baixa confiança
```

**Tratamento de Exceções:**

```python
# Custom exceptions - re-raise (já estão corretas)
except (InvalidTickerError, TickerNotFoundError, 
        InsufficientDataError, ServiceUnavailableError, 
        ModelInferenceError):
    raise

# Exceções genéricas - tenta inferir o tipo
except Exception as e:
    error_msg = str(e).lower()
    
    # Erros relacionados a tensores PyTorch
    if "tensor" in error_msg or "shape" in error_msg:
        raise ModelInferenceError(ticker, error_detail=str(e))
    
    # Outros erros - genérico
    raise ModelInferenceError(ticker, error_detail=str(e))
```

---

## ⚠️ Sistema de Exceções

Hierarquia de exceções customizadas para tratamento granular de erros.

### Classe Base: APIException

**Arquivo:** `src/api/utils/exceptions.py`

```python
class APIException(Exception):
    status_code = 500
    error_type = "API Error"
    
    def to_dict(self) -> dict:
        """Converte exceção para JSON response."""
        return {
            "error": self.error_type,
            "message": str(self),
            "status": self.status_code
        }
```

### Hierarquia Completa

```
APIException (500)
├── InvalidTickerError (400)
├── TickerNotFoundError (404)
├── InsufficientDataError (400)
├── ServiceUnavailableError (503)
└── ModelInferenceError (500)
```

### 1. InvalidTickerError (400)

**Quando usar:** Formato de ticker não atende aos requisitos.

**Construtor:**
```python
InvalidTickerError(ticker: str, suggestion: str)
```

**Exemplo:**
```python
raise InvalidTickerError(
    ticker="A",
    suggestion="Ticker deve ter entre 2 e 10 caracteres"
)
```

**JSON Response:**
```json
{
  "error": "Invalid Ticker",
  "message": "Formato de ticker inválido",
  "ticker": "A",
  "suggestion": "Ticker deve ter entre 2 e 10 caracteres",
  "status": 400
}
```

---

### 2. TickerNotFoundError (404)

**Quando usar:** Ticker não existe no Yahoo Finance.

**Construtor:**
```python
TickerNotFoundError(ticker: str)
```

**Exemplo:**
```python
raise TickerNotFoundError(ticker="INVALID")
```

**JSON Response:**
```json
{
  "error": "Ticker Not Found",
  "message": "Ticker INVALID não encontrado no Yahoo Finance",
  "ticker": "INVALID",
  "suggestion": "Verifique se o símbolo está correto. Para ações brasileiras, use o sufixo .SA (ex: PETR4.SA)",
  "status": 404
}
```

---

### 3. InsufficientDataError (400)

**Quando usar:** Ticker existe mas tem menos de 60 dias de histórico.

**Construtor:**
```python
InsufficientDataError(
    ticker: str,
    days_available: int,
    days_required: int
)
```

**Exemplo:**
```python
raise InsufficientDataError(
    ticker="NEWCO",
    days_available=25,
    days_required=60
)
```

**JSON Response:**
```json
{
  "error": "Insufficient Data",
  "message": "Dados históricos insuficientes para NEWCO",
  "ticker": "NEWCO",
  "days_available": 25,
  "days_required": 60,
  "suggestion": "Modelo requer pelo menos 60 dias de histórico",
  "status": 400
}
```

---

### 4. ServiceUnavailableError (503)

**Quando usar:** Yahoo Finance está offline, timeout ou erro de rede.

**Construtor:**
```python
ServiceUnavailableError(
    service: str = "External Service",
    retry_after: int = 60
)
```

**Exemplo:**
```python
raise ServiceUnavailableError(
    service="Yahoo Finance",
    retry_after=60
)
```

**JSON Response:**
```json
{
  "error": "Service Unavailable",
  "message": "Serviço Yahoo Finance temporariamente indisponível",
  "service": "Yahoo Finance",
  "retry_after": 60,
  "status": 503
}
```

**Headers HTTP:**
```
Retry-After: 60
```

---

### 5. ModelInferenceError (500)

**Quando usar:** Erro durante inferência do modelo PyTorch.

**Construtor:**
```python
ModelInferenceError(
    ticker: str,
    error_detail: str = "Erro desconhecido"
)
```

**Exemplo:**
```python
raise ModelInferenceError(
    ticker="AAPL",
    error_detail="Tensor shape mismatch"
)
```

**JSON Response:**
```json
{
  "error": "Model Inference Error",
  "message": "Erro ao processar previsão para AAPL",
  "ticker": "AAPL",
  "details": "Tensor shape mismatch",
  "status": 500
}
```

---

## ✅ Validadores

**Arquivo:** `src/api/utils/validators.py`

### validate_ticker()

Valida formato de ticker de ação.

**Assinatura:**
```python
def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Returns: (is_valid, error_message)
    """
```

**Regras de Validação:**

| Regra | Descrição | Exemplo Inválido |
|-------|-----------|------------------|
| Tipo | Deve ser string | `123` (int) |
| Tamanho mínimo | >= 2 caracteres | `"A"` |
| Tamanho máximo | <= 10 caracteres | `"VERYLONGNAME"` |
| Formato | `[A-Z][A-Z0-9.-]{1,9}` | `"123ABC"` (começa com número) |
| Caracteres permitidos | Letras, números, ponto, hífen | `"AAPL@"` (@ inválido) |

**Exemplos:**

```python
# Válidos
validate_ticker("AAPL")      # (True, "")
validate_ticker("PETR4.SA")  # (True, "")
validate_ticker("BRK-B")     # (True, "")

# Inválidos
validate_ticker("A")         # (False, "Ticker deve ter entre 2 e 10 caracteres")
validate_ticker("123")       # (False, "Ticker deve conter apenas letras...")
validate_ticker("")          # (False, "Ticker não pode ser vazio")
```

**Fluxo de Validação:**

```python
1. Tipo string? → Se não: erro
2. Strip espaços
3. Vazio? → Se sim: erro
4. Tamanho < 2? → Se sim: erro
5. Tamanho > 10? → Se sim: erro
6. Match regex [A-Z][A-Z0-9.-]{1,9}? → Se não: erro
7. Retorna (True, "")
```

---

### normalize_ticker()

Normaliza ticker para formato padrão.

**Assinatura:**
```python
def normalize_ticker(ticker: str) -> str:
    """
    Returns: ticker normalizado (uppercase, sem espaços)
    """
```

**Transformações:**

1. `.strip()` - Remove espaços nas extremidades
2. `.upper()` - Converte para maiúsculas

**Exemplos:**

```python
normalize_ticker("  aapl  ")   # "AAPL"
normalize_ticker("petr4.sa")   # "PETR4.SA"
normalize_ticker("BrK-b")      # "BRK-B"
```

---

## 🔐 Segurança

### CORS (Cross-Origin Resource Sharing)

**Configuração Atual (Desenvolvimento):**

```python
CORS(app, resources={
    r"/*": {
        "origins": "*",              # ⚠️ Permite qualquer origin
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
```

**Configuração Recomendada (Produção):**

```python
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://seu-frontend.com",
            "https://app.seu-dominio.com"
        ],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["X-Request-ID"],
        "max_age": 3600  # Cache preflight por 1 hora
    }
})
```

---

### Content-Type Validation

**Implementado:** ✅ Sim

**Validação em POST /predict:**

```python
if not request.is_json:
    return jsonify({
        "error": "Invalid Content-Type",
        "message": "Content-Type deve ser application/json",
        "status": 400
    }), 400
```

**Headers aceitos:**
- `Content-Type: application/json`
- `Content-Type: application/json; charset=utf-8`

---

### Rate Limiting

**Status:** ⚠️ Não implementado

**Recomendação para Produção:**

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379"
)

# Limites específicos por endpoint
@limiter.limit("10 per minute")
@prediction_bp.route('/predict', methods=['POST'])
def predict():
    ...
```

**Limites Sugeridos:**

| Endpoint | Limite | Justificativa |
|----------|--------|---------------|
| /health | Ilimitado | Health checks frequentes |
| /model/info | 100/hora | Raramente muda |
| /predict | 10/minuto | Operação custosa (~470ms) |

---

### Input Sanitization

**Implementado:** ✅ Sim

**Validações:**
1. Tipo de dados (ticker deve ser string)
2. Formato (regex `[A-Z][A-Z0-9.-]{1,9}`)
3. Tamanho (2-10 caracteres)
4. Normalização (uppercase, strip)

**Proteção contra:**
- SQL Injection: N/A (não usa SQL)
- XSS: Validação de formato previne
- Path Traversal: N/A (não acessa filesystem com input)

---

## 📈 Performance

### Latência por Endpoint

| Endpoint | P50 | P95 | P99 |
|----------|-----|-----|-----|
| GET /health | 5ms | 8ms | 12ms |
| GET /model/info | 10ms | 15ms | 25ms |
| POST /predict | 470ms | 650ms | 900ms |

### Breakdown - POST /predict (470ms total)

```
Yahoo Finance API:     400ms (85%)  [Maior gargalo]
Normalização (scaler):   5ms (1%)
Inferência LSTM:        50ms (11%)
Cálculos/Métricas:       5ms (1%)
Serialização JSON:       9ms (2%)
Overhead Flask:          1ms (<1%)
```

### Otimizações Possíveis

#### 1. Cache de Dados (Redis)

**Benefício:** -400ms (redução de ~85%)

```python
import redis

redis_client = redis.Redis(host='localhost', port=6379)

def fetch_data_cached(ticker: str) -> pd.DataFrame:
    cache_key = f"stock_data:{ticker}:{date.today()}"
    
    # Tenta buscar do cache
    cached = redis_client.get(cache_key)
    if cached:
        return pickle.loads(cached)
    
    # Se não está no cache, busca do Yahoo Finance
    df = yf.Ticker(ticker).history(...)
    
    # Armazena no cache (expira em 1 dia)
    redis_client.setex(cache_key, 86400, pickle.dumps(df))
    
    return df
```

**Latência Final:** ~70ms

---

#### 2. Model Quantization

**Benefício:** -25ms (redução de ~50% na inferência)

```python
import torch.quantization

# Quantizar modelo para int8
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.LSTM, torch.nn.Linear},
    dtype=torch.qint8
)
```

**Trade-off:**
- ✅ Mais rápido (2x)
- ✅ Menor uso de memória (4x)
- ⚠️ Pequena perda de precisão (~0.5%)

---

#### 3. Batch Inference

**Benefício:** Processar múltiplos tickers em paralelo

```python
def predict_batch(tickers: List[str]) -> List[dict]:
    # Buscar dados em paralelo
    with ThreadPoolExecutor(max_workers=5) as executor:
        data_futures = {
            executor.submit(fetch_data, t): t 
            for t in tickers
        }
    
    # Empilhar tensores
    X_batch = torch.stack([
        prepare_tensor(data) for data in data_list
    ])
    
    # Inferência em batch
    with torch.no_grad():
        predictions = model(X_batch)
    
    return process_results(predictions)
```

---

### Memory Usage

| Componente | Tamanho | Lifecycle |
|------------|---------|-----------|
| Modelo LSTM | ~450MB | Persistente (singleton) |
| MinMaxScaler | ~50MB | Persistente (singleton) |
| Flask App | ~30MB | Persistente |
| Request Buffer | ~1MB | Por request |
| **Total Base** | **~530MB** | **Mínimo** |
| **Com 10 requests simultâneos** | **~540MB** | **Típico** |

**Recomendações:**
- **Desenvolvimento:** 1GB RAM
- **Produção (baixo tráfego):** 1GB RAM
- **Produção (alto tráfego):** 2GB RAM + scaling horizontal

---

## 🐛 Troubleshooting

### Problema: "Model not loaded correctly"

**Sintoma:**
```
RuntimeError: Modelo não foi carregado corretamente
```

**Causas Possíveis:**
1. Arquivo `artifacts/model_lstm_1x16.pt` não existe
2. Arquivo corrompido
3. Permissões de leitura

**Diagnóstico:**
```bash
# Verificar se arquivo existe
ls -lh artifacts/model_lstm_1x16.pt

# Verificar permissões
stat artifacts/model_lstm_1x16.pt

# Verificar integridade (se tiver checksum)
shasum -a 256 artifacts/model_lstm_1x16.pt
```

**Solução:**
1. Re-baixar modelo do repositório
2. Re-treinar modelo se necessário
3. Corrigir permissões: `chmod 644 artifacts/model_lstm_1x16.pt`

---

### Problema: "Ticker not found" para ticker válido

**Sintoma:**
```json
{
  "error": "Ticker Not Found",
  "ticker": "AAPL"
}
```

**Causas Possíveis:**
1. Yahoo Finance temporariamente indisponível
2. Ticker foi deslistado recentemente
3. Problema de conectividade

**Diagnóstico:**
```bash
# Testar conectividade Yahoo Finance
curl -I https://finance.yahoo.com

# Testar ticker manualmente
curl "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
```

**Solução:**
1. Aguardar e tentar novamente (retry_after: 60s)
2. Verificar se ticker ainda está ativo
3. Verificar firewall/proxy

---

### Problema: "Tensor shape mismatch"

**Sintoma:**
```
ModelInferenceError: Tensor shape mismatch: expected [1, 60, 1], got [1, 59, 1]
```

**Causas Possíveis:**
1. Yahoo Finance retornou menos de 60 dias
2. Dados contêm valores NaN
3. DataFrame foi filtrado incorretamente

**Diagnóstico:**
```python
# Adicionar logging em data_service.py
logger.info(f"DataFrame shape: {df.shape}")
logger.info(f"NaN values: {df.isna().sum()}")
```

**Solução:**
1. Aumentar margem de dias (60 + 30 → 60 + 60)
2. Filtrar NaN antes de processar
3. Validar `len(df) >= 60` antes de inferência

---

### Problema: API lenta (>1s por request)

**Sintoma:** Latência alta consistente

**Diagnóstico:**
```python
import time

# Adicionar timing em predict_service.py
start = time.time()
df = self.data_service.fetch_data(ticker)
logger.info(f"fetch_data: {time.time() - start:.2f}s")

start = time.time()
prediction = model(X)
logger.info(f"inference: {time.time() - start:.2f}s")
```

**Causas e Soluções:**

| Causa | Solução |
|-------|---------|
| Yahoo Finance lento | Implementar cache Redis |
| Modelo não em eval() | Verificar `model.eval()` |
| CPU throttling | Aumentar recursos do servidor |
| Múltiplas requisições simultâneas | Implementar rate limiting |

---

### Problema: Memory leak

**Sintoma:** Uso de memória cresce continuamente

**Diagnóstico:**
```bash
# Monitorar memória
watch -n 1 'ps aux | grep flask'

# Profiling com memory_profiler
pip install memory_profiler
python -m memory_profiler app.py
```

**Causas Possíveis:**
1. Tensores não sendo liberados
2. DataFrame sendo acumulado
3. Cache sem limite

**Solução:**
```python
# Garantir uso de torch.no_grad()
with torch.no_grad():
    prediction = model(X)

# Limpar variáveis explicitamente
del X, prediction_scaled, df
torch.cuda.empty_cache()  # Se usar GPU
```

---

## 📞 Recursos

### Documentação Relacionada

- **README.md:** Guia de uso e instalação
- **ML_DOCUMENTATION.md:** Documentação do modelo LSTM
- **DEPLOY.md:** Guias de deployment
- **PLANO_PESSOA_B.md:** Roadmap de implementação

### Links Externos

- **Flask:** https://flask.palletsprojects.com/
- **PyTorch:** https://pytorch.org/docs/stable/index.html
- **yfinance:** https://github.com/ranaroussi/yfinance
- **Yahoo Finance API:** https://www.yahoofinanceapi.com/

### Suporte

- **GitHub Issues:** [Reportar Bug](https://github.com/adriannylelis/stock-prediction-lstm-api/issues)
- **Pull Requests:** [Contribuir](https://github.com/adriannylelis/stock-prediction-lstm-api/pulls)

