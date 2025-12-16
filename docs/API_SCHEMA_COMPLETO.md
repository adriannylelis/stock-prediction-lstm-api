# 📡 API Schema - Comunicação entre Pessoa A e Pessoa B

**Projeto:** Stock Prediction LSTM API  
**Fase:** Handover Pessoa A → Pessoa B  
**Data:** 15/12/2025  
**Objetivo:** Documentar schema completo da API para implementação do backend

---

## 🎯 Visão Geral

A API receberá apenas o **ticker da ação** (ex: PETR4.SA), buscará automaticamente os **últimos 60 dias** do Yahoo Finance, calculará os indicadores técnicos e retornará a **previsão do preço de fechamento** para o próximo dia.

### Arquitetura
```
┌─────────────┐    POST {"ticker"}    ┌──────────────┐   yfinance   ┌─────────────┐
│   Frontend  │  ───────────────────> │  Flask/      │  ──────────> │ Yahoo       │
│  (Pessoa B) │                        │  FastAPI     │  <──────────  │ Finance     │
└─────────────┘                        └──────────────┘   60 days    └─────────────┘
       ▲                                      │
       │                                      │ Calculate Indicators
       │                                      │ (pandas-ta)
       │                                      ▼
       │              ┌──────────────┐   ┌──────────────┐
       │              │ Artifacts/   │   │ Technical    │
       │              │ - model.pt   │   │ Indicators   │
       │              │ - scaler.pkl │   │ + Features   │
       │              └──────────────┘   └──────────────┘
       │                     │                   │
       │                     └───────┬───────────┘
       │                             ▼
       │                      ┌──────────────┐
       └───── JSON ─────────  │ LSTM Model   │
            Response          │ Prediction   │
                              └──────────────┘
```

---

## 📋 Endpoints da API

### 1. Health Check
**GET** `/health`

**Descrição:** Verifica se a API está funcionando e se o modelo está carregado.

**Response 200:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-15T10:30:00Z",
  "version": "1.0.0"
}
```

---

### 2. Informações do Modelo
**GET** `/model/info`

**Descrição:** Retorna metadados sobre o modelo treinado.

**Response 200:**
```json
{
  "model_name": "StockLSTM",
  "ticker": "PETR4.SA",
  "training_period": "2020-01-01 to 2024-12-15",
  "lookback_days": 60,
  "features_count": 20,
  "features_list": [
    "Open", "High", "Low", "Close", "Volume", "Adj Close",
    "SMA_20", "SMA_50", "SMA_200",
    "EMA_12", "EMA_26",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_middle", "BB_lower",
    "ATR_14", "Volume"
  ],
  "architecture": {
    "input_size": 20,
    "hidden_size": 50,
    "num_layers": 2,
    "dropout": 0.2
  },
  "metrics": {
    "mae": 0.85,
    "rmse": 1.12,
    "mape": 3.45
  },
  "last_training_date": "2025-12-15T08:00:00Z"
}
```

---

### 3. Previsão de Preço (Principal)
**POST** `/predict`

**Descrição:** Recebe ticker da ação, busca automaticamente os últimos 60 dias e retorna previsão do próximo dia.

#### Request Body:
```json
{
  "ticker": "PETR4.SA"
}
```

**Opcional - Múltiplos Tickers:**
```json
{
  "tickers": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
}
```

#### Response 200 (Ticker Único):
```json
{
  "success": true,
  "ticker": "PETR4.SA",
  "prediction": {
    "next_day_close": 39.15,
    "confidence_interval": {
      "lower": 38.50,
      "upper": 39.80
    },
    "prediction_date": "2025-12-16",
    "last_known_close": 38.90,
    "last_known_date": "2025-12-15",
    "change_percent": 0.64
  },
  "data_info": {
    "days_used": 60,
    "date_range": {
      "start": "2025-10-01",
      "end": "2025-12-15"
    },
    "data_source": "Yahoo Finance"
  },
  "metadata": {
    "model_version": "1.0.0",
    "prediction_timestamp": "2025-12-15T10:35:00Z",
    "processing_time_ms": 245
  }
}
```

#### Response 200 (Múltiplos Tickers):
```json
{
  "success": true,
  "predictions": [
    {
      "ticker": "PETR4.SA",
      "prediction": {
        "next_day_close": 39.15,
        "confidence_interval": {
          "lower": 38.50,
          "upper": 39.80
        },
        "prediction_date": "2025-12-16",
        "last_known_close": 38.90,
        "change_percent": 0.64
      }
    },
    {
      "ticker": "VALE3.SA",
      "prediction": {
        "next_day_close": 62.80,
        "confidence_interval": {
          "lower": 61.50,
          "upper": 64.10
        },
        "prediction_date": "2025-12-16",
        "last_known_close": 62.35,
        "change_percent": 0.72
      }
    }
  ],
  "metadata": {
    "model_version": "1.0.0",
    "prediction_timestamp": "2025-12-15T10:35:00Z",
    "total_predictions": 2
  }
}
```

#### Response 400 (Erro de Validação):
```json
{
  "success": false,
  "error": "invalid_ticker",
  "message": "Ticker inválido ou não encontrado",
  "details": {
    "ticker": "INVALID",
    "suggestion": "Use formato correto: PETR4.SA, VALE3.SA"
  }
}
```

#### Response 400 (Dados Insuficientes):
```json
{
  "success": false,
  "error": "insufficient_data",
  "message": "Não há dados suficientes para previsão",
  "details": {
    "ticker": "PETR4.SA",
    "days_available": 45,
    "days_required": 60,
    "message": "Ação muito recente ou dados incompletos"
  }
}
```

#### Response 500 (Erro Interno):
```json
{
  "success": false,
  "error": "prediction_error",
  "message": "Erro ao processar previsão",
  "details": "Model inference failed: tensor dimension mismatch"
}
```

#### Response 503 (Serviço Indisponível):
```json
{
  "success": false,
  "error": "service_unavailable",
  "message": "Yahoo Finance indisponível no momento",
  "details": "Timeout ao buscar dados. Tente novamente em instantes."
}
```

---

### 4. Previsão Detalhada (Opcional)
**POST** `/predict/detailed`

**Descrição:** Retorna previsão + análise técnica detalhada + gráficos dos últimos 60 dias.

#### Request Body:
```json
{
  "ticker": "PETR4.SA",
  "include_chart_data": true
}
```

#### Response 200:
```json
{
  "success": true,
  "ticker": "PETR4.SA",
  "prediction": {
    "next_day_close": 39.15,
    "confidence_interval": {
      "lower": 38.50,
      "upper": 39.80
    },
    "prediction_date": "2025-12-16",
    "last_known_close": 38.90,
    "change_percent": 0.64
  },
  "technical_analysis": {
    "trend": "bullish",
    "trend_strength": "moderate",
    "rsi_signal": "neutral",
    "rsi_value": 58.2,
    "macd_signal": "buy",
    "macd_value": 0.48,
    "bb_position": "middle",
    "volatility": "moderate",
    "atr_value": 1.30
  },
  "feature_importance": {
    "Close": 0.35,
    "SMA_20": 0.12,
    "RSI_14": 0.08,
    "MACD": 0.07,
    "BB_upper": 0.05
  },
  "chart_data": {
    "historical_prices": [
      {"date": "2025-10-01", "close": 37.50},
      {"date": "2025-10-02", "close": 37.80},
      // ... 60 dias
    ],
    "indicators": {
      "sma_20": [38.20, 38.25, 38.30],
      "rsi_14": [55.2, 56.8, 58.2]
    }
  },
  "metadata": {
    "model_version": "1.0.0",
    "prediction_timestamp": "2025-12-15T10:35:00Z",
    "processing_time_ms": 312
  }
}
```

---

## 🔧 Validações Necessárias (Backend)

### 1. Validação do Request
- ✅ `ticker`: String não-vazia, formato válido (ex: "PETR4.SA", "VALE3.SA")
- ✅ `tickers`: Array de strings (se múltiplos tickers), máximo 10 tickers por request
- ✅ Ticker deve existir no Yahoo Finance

### 2. Busca e Validação dos Dados
- ✅ Buscar últimos 60 dias de pregão do Yahoo Finance
- ✅ Verificar se há dados suficientes (mínimo 60 dias)
- ✅ Calcular todos os 14 indicadores técnicos usando pandas-ta
- ✅ Tratar valores NaN (forward-fill + backfill)
- ✅ Validar ranges dos indicadores:
  ```python
  # RSI: 0 - 100
  # Close, Open, High, Low, Volume: > 0
  # MACD, MACD_signal, MACD_hist: podem ser negativos
  # SMAs, EMAs, BBands: > 0
  # ATR: > 0
  ```

### 3. Tratamento de Erros
| Erro | HTTP Code | Response |
|------|-----------|----------|
| Ticker inválido | 400 | `{"error": "invalid_ticker"}` |
| Ticker não encontrado | 404 | `{"error": "ticker_not_found"}` |
| Dados insuficientes | 400 | `{"error": "insufficient_data"}` |
| Yahoo Finance offline | 503 | `{"error": "service_unavailable"}` |
| Modelo não carregado | 503 | `{"error": "model_unavailable"}` |
| Erro de inferência | 500 | `{"error": "prediction_error"}` |
| Muitos tickers | 400 | `{"error": "too_many_tickers"}` |

---

### Python (Cliente de Teste)
```python
import requests

def predict_stock(ticker: str):
    url = "http://localhost:5000/predict"
    payload = {"ticker": ticker}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

# Uso - Ticker único:
result = predict_stock("PETR4.SA")
print(f"Ticker: {result['ticker']}")
print(f"Previsão: R$ {result['prediction']['next_day_close']:.2f}")
print(f"Data: {result['prediction']['prediction_date']}")
print(f"Variação: {result['prediction']['change_percent']:.2f}%")

# Uso - Múltiplos tickers:
def predict_multiple_stocks(tickers: list[str]):
    url = "http://localhost:5000/predict"
    payload = {"tickers": tickers}
    response = requests.post(url, json=payload)
    return response.json()

results = predict_multiple_stocks(["PETR4.SA", "VALE3.SA", "ITUB4.SA"])
for pred in results['predictions']:
    print(f"{pred['ticker']}: R$ {pred['prediction']['next_day_close']:.2f}")
```

---

## 🚀 Implementação Backend (Checklist para Pessoa B)

### Fase 1: Setup Inicial
- [ ] Criar projeto Flask ou FastAPI
- [ ] Estruturar pastas: `/app`, `/models`, `/utils`, `/tests`
- [ ] Configurar CORS para comunicação com frontend
- [ ] Implementar logging estruturado

### Fase 2: Carregamento do Modelo
- [ ] Criar classe `ModelLoader` para carregar `model.pt` e `scaler.pkl`
- [ ] Implementar singleton pattern (carregar modelo 1x na inicialização)
- [ ] Validar arquivos de artefatos existem

### Fase 3: Integração com Yahoo Finance
- [ ] Instalar biblioteca `yfinance`
- [ ] Criar função para buscar últimos 60 dias de dados
- [ ] Implementar cache de dados (opcional, para reduzir chamadas à API)
- [ ] Tratar timeout e erros de conexão
- [ ] Validar se há dados suficientes (mínimo 60 dias)

### Fase 4: Cálculo de Indicadores Técnicos
- [ ] Instalar biblioteca `pandas-ta`
- [ ] Criar função para calcular os 14 indicadores técnicos
- [ ] Implementar tratamento de NaN (forward-fill + backfill)
- [ ] Validar ranges dos indicadores calculados

### Fase 5: Endpoints
- [ ] Implementar `GET /health`
- [ ] Implementar `GET /model/info`
- [ ] Implementar `POST /predict` (ticker único)
- [ ] Implementar `POST /predict` (múltiplos tickers)
- [ ] Implementar `POST /predict/detailed` (opcional)

### Fase 6: Validações
- [ ] Validar schema do request com Pydantic (FastAPI) ou Marshmallow (Flask)
- [ ] Validar formato do ticker (padrão brasileiro .SA)
- [ ] Validar limite máximo de tickers (10 por request)
- [ ] Validar ranges dos valores numéricos
- [ ] Implementar timeout para busca de dados (máximo 10s)

### Fase 7: Processamento
- [ ] Buscar dados históricos do Yahoo Finance
- [ ] Calcular indicadores técnicos com pandas-ta
- [ ] Converter dados → Pandas DataFrame
- [ ] Normalizar dados com scaler carregado
- [ ] Converter para tensor PyTorch
- [ ] Executar inferência do modelo
- [ ] Desnormalizar resultado
- [ ] Calcular intervalo de confiança

### Fase 8: Testes
- [ ] Testes unitários dos endpoints
- [ ] Testes com tickers válidos (PETR4.SA, VALE3.SA)
- [ ] Testes com tickers inválidos
- [ ] Testes com múltiplos tickers
- [ ] Testes de timeout do Yahoo Finance
- [ ] Testes de carga (performance)
- [ ] Teste end-to-end completo

### Fase 9: Deploy
- [ ] Criar `Dockerfile`
- [ ] Configurar variáveis de ambiente
- [ ] Deploy em servidor/cloud
- [ ] Configurar monitoramento (Prometheus/Grafana)
- [ ] Documentar API com Swagger/OpenAPI

---

## 📁 Artefatos a Serem Compartilhados

### Arquivos que a Pessoa A entregará para Pessoa B:
```
artifacts/
├── model.pt           # Modelo LSTM treinado (PyTorch state_dict)
├── scaler.pkl         # MinMaxScaler treinado (joblib)
└── model_config.json  # Configurações do modelo (opcional)

docs/
├── API_SCHEMA_COMPLETO.md         # Este documento
├── MODEL_DOCUMENTATION.md          # Documentação técnica do modelo
└── DECISOES_DADOS_FINANCEIROS.md  # Decisões sobre features

src/model_training/
├── model.py           # Classe StockLSTM (para referência)
├── predict.py         # StockPredictor (para referência)
└── evaluate.py        # Funções de métricas

notebooks/
└── eda.ipynb          # Notebook completo com análise e treinamento
```

---

## 💡 Fluxo de Processamento (Backend)

```
1. Frontend → POST /predict {"ticker": "PETR4.SA"}
   
2. Backend valida ticker
   ↓
3. Backend busca últimos 60 dias do Yahoo Finance
   ↓
4. Backend calcula 14 indicadores técnicos (pandas-ta)
   ↓
5. Backend normaliza dados (scaler.pkl)
   ↓
6. Backend converte para tensor PyTorch
   ↓
7. Backend executa modelo LSTM (model.pt)
   ↓
8. Backend desnormaliza resultado
   ↓
9. Backend calcula intervalo de confiança
   ↓
10. Backend retorna JSON → Frontend
```

---

## 📊 Exemplo Simplificado de Request/Response

### Request Simples:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'
```

### Response Simples:
```json
{
  "success": true,
  "ticker": "PETR4.SA",
  "prediction": {
    "next_day_close": 39.15,
    "last_known_close": 38.90,
    "change_percent": 0.64,
    "prediction_date": "2025-12-16"
  }
}
```

**Nota:** O backend faz todo o trabalho pesado:
- ✅ Busca os 60 dias automaticamente
- ✅ Calcula todos os indicadores técnicos
- ✅ Normaliza os dados
- ✅ Executa o modelo
- ✅ Retorna apenas o resultado final

**Vantagens:**
- Frontend não precisa saber sobre indicadores técnicos
- Payload de request extremamente pequeno (< 50 bytes)
- Menos chance de erros de validação
- Backend tem controle total sobre a qualidade dos dados

---

## 🎯 Métricas de Performance (SLA)

| Métrica | Target | Descrição |
|---------|--------|-----------|
| Latency (p95) | < 100ms | Tempo de resposta do `/predict` |
| Throughput | > 100 req/s | Requests por segundo |
| Availability | > 99.5% | Uptime da API |
| Error Rate | < 0.5% | Taxa de erros 5xx |

---

## 🔒 Segurança e Boas Práticas

1. **Rate Limiting:** Limitar requests por IP/usuário
2. **Input Sanitization:** Validar todos os inputs
3. **HTTPS:** Usar SSL em produção
4. **API Key:** Implementar autenticação básica (opcional)
5. **Logging:** Registrar todas as requisições e erros
6. **Monitoring:** Alertas para erros críticos

---

## 📞 Contato e Dúvidas

**Pessoa A (Data Science):**  
- Responsável por: Modelo LSTM, features, métricas
- Disponível para: Esclarecimentos sobre modelo e dados

**Pessoa B (ML Engineering/Backend):**  
- Responsável por: API REST, deploy, infraestrutura
- Deve implementar: Endpoints, validações, testes

---

## ✅ Checklist de Handover

### Pessoa A deve entregar:
- [x] Modelo treinado (`model.pt`)
- [x] Scaler normalizado (`scaler.pkl`)
- [x] Documentação do schema da API (simplificada)
- [x] Código de referência (predict.py, model.py)
- [x] Notebook com análise completa
- [x] Lista de indicadores técnicos necessários

### Pessoa B deve implementar:
- [ ] Integração com Yahoo Finance (yfinance)
- [ ] Cálculo automático de indicadores (pandas-ta)
- [ ] Endpoints da API REST
- [ ] Validações e tratamento de erros
- [ ] Testes com tickers brasileiros
- [ ] Cache de dados (opcional)
- [ ] Deploy e monitoramento

---

**Documento criado por:** Pessoa A - Data Science Team  
**Versão:** 1.0.0  
**Última atualização:** 15/12/2025  
**Status:** ✅ Pronto para implementação
