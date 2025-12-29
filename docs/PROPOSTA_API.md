# 📡 Proposta de API - Stock Prediction LSTM

**Pessoa A → Pessoa B**  
**Data:** 15/12/2025

---

## 🎯 Resumo

API REST que recebe um **ticker de ação** e retorna a **previsão do preço** para o próximo dia.

**Simplicidade:** Frontend só envia o ticker. Backend busca dados históricos, calcula indicadores e faz a previsão.

---

## 📋 Endpoints Propostos

### 1. Health Check
**GET** `/health`

Verifica se a API está funcionando.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-15T10:30:00Z"
}
```

---

### 2. Informações do Modelo
**GET** `/model/info`

Retorna metadados sobre o modelo treinado.

**Response:**
```json
{
  "model_name": "StockLSTM",
  "ticker_trained": "PETR4.SA",
  "lookback_days": 60,
  "features_count": 20,
  "architecture": {
    "hidden_size": 50,
    "num_layers": 2,
    "dropout": 0.2
  },
  "metrics": {
    "mae": 0.85,
    "rmse": 1.12,
    "mape": 3.45
  }
}
```

---

### 3. Previsão de Preço ⭐
**POST** `/predict`

Recebe ticker, busca últimos 60 dias automaticamente e retorna previsão.

#### Request:
```json
{
  "ticker": "PETR4.SA"
}
```

**Ou múltiplos tickers:**
```json
{
  "tickers": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
}
```

#### Response (ticker único):
```json
{
  "success": true,
  "ticker": "PETR4.SA",
  "prediction": {
    "next_day_close": 39.15,
    "prediction_date": "2025-12-16",
    "last_known_close": 38.90,
    "last_known_date": "2025-12-15",
    "change_percent": 0.64,
    "confidence_interval": {
      "lower": 38.50,
      "upper": 39.80
    }
  },
  "metadata": {
    "model_version": "1.0.0",
    "processing_time_ms": 245
  }
}
```

#### Response (múltiplos tickers):
```json
{
  "success": true,
  "predictions": [
    {
      "ticker": "PETR4.SA",
      "prediction": {
        "next_day_close": 39.15,
        "change_percent": 0.64
      }
    },
    {
      "ticker": "VALE3.SA",
      "prediction": {
        "next_day_close": 62.80,
        "change_percent": 0.72
      }
    }
  ]
}
```

#### Erros:
```json
// Ticker inválido - 400
{
  "success": false,
  "error": "invalid_ticker",
  "message": "Ticker não encontrado"
}

// Dados insuficientes - 400
{
  "success": false,
  "error": "insufficient_data",
  "message": "Menos de 60 dias disponíveis"
}

// Yahoo Finance offline - 503
{
  "success": false,
  "error": "service_unavailable",
  "message": "Não foi possível buscar dados"
}
```

---

## 🔧 O que o Backend deve fazer

### Fluxo de Processamento:
1. Recebe `{"ticker": "PETR4.SA"}`
2. **Busca dados** → yfinance (últimos 60 dias)
3. **Calcula indicadores** → pandas-ta (14 indicadores técnicos)
4. **Normaliza** → scaler.pkl
5. **Predição** → model.pt (LSTM)
6. **Desnormaliza** → resultado final
7. Retorna JSON

### Indicadores Técnicos a Calcular:
- SMA (20, 50, 200 dias)
- EMA (12, 26 dias)
- RSI (14 dias)
- MACD (12, 26, 9)
- Bollinger Bands (20 dias, 2σ)
- ATR (14 dias)

**Total: 20 features** (6 preços básicos + 14 indicadores)
  
---

## 📦 Artefatos Disponíveis

```
artifacts/
├── model.pt           # Modelo LSTM treinado
└── scaler.pkl         # Normalizador treinado

src/model_training/
├── model.py           # Classe StockLSTM (referência)
├── predict.py         # Código de inferência (referência)
└── evaluate.py        # Métricas

docs/
├── PROPOSTA_API.md                # Este documento
├── MODEL_DOCUMENTATION.md         # Documentação técnica
└── DECISOES_DADOS_FINANCEIROS.md  # Decisões sobre features
```

---

## 🎨 Funcionalidades Opcionais

A Pessoa B pode decidir implementar:

- **Endpoint `/predict/detailed`** com mais informações técnicas
- **Dados para gráficos** (histórico de 60 dias para plotar) ⭐
- **Cache de dados** para reduzir chamadas ao Yahoo Finance
- **Rate limiting** por IP
- **Autenticação** com API key
- **SSE** para atualizações event-driven

**Sugestão:** Começar simples com os 3 endpoints principais e depois expandir conforme necessário.

---

## 📊 Sugestão de Gráfico (Opcional)

### Gráfico de Linha com Previsão

Mostrar os **últimos 60 dias de preços reais** + **previsão do dia seguinte** em destaque.

**Elementos sugeridos:**
1. **Linha principal:** Preços de fechamento históricos (60 dias)
2. **Ponto destacado:** Previsão do próximo dia (cor diferente, maior)
3. **Médias móveis:** SMA 20 e SMA 50 (linhas tracejadas)
4. **Área de confiança:** Faixa sombreada entre upper/lower confidence
5. **Eixos:** Datas no X, Preços (R$) no Y

#### Mockup Visual:
```
Preço (R$)
   ↑
41 │                                              ⭐ (Previsão)
40 │                                    ╱─────○
39 │                          ╱────────╱
38 │              ╱──────────╱
37 │   ──────────╱               [Área cinza = confiança]
36 │                             ─ ─ ─ ─ SMA 20
35 │                             ─ ─ ─ ─ SMA 50
   └─────────────────────────────────────────────────→
     -60d    -40d    -20d     hoje    +1d (Previsão)
```

#### Endpoint para Retornar Dados do Gráfico:

**Request simples - Backend busca tudo automaticamente:**
```
GET /predict/chart?ticker=PETR4.SA
```

**Request múltiplos tickers:**
```
GET /predict/chart?tickers=PETR4.SA,VALE3.SA,ITUB4.SA
```

**Response (ticker único):**
```json
{
  "success": true,
  "ticker": "PETR4.SA",
  "chart_data": {
    "historical": [
      {"date": "2025-10-01", "close": 37.50},
      {"date": "2025-10-02", "close": 37.80},
      // ... 60 dias automaticamente buscados
      {"date": "2025-12-15", "close": 38.90}
    ],
    "indicators": {
      "sma_20": [38.20, 38.25, ..., 38.75],
      "sma_50": [37.80, 37.85, ..., 38.20]
    },
    "prediction": {
      "date": "2025-12-16",
      "value": 39.15,
      "confidence_upper": 39.80,
      "confidence_lower": 38.50
    }
  }
}
```

**Response (múltiplos tickers):**
```json
{
  "success": true,
  "charts": [
    {
      "ticker": "PETR4.SA",
      "chart_data": {
        "historical": [
          {"date": "2025-10-01", "close": 37.50},
          // ... 60 dias
          {"date": "2025-12-15", "close": 38.90}
        ],
        "prediction": {
          "date": "2025-12-16",
          "value": 39.15,
          "confidence_upper": 39.80,
          "confidence_lower": 38.50
        }
      }
    },
    {
      "ticker": "VALE3.SA",
      "chart_data": {
        "historical": [
          {"date": "2025-10-01", "close": 61.20},
          // ... 60 dias
          {"date": "2025-12-15", "close": 62.35}
        ],
        "prediction": {
          "date": "2025-12-16",
          "value": 62.80,
          "confidence_upper": 64.10,
          "confidence_lower": 61.50
        }
      }
    }
  ],
  "metadata": {
    "total_charts": 2,
    "processing_time_ms": 520
  }
}
```

**Comportamento do Backend:**
- Busca automaticamente os **últimos 60 dias de pregão** do Yahoo Finance
- Calcula indicadores técnicos (SMA 20, SMA 50, etc.)
- Gera a previsão para o próximo dia
- Retorna tudo formatado para o frontend plotar

**Nota:** Frontend só passa o ticker. Backend faz todo o trabalho de busca e cálculo.

#### Bibliotecas Frontend Sugeridas:
- **Chart.js** - Simples e responsivo
- **Recharts** - Nativo React, fácil customização
- **Plotly.js** - Interativo, zoom, hover
- **Apache ECharts** - Profissional, muitas features

**Recomendação:** Recharts para React ou Chart.js para vanilla JS.
 

**Nota:** O backend pode adicionar um endpoint `/predict/chart` que:
- Recebe apenas o **ticker** como parâmetro
- Busca automaticamente os **últimos 60 dias** do Yahoo Finance
- Calcula todos os **indicadores técnicos**
- Retorna tudo **formatado e pronto** para plotar no frontend

**Vantagem:** Frontend não precisa se preocupar com datas, ranges ou cálculos. Só passa o ticker e renderiza o gráfico!

---

## ✅ Checklist de Implementação

### Backend (Pessoa B):
- [ ] Setup Flask 
- [ ] Integração com yfinance
- [ ] Cálculo de indicadores (pandas-ta)
- [ ] Carregar model.pt + scaler.pkl
- [ ] Implementar 3 endpoints principais
- [ ] Validações e tratamento de erros
- [ ] Testes com PETR4.SA, VALE3.SA
- [ ] Deploy

### Frontend (Depois):
- [ ] Interface para selecionar ticker
- [ ] Exibir previsão e variação %
- [ ] Gráficos (se implementado)
- [ ] Loading states
- [ ] Tratamento de erros

--- 