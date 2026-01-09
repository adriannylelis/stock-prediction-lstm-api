# Relatório de Status - API de Predição LSTM

**Data:** 09/01/2026  
**Status:** ⚠️ Modelo treinado com sucesso, mas API com falhas de inferência
---
Teste com: `test_api_predictions.py`
---

## ✅ O que está funcionando

### 1. Treinamento do Modelo v118
- **Status:** ✅ Treinamento completado com sucesso
- **Métricas:** Loss final de 0.0004 (excelente!)
- **Tickers:** 36 tickers válidos
- **Features:** 19 features (OHLCV + 14 indicadores técnicos)
- **Arquitetura:** 342k parâmetros, embedding de 8 dimensões
- **Artifacts:** Scalers salvos corretamente no MLflow (X scaler 19 cols + y_scaler 1 col)

### 2. Script de Promoção
- **Script:** `promote_to_production.py` funcionando corretamente
- **Ação:** Promove modelo mais recente para Production stage no MLflow
- **Processo:**
  1. Arquiva modelo atual em Production → Archived
  2. Promove novo modelo → Production
  3. Atualiza `production_model.yaml`
- **Resultado:** ✅ Modelo v118 promovido com sucesso para Production

### 3. Correções Aplicadas
- ✅ Removido colunas extras da API (`Adj Close`, `Dividends`, `Stock Splits`)
- ✅ Pipeline de features agora gera exatos 19 features
- ✅ Multi-scaler implementado (X scaler + y_scaler separados)

---

## ❌ Problemas Identificados na API

### Problema 1: SMA_200 gerando NaN
**Root Cause:**
```python
# DataService busca apenas 60 dias de histórico
lookback_days = 60  # ❌ Insuficiente para SMA_200!

# Mas TechnicalIndicators tenta calcular SMA_200
def add_sma(self, window=200):  # ❌ Precisa de 200 dias!
    self.df[f'SMA_{window}'] = self.df['Close'].rolling(window=window).mean()
    # Resultado: TODOS os valores = NaN (temos só 60 dias)
```

**Evidência:**
```
🔍 NaN count per feature: [0 0 0 0 0 0 0 60 0 0 0 0 0 0 0 0 0 0 0]
                                      ↑
                                   SMA_200
```

**Impacto:**
- MinMaxScaler não consegue normalizar features com NaN
- Modelo recebe inputs corrompidos
- Predições retornam NaN

### Problema 2: Lookback insuficiente
**Atual:**
```python
# src/api/services/data_service.py
self.lookback_days = 60  # Apenas 60 dias
start_date = end_date - timedelta(days=self.lookback_days + 30)  # 90 dias total
```

**Necessário:**
- SMA_200 precisa de **200 dias** de histórico
- SMA_50 precisa de **50 dias**
- Com fillna/ffill, precisamos de margem extra
- **Mínimo recomendado:** 250 dias (200 + 50 de buffer)

---

## 🔧 Soluções Propostas

### Opção 1: Aumentar lookback do DataService (RECOMENDADO)
```python
# src/api/services/data_service.py
def __init__(self, lookback_days: int = 60):
    self.lookback_days = lookback_days
    
def fetch_data(self, ticker: str) -> pd.DataFrame:
    # Buscar dados suficientes para SMA_200
    start_date = end_date - timedelta(days=300)  # 300 dias (200 + buffer)
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    # Retornar apenas últimos 60 para predição
    # (mas com indicadores calculados corretamente)
    return df.tail(self.lookback_days)
```

**Vantagens:**
- Mantém todos os 19 features
- SMA_200 calculado corretamente
- Alinhado com treinamento

### Opção 2: Remover SMA_200 (NÃO RECOMENDADO)
```python
# feature_engineering.py - remover SMA_200
def add_all_indicators(self):
    self.add_sma(window=20)   # ✅
    self.add_sma(window=50)   # ✅
    # self.add_sma(window=200)  # ❌ REMOVIDO
```

**Desvantagens:**
- ⚠️ Modelo treinado com 19 features, API teria 18
- ⚠️ Shape mismatch novamente
- ⚠️ Precisaria retreinar modelo

---

## 📋 Checklist de Correções

### Imediatas (hoje):
- [ ] Modificar `DataService.fetch_data()` para buscar 300 dias
- [ ] Verificar se SMA_200 é calculado sem NaN
- [ ] Testar API com curl/Postman
- [ ] Validar predições retornam valores reais (não NaN)

### Próximos passos:
- [ ] Rodar testes E2E completos
- [ ] Documentar lookback requirements no README
- [ ] Adicionar validação de NaN antes de scaler.transform()
- [ ] Commitar todas as mudanças

---

## 🎯 Resumo Executivo

**Situação:**
- ✅ Modelo v118 treinado perfeitamente (loss 0.0004)
- ✅ Promoção para Production funcionando
- ❌ API falhando por dados insuficientes para SMA_200

**Causa Raiz:**
- DataService busca apenas 60 dias
- SMA_200 precisa de 200 dias → retorna NaN
- Scaler falha com NaN

**Solução:**
- Aumentar fetch para 300 dias
- Calcular indicadores com dados suficientes
- Retornar últimos 60 para inferência

**Tempo estimado:** 15-30 minutos para fix + testes

---

**Próxima ação:** Implementar fix no DataService e testar.
