# 📊 Decisões sobre Dados Financeiros - Tech Challenge Fase 4

**Data:** 15/12/2025  
**Projeto:** Stock Prediction LSTM API  
**Fase:** Pessoa A - Data Science & Modelagem

---

## 🎯 Objetivo do Projeto

Criar modelo LSTM para prever preço de fechamento de ações usando dados do Yahoo Finance com:
- API RESTful para servir predições
- Métricas: MAE, RMSE, MAPE
- Deploy em nuvem
- Monitoramento de performance

---

## 📊 Análise de Dados Disponíveis (Rating 0-10)

### 🟢 ALTA PRIORIDADE (Rating 7-10)

#### 1. **Preços Históricos** - Rating: 10/10
**Status:** ✅ Implementado

**Dados:**
- Close (Fechamento)
- Open (Abertura)
- High (Máxima)
- Low (Mínima)
- Volume (Volume negociado)

**Justificativa:**
- Essencial para LSTM de séries temporais
- Dados diários completos e confiáveis
- Base principal do modelo
- Volume indica força dos movimentos

**Decisão:** MANTER como feature principal

---

#### 2. **Indicadores Técnicos** - Rating: 7/10
**Status:** 🔄 Recomendado para implementação

**Indicadores Sugeridos:**
- SMA (Simple Moving Average) - 20, 50, 200 dias
- EMA (Exponential Moving Average) - 12, 26 dias
- RSI (Relative Strength Index) - 14 dias
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

**Justificativa:**
- Capturam padrões técnicos conhecidos
- Médias móveis identificam tendências
- RSI detecta sobrecompra/sobrevenda
- MACD mede momentum
- Úteis como features adicionais para LSTM

**Decisão:** IMPLEMENTAR na versão atual
- Adicionar ao pré-processamento
- Incluir como features extras do modelo
- Requer biblioteca: `pandas-ta` ou `ta-lib`

---

### 🟡 MÉDIA PRIORIDADE (Rating 5-6)

#### 3. **Earnings (Resultados Trimestrais)** - Rating: 6/10
**Status:** 💡 Considerar para versão 2.0

**Dados:**
- Datas de anúncio de resultados
- Earnings por ação (EPS)
- Surpresas (beat/miss)

**Justificativa:**
- Eventos trimestrais com forte impacto
- Alta volatilidade próximo às datas
- Poderia ser feature: "dias até próximo earnings"

**Decisão:** NÃO implementar agora
- Complexidade adicional
- Dados esporádicos (4x/ano)
- Melhor para versão avançada

---

#### 4. **Notícias + Sentimento** - Rating: 5/10
**Status:** 📰 Projeto futuro

**Dados:**
- Feed de notícias da empresa
- Títulos e conteúdo
- Data/hora de publicação

**Justificativa:**
- Impacto no sentimento do mercado
- Requer NLP (processamento de linguagem natural)
- Análise de sentimento é complexa
- Fora do escopo do MVP

**Decisão:** NÃO implementar agora
- Complexidade muito alta
- Requer pipeline separado de NLP
- Ideal para projeto avançado/mestrado

---

### 🔴 BAIXA PRIORIDADE (Rating 2-4)

#### 5. **Demonstrações Financeiras** - Rating: 3-4/10
**Dados:**
- Income Statement (DRE)
- Balance Sheet (Balanço)
- Cash Flow (Fluxo de Caixa)

**Justificativa:**
- Dados TRIMESTRAIS/ANUAIS
- Impacto de longo prazo
- Não captura movimentos diários
- Melhor para análise fundamentalista

**Decisão:** NÃO implementar
- Não adequado para previsão diária
- Frequência incompatível com LSTM diário

---

#### 6. **Dividendos** - Rating: 3/10
**Status:** ❌ Não prioritário

**Justificativa:**
- Eventos esporádicos (não diários)
- Impacto pontual no preço
- Não crítico para modelo de curto prazo

**Decisão:** NÃO implementar

---

#### 7. **Splits (Desdobramentos)** - Rating: 2/10
**Status:** ✅ Já tratado automaticamente

**Justificativa:**
- Eventos raríssimos
- yfinance já ajusta preços automaticamente
- Não adiciona valor ao modelo

**Decisão:** Nenhuma ação necessária

---

#### 8. **Recomendações de Analistas** - Rating: 4/10
**Status:** ❌ Não prioritário

**Justificativa:**
- Dados esporádicos
- Impacto limitado em movimentos diários
- Melhor para análise qualitativa

**Decisão:** NÃO implementar

---

## 📋 RESUMO DAS DECISÕES

### ✅ IMPLEMENTADO NO MVP

| Categoria | Rating | Status |
|-----------|--------|--------|
| **Preços Históricos** | 10/10 | ✅ Completo |
| - Close, Open, High, Low | | |
| - Volume | | |

### 🔄 A IMPLEMENTAR (Versão Atual)

| Categoria | Rating | Ação |
|-----------|--------|------|
| **Indicadores Técnicos** | 7/10 | 🔨 Adicionar |
| - SMA (20, 50, 200) | | |
| - EMA (12, 26) | | |
| - RSI (14) | | |
| - MACD | | |
| - Bollinger Bands | | |

**Biblioteca:** `pandas-ta`
```bash
pip install pandas-ta
```

### 💡 CONSIDERAR FUTURO (V2.0)

| Categoria | Rating | Quando |
|-----------|--------|--------|
| Earnings | 6/10 | Versão 2.0 |
| Notícias + NLP | 5/10 | Projeto avançado |

### ❌ NÃO IMPLEMENTAR

| Categoria | Rating | Motivo |
|-----------|--------|--------|
| Demonstrações Financeiras | 3-4/10 | Frequência incompatível |
| Dividendos | 3/10 | Impacto limitado |
| Splits | 2/10 | Já tratado |
| Recomendações | 4/10 | Dados esporádicos |

---

## 🎯 Plano de Ação

### Fase Atual (MVP)
1. ✅ Preços históricos implementados
2. 🔨 Adicionar indicadores técnicos
3. ✅ Treinar modelo LSTM
4. ✅ Avaliar métricas (MAE, RMSE, MAPE)
5. ✅ Deploy da API

### Fase 2 (Melhorias)
1. Testar impacto dos indicadores técnicos
2. Feature engineering adicional
3. Hyperparameter tuning
4. Avaliar adição de earnings data

### Fase 3 (Avançado)
1. Análise de sentimento de notícias
2. Ensemble com outros modelos
3. Múltiplos tickers simultâneos

---

## 📊 Justificativa Técnica

### Por que Indicadores Técnicos?

**Vantagens:**
- Calculados a partir dos dados existentes
- Sem necessidade de APIs adicionais
- Padrões reconhecidos pelo mercado
- Melhora potencial do modelo

**Implementação:**
```python
import pandas_ta as ta

# SMA
df['SMA_20'] = ta.sma(df['Close'], length=20)
df['SMA_50'] = ta.sma(df['Close'], length=50)

# RSI
df['RSI_14'] = ta.rsi(df['Close'], length=14)

# MACD
macd = ta.macd(df['Close'])
df = df.join(macd)
```

### Por que NÃO usar Demonstrações Financeiras?

**Limitações:**
- Periodicidade: Trimestral/Anual vs Diária
- Delay: Publicadas semanas após o período
- Impacto: Longo prazo vs Curto prazo
- Escopo: Fundamentalista vs Técnico

**Conclusão:** Incompatível com LSTM de previsão diária

---

## 📈 Métricas de Sucesso

### MVP (Baseline)
- MAE < 5% do preço médio
- MAPE < 10%
- RMSE proporcional à volatilidade

### Com Indicadores Técnicos
- MAE: Redução de 10-20%
- MAPE: Melhoria de 1-2 pontos percentuais
- R²: Aumento de 0.05-0.10

---

## 🔗 Referências

1. **yfinance Documentation**  
   https://ranaroussi.github.io/yfinance/

2. **pandas-ta Documentation**  
   https://github.com/twopirllc/pandas-ta

3. **Technical Analysis Indicators**  
   - SMA: https://www.investopedia.com/terms/s/sma.asp
   - RSI: https://www.investopedia.com/terms/r/rsi.asp
   - MACD: https://www.investopedia.com/terms/m/macd.asp

---

## ✅ Aprovações

**Decisões aprovadas por:** Pessoa A (Data Science)  
**Data:** 15/12/2025  
**Status:** ✅ Documentado e pronto para implementação

**Próximo passo:** Implementar indicadores técnicos no notebook EDA
