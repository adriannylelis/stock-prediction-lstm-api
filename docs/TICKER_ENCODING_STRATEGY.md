# Ticker Encoding Strategy: One-Hot vs Embedding

## 📊 Problema: Como Representar Múltiplos Tickers?

Quando treinamos **um único modelo para múltiplos tickers**, precisamos dizer ao modelo "de qual ação são esses dados". Existem duas abordagens principais:

## 🔧 Abordagem 1: One-Hot Encoding (Atual)

### Como Funciona
```python
# Lista de tickers
tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]

# Encoding para cada ticker
PETR4.SA → [1, 0, 0]
VALE3.SA → [0, 1, 0]
ITUB4.SA → [0, 0, 1]

# Input final para LSTM
Input shape: (batch, 60, 19 + 3) = (batch, 60, 22)
```

### Vantagens ✅
- **Simples**: Fácil de implementar
- **Interpretável**: 1 = "é este ticker", 0 = "não é"
- **Funciona bem**: Para 2-10 tickers

### Desvantagens ❌
- **Não escala**: Para 50 tickers = 50 colunas extras
- **Esparsidade**: 98% dos valores são 0
- **Sem semântica**: PETR3 e PETR4 tratados como completamente diferentes
- **Alta dimensionalidade**: Input size explode

### Escalabilidade
| Tickers | Input Size | Viável? |
|---------|-----------|---------|
| 2       | 19 + 2 = **21** | ✅ Ótimo |
| 10      | 19 + 10 = **29** | ✅ OK |
| 50      | 19 + 50 = **69** | ⚠️ Ruim |
| 100     | 19 + 100 = **119** | 🔴 Inviável |

## 🚀 Abordagem 2: Embedding Layer (Recomendada para Produção)

### Como Funciona
```python
# Mapeamento ticker → ID
ticker_to_id = {
    "PETR4.SA": 0,
    "VALE3.SA": 1,
    "ITUB4.SA": 2,
    # ... pode ter 1000+ tickers ...
}

# Embedding aprendido (exemplo com 8 dimensões)
PETR4.SA (id=0) → [0.23, -0.15, 0.87, 0.42, -0.31, 0.65, -0.09, 0.18]
VALE3.SA (id=1) → [0.11, -0.08, 0.92, 0.38, -0.28, 0.71, -0.12, 0.21]
ITUB4.SA (id=2) → [-0.45, 0.32, -0.15, 0.68, 0.21, -0.39, 0.55, -0.12]

# Input final para LSTM (SEMPRE mesmo tamanho)
Input shape: (batch, 60, 19 + 8) = (batch, 60, 27)
```

### Vantagens ✅
- **Escalável**: Tamanho fixo independente de #tickers
- **Semântica aprendida**: Tickers similares ficam próximos
- **Eficiente**: Representação densa (sem zeros)
- **Transfer learning**: Aprende relações entre setores

### Desvantagens ❌
- **Mais complexo**: Requer nn.Embedding
- **Menos interpretável**: Valores abstratos
- **Precisa IDs**: Mapeamento ticker → número

### Escalabilidade
| Tickers | Input Size | Viável? |
|---------|-----------|---------|
| 2       | 19 + 8 = **27** | ✅ Ótimo |
| 10      | 19 + 8 = **27** | ✅ Ótimo |
| 50      | 19 + 8 = **27** | ✅ Ótimo |
| 100     | 19 + 8 = **27** | ✅ Ótimo |
| 1000    | 19 + 8 = **27** | ✅ Ótimo |

## 🎯 Quando Usar Cada Abordagem?

### Use One-Hot Encoding se:
- ✅ Você tem **2-10 tickers** fixos
- ✅ Prototipagem rápida
- ✅ Simplicidade é prioridade
- ✅ Tickers não mudam frequentemente

**Exemplo:** Projeto acadêmico focado em poucas ações blue chips

### Use Embedding Layer se:
- ✅ Você tem **10+ tickers**
- ✅ Número de tickers pode crescer
- ✅ Quer escalabilidade
- ✅ Quer aprender relações semânticas (PETR3 ≈ PETR4)
- ✅ Produção real

**Exemplo:** Sistema de produção que monitora toda B3 (50+ ações)

## 📈 Comparação de Performance

### Memória GPU (Batch=32, Seq=60)
```
One-Hot (50 tickers):
- Input: (32, 60, 69) = 132,480 valores
- LSTM(69 → 100): ~70K parâmetros

Embedding (50 tickers, dim=8):
- Input: (32, 60, 27) = 51,840 valores (-61%)
- Embedding: 50 × 8 = 400 parâmetros
- LSTM(27 → 100): ~30K parâmetros (-57%)
```

### Tempo de Treinamento (Estimado)
- **One-Hot (50 tickers)**: 1.0x (baseline)
- **Embedding (dim=8)**: ~0.6x (-40% mais rápido)

## 🛠️ Como Implementar Embedding

### 1. Modelo com Embedding
```python
from src.ml.models.lstm import StockLSTM

model = StockLSTM(
    num_tickers=50,        # Número total de tickers
    num_features=19,       # Features técnicas
    embedding_dim=8,       # Dimensão do embedding (8-16 típico)
    hidden_size=100,
    num_layers=3,
    dropout=0.3
)
```

### 2. Preparação de Dados
```python
# Criar mapeamento ticker → ID
ticker_to_id = {ticker: idx for idx, ticker in enumerate(tickers)}
id_to_ticker = {idx: ticker for ticker, idx in ticker_to_id.items()}

# Durante treinamento
for batch_features, batch_targets in dataloader:
    # batch_features: (batch, 60, 19) - SEM ticker encoding
    # ticker_ids: (batch,) - ID do ticker de cada sequência
    
    predictions = model(batch_features, ticker_ids)
```

### 3. Mudanças no Preprocessing
```python
# ANTES (One-Hot):
# - Concatenar [features + one_hot_ticker] → (60, 21)
# - Criar tensor único

# DEPOIS (Embedding):
# - Manter features separadas → (60, 19)
# - Passar ticker_id como argumento separado → escalar
# - Modelo faz embedding internamente
```

## 🔄 Estratégia de Migração

### Fase 1: Desenvolvimento (Atual) ✅
- **Usar:** One-Hot Encoding
- **Tickers:** 2-10 (PETR4, VALE3, etc.)
- **Motivo:** Simplicidade, prototipagem rápida
- **Status:** Implementado e funcionando

### Fase 2: Validação
- **Implementar:** Embedding Layer (já criado em `lstm_with_embedding.py`)
- **Testar:** Comparar performance One-Hot vs Embedding
- **Validar:** Métricas (MAE, R², tempo de treino)

### Fase 3: Produção
- **Migrar para:** Embedding Layer
- **Escalar para:** 20-50 tickers (categorias da B3)
- **Benefícios:** Escalabilidade, performance

## 📚 Exemplos de Uso

### Cenário 1: Teste Atual (2 tickers)
```python
# ✅ One-Hot funciona perfeitamente
tickers = ["PETR4.SA", "VALE3.SA"]
# Input: (batch, 60, 21) = 19 features + 2 one-hot
```

### Cenário 2: Blue Chips (10 tickers)
```python
# ⚠️ One-Hot ainda OK, mas Embedding recomendado
tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", ..., "SUZB3.SA"]
# One-Hot: (batch, 60, 29) = 19 + 10
# Embedding: (batch, 60, 27) = 19 + 8 ✅ Melhor
```

### Cenário 3: Toda B3 (50+ tickers)
```python
# 🔴 One-Hot inviável, DEVE usar Embedding
tickers = ALL_TICKERS  # ~50 tickers
# One-Hot: (batch, 60, 69) ❌ Muito grande
# Embedding: (batch, 60, 27) ✅ Eficiente
```

## 🎓 Aprendizado do Embedding

Após treinar, você pode analisar as relações aprendidas:

```python
import torch.nn.functional as F

# Obter embeddings
petr4_emb = model.get_ticker_embedding(ticker_to_id["PETR4.SA"])
petr3_emb = model.get_ticker_embedding(ticker_to_id["PETR3.SA"])
vale3_emb = model.get_ticker_embedding(ticker_to_id["VALE3.SA"])
itub4_emb = model.get_ticker_embedding(ticker_to_id["ITUB4.SA"])

# Calcular similaridades
sim_petr4_petr3 = F.cosine_similarity(petr4_emb, petr3_emb, dim=0)
sim_petr4_vale3 = F.cosine_similarity(petr4_emb, vale3_emb, dim=0)
sim_petr4_itub4 = F.cosine_similarity(petr4_emb, itub4_emb, dim=0)

print(f"PETR4 vs PETR3: {sim_petr4_petr3:.4f}")  # Alto (mesmo setor)
print(f"PETR4 vs VALE3: {sim_petr4_vale3:.4f}")  # Médio (commodities)
print(f"PETR4 vs ITUB4: {sim_petr4_itub4:.4f}")  # Baixo (setores diferentes)
```

Você pode visualizar clusters de setores:
- **Energia**: PETR3, PETR4, ELET3, ELET6
- **Mineração**: VALE3, CMIN3, GOAU4
- **Bancos**: ITUB4, BBDC4, BBAS3, SANB11

## 🎯 Recomendação Final

### Para seu projeto atual:
✅ **Mantenha One-Hot** para desenvolvimento/testes (2-10 tickers)

### Para produção futura:
✅ **Migre para Embedding** quando escalar (10+ tickers)

### Próximos passos:
1. ✅ Validar funcionamento com One-Hot (em andamento)
2. ⏳ Implementar pipeline com Embedding (código pronto)
3. ⏳ Comparar performance (A/B test)
4. ⏳ Deploy com Embedding se métricas forem melhores

---

**Arquivos relevantes:**
- One-Hot: `scripts/train_unified_model.py` (atual)
- Embedding: `src/ml/models/lstm_with_embedding.py` (novo)
- Testes: `tests/e2e/test_mlops_complete.py`
