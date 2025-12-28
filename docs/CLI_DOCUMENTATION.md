# 🚀 Stock Prediction ML Pipeline - CLI Documentation

## ✅ Implementação Completa

### Fase 2: Hyperparameter Tuning ✅
**Implementado:** `HyperparameterTuner` com Optuna para otimização Bayesiana

#### Módulos Criados:
1. **`src/ml/training/hyperparameter_tuner.py`**
   - Classe `HyperparameterTuner` com Optuna
   - Otimização de 6 hiperparâmetros:
     - `learning_rate`: [1e-5, 1e-2] (log scale)
     - `hidden_size`: [32, 256] (step=16)
     - `num_layers`: [1, 4]
     - `dropout`: [0.1, 0.5]
     - `batch_size`: [16, 32, 64, 128]
     - `weight_decay`: [1e-6, 1e-3] (log scale)
   
2. **`examples/hyperparameter_tuning.py`**
   - Script completo de tuning end-to-end
   - 50 trials (cada trial: 30 epochs)
   - Salva gráficos de otimização
   - Exporta melhor config em JSON

#### Features:
- ✅ Bayesian optimization (Optuna)
- ✅ Integração com MLflow (tracking de trials)
- ✅ Early stopping por trial (patience=10)
- ✅ Progress bar interativo
- ✅ Plots de otimização (history + importances)
- ✅ Exportação de best config (`artifacts/best_hyperparameters.json`)

#### Uso:
```bash
python examples/hyperparameter_tuning.py
```

**Status:** 🟢 Tuning rodando em background (50 trials, ~30-60 min)

---

### Fase 3: CLI Development ✅
**Implementado:** CLI profissional com Click

#### Comandos Disponíveis:

##### 1. **`stock-ml train`** - Treinar modelo
```bash
python -m cli.main train \
    --ticker PETR4.SA \
    --epochs 100 \
    --lr 0.001 \
    --hidden-size 50 \
    --num-layers 2 \
    --dropout 0.2 \
    --batch-size 32 \
    --experiment-name lstm-petr4
```

**Opções:**
- `--ticker`: Ticker do ativo (e.g., PETR4.SA)
- `--start-date`: Data inicial (default: 2020-01-01)
- `--end-date`: Data final (default: hoje)
- `--lookback`: Período de lookback (default: 60)
- `--hidden-size`: Tamanho oculto LSTM (default: 50)
- `--num-layers`: Número de camadas (default: 2)
- `--dropout`: Taxa de dropout (default: 0.2)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Máximo de epochs (default: 100)
- `--batch-size`: Tamanho do batch (default: 32)
- `--experiment-name`: Nome do experimento MLflow
- `--model-path`: Caminho para salvar modelo
- `--seed`: Seed para reprodutibilidade (default: 42)

**Pipeline:**
1. Data Ingestion (yfinance)
2. Feature Engineering (7 indicators)
3. Preprocessing (normalização + sequences)
4. Model Creation (LSTM)
5. Training (com early stopping + MLflow)

---

##### 2. **`stock-ml predict`** - Previsões batch
```bash
python -m cli.main predict \
    --model-path artifacts/models/best_model.pt \
    --ticker PETR4.SA \
    --days-ahead 5 \
    --output data/predictions/petr4_20251228.csv
```

**Opções:**
- `--model-path`: Path do modelo treinado (.pt)
- `--ticker`: Ticker do ativo
- `--days-ahead`: Dias a prever (default: 5)
- `--lookback`: Período de lookback (default: 60)
- `--output`: Path do CSV de saída

**Output:**
- CSV com colunas: `Date`, `Predicted_Close`
- Previsões desnormalizadas
- Rolling predictions (multi-step)

---

##### 3. **`stock-ml evaluate`** - Avaliar modelo
```bash
python -m cli.main evaluate \
    --model-path artifacts/models/best_model.pt \
    --ticker PETR4.SA
```

**Opções:**
- `--model-path`: Path do modelo (.pt)
- `--ticker`: Ticker do ativo
- `--start-date`: Data inicial dos dados
- `--lookback`: Período de lookback

**Métricas calculadas:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Directional Accuracy

**Output:**
- Métricas no console
- JSON salvo em `artifacts/models/test_metrics.json`

---

##### 4. **`stock-ml monitor`** - MLflow UI
```bash
python -m cli.main monitor
```

**Opções:**
- `--port`: Porta do servidor (default: 5000)
- `--host`: Host (default: 127.0.0.1)
- `--backend-store-uri`: Tracking URI (default: file:./mlruns)

**Acesso:**
- Interface web: http://localhost:5000
- Visualizar experimentos, métricas, modelos

---

#### Estrutura CLI:

```
cli/
├── __init__.py         # Exporta cli()
├── main.py             # Entry point com registro de comandos
├── train.py            # Comando train
├── predict.py          # Comando predict
├── evaluate.py         # Comando evaluate
└── monitor.py          # Comando monitor
```

#### Instalação:

```bash
# Instalar CLI como comando
pip install -e .

# Usar CLI (após instalação)
stock-ml --help
stock-ml train --ticker PETR4.SA
```

**Alternativa (sem instalação):**
```bash
python -m cli.main --help
python -m cli.main train --ticker PETR4.SA
```

---

## 📊 Status do Projeto

### ✅ Implementado (Fases 1-3):
- [x] Refatoração completa do notebook
- [x] Módulos de dados (ingestion, preprocessing, feature_engineering)
- [x] Modelo LSTM com factory pattern
- [x] Training pipeline com early stopping + MLflow
- [x] 20 testes unitários (100% passing)
- [x] Otimizações PyTorch (unfold, split)
- [x] **Hyperparameter tuning com Optuna**
- [x] **CLI profissional com Click (4 comandos)**

### 🟡 Em Progresso:
- [ ] Hyperparameter tuning rodando (50 trials, ~30-60 min)

### 📋 Próximas Fases:
- **Fase 4:** Pipeline Orchestration (TrainPipeline, PredictPipeline)
- **Fase 5:** Batch Prediction & Scheduling (APScheduler)
- **Fase 6:** Testing & Documentation
- **Fase 7:** Dockerização & Handover

---

## 🎯 Exemplo de Uso Completo

### 1. Treinar modelo com CLI:
```bash
python -m cli.main train \
    --ticker PETR4.SA \
    --epochs 100 \
    --lr 0.001 \
    --experiment-name lstm-petr4-v1
```

### 2. Avaliar modelo:
```bash
python -m cli.main evaluate \
    --model-path artifacts/models/best_model.pt \
    --ticker PETR4.SA
```

### 3. Fazer previsões:
```bash
python -m cli.main predict \
    --model-path artifacts/models/best_model.pt \
    --ticker PETR4.SA \
    --days-ahead 5
```

### 4. Monitorar experimentos:
```bash
python -m cli.main monitor
# Abrir http://localhost:5000
```

---

## 🔬 Hyperparameter Tuning

### Rodando agora:
- **Trials:** 50
- **Tempo estimado:** 30-60 minutos
- **Experimento MLflow:** `hyperparameter_tuning`
- **Output:**
  - Best config: `artifacts/best_hyperparameters.json`
  - Plots: `artifacts/plots/optuna_optimization.png`

### Após tuning:
1. Carregar best config
2. Re-treinar modelo completo (100 epochs)
3. Avaliar performance
4. Comparar com baseline

---

## 📈 Melhorias vs Baseline

### Baseline (hiperparâmetros fixos):
- lr: 0.001
- hidden_size: 50
- num_layers: 2
- dropout: 0.2
- Val Loss: 0.001

### Esperado com tuning:
- **15-30% de melhoria** nas métricas
- Val Loss: ~0.0007-0.0008 (estimativa)
- Melhor generalização

---

**Status:** 🟢 Tuning em background + CLI funcional  
**Próximo passo:** Aguardar tuning + implementar Fase 4 (Pipeline Orchestration)
