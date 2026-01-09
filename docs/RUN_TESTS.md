# 🧪 Guia Rápido: Como Rodar os Testes

**Última Atualização**: 09/01/2026  
**Total de Testes**: 78 (53 unit, 8 integration, 7 e2e)  
**Tempo Total**: ~2 minutos (com GPU)

---

## ⚡ Quick Start (Copiar e Colar)

### Testes Rápidos - Unit Only (30s)
```bash
pytest tests/unit/ -v
```

### Todos os Testes (2min)
```bash
pytest -v
```

### Com Relatório de Cobertura (3min)
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing -v
```

---

## 📋 Pré-Requisitos

### 1. Dependências Instaladas
```bash
# Instalar todas as dependências de teste
pip install -r requirements.txt

# Ou apenas as essenciais
pip install pytest pytest-cov pytest-mock loguru torch mlflow flask
```

### 2. MLflow Tracking Configurado
```bash
# Garantir que tracking URI existe
mkdir -p data/mlflow/tracking

# Ou definir variável de ambiente
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/tracking
```

### 3. GPU (Opcional, mas recomendado para E2E)
```bash
# Verificar GPU disponível
python -c "import torch; print(torch.cuda.is_available())"

# Se False, testes E2E serão mais lentos (CPU)
```

---

## 🎯 Comandos por Categoria

### 📦 Unit Tests (53 tests - ~30s)

Testam componentes isolados com mocks.

```bash
# Todos os unit tests
pytest tests/unit/ -v

# Por módulo
pytest tests/unit/test_model.py -v                    # LSTM architecture (5 tests)
pytest tests/unit/test_trainer.py -v                  # Training loop (8 tests)
pytest tests/unit/test_data_ingestion.py -v           # yfinance (6 tests)
pytest tests/unit/test_feature_engineering.py -v      # Indicators (12 tests)
pytest tests/unit/test_preprocessing.py -v            # Scaler, sequences (7 tests)
pytest tests/unit/test_metrics.py -v                  # Loss, MAE, RMSE (5 tests)
pytest tests/unit/test_utils.py -v                    # Device, seed (6 tests)
pytest tests/unit/test_mlflow_tracker.py -v           # MLflow (4 tests)

# Apenas um teste específico
pytest tests/unit/test_model.py::test_lstm_forward_pass -v
```

### 🔗 Integration Tests (8 tests - ~20s)

Testam múltiplos componentes interagindo.

```bash
# Todos os integration tests
pytest tests/integration/ -v

# Por módulo
pytest tests/integration/test_api_routes.py -v        # Flask endpoints (3 tests)
pytest tests/integration/test_train_pipeline.py -v    # Full pipeline (2 tests)
pytest tests/integration/test_mlflow_integration.py -v # Registry (2 tests)
pytest tests/integration/test_cli_commands.py -v      # CLI (1 test)
```

### 🌐 E2E Tests (7 tests - ~1-2min)

Testam workflows completos end-to-end.

```bash
# Todos os E2E tests
pytest tests/e2e/ -v

# ATENÇÃO: E2E tests modificam o MLflow registry!
# Recomendado rodar em ambiente de teste isolado.

# Teste individual
pytest tests/e2e/test_mlops_complete.py::TestMLOpsComplete::test_1_training_with_optuna -v
pytest tests/e2e/test_mlops_complete.py::TestMLOpsComplete::test_4_prediction_via_mlflow -v
```

---

## 🛠️ Opções Úteis do Pytest

### Controle de Execução

```bash
# Parar no primeiro erro
pytest -v --maxfail=1

# Rodar apenas últimos tests que falharam
pytest --lf -v

# Rodar primeiro os que falharam, depois os demais
pytest --ff -v

# Timeout para cada teste (útil para evitar hangs)
pytest -v --timeout=60
```

### Output e Logs

```bash
# Mostrar prints durante execução
pytest -v -s

# Mais verboso (mostra cada assert)
pytest -vv

# Menos output (só summary)
pytest -q

# Traceback completo
pytest -v --tb=long

# Traceback resumido
pytest -v --tb=short

# Sem traceback
pytest -v --tb=no
```

### Cobertura de Código

```bash
# Relatório básico no terminal
pytest --cov=src -v

# Relatório HTML interativo
pytest --cov=src --cov-report=html -v
# Abrir: htmlcov/index.html

# Mostrar linhas faltando
pytest --cov=src --cov-report=term-missing -v

# Múltiplos formatos
pytest --cov=src --cov-report=html --cov-report=term --cov-report=xml -v

# Cobertura mínima exigida (falha se < 80%)
pytest --cov=src --cov-fail-under=80 -v

# Apenas para módulo específico
pytest tests/unit/test_model.py --cov=src.ml.models.lstm -v
```

### Execução Paralela (Mais Rápido)

```bash
# Instalar plugin
pip install pytest-xdist

# Usar todos os cores disponíveis
pytest -v -n auto

# Usar N workers
pytest -v -n 4

# ATENÇÃO: Alguns E2E tests podem conflitar em paralelo!
# Rodar apenas unit/integration em paralelo:
pytest tests/unit/ tests/integration/ -v -n auto
```

---

## 🎨 Filtrando Testes com Markers

### Markers Disponíveis

```bash
# Listar todos os markers
pytest --markers
```

Markers comuns:
- `@pytest.mark.slow` - Testes demorados (> 10s)
- `@pytest.mark.integration` - Testes de integração
- `@pytest.mark.e2e` - Testes end-to-end
- `@pytest.mark.cuda` - Requerem GPU
- `@pytest.mark.skip` - Temporariamente desabilitados
- `@pytest.mark.xfail` - Esperados falhar (bugs conhecidos)

### Usando Markers

```bash
# Apenas testes rápidos (excluir slow)
pytest -v -m "not slow"

# Apenas testes de integração
pytest -v -m integration

# Apenas E2E
pytest -v -m e2e

# Apenas testes que usam GPU
pytest -v -m cuda

# Combinar markers (integration OU e2e)
pytest -v -m "integration or e2e"

# Combinar markers (integration E não slow)
pytest -v -m "integration and not slow"
```

---

## 🐛 Debugging de Testes

### Modo Debug Interativo

```bash
# Parar no primeiro erro e entrar no debugger
pytest --pdb -v

# Parar em todos os testes (útil para explorar fixtures)
pytest --trace -v

# Usar ipdb ao invés de pdb (mais features)
pip install ipdb
pytest --pdb --pdbcls=IPython.terminal.debugger:Pdb -v
```

### Logs Detalhados

```bash
# Mostrar logs do loguru
pytest -v -s --log-cli-level=INFO

# Apenas logs de WARNING ou superior
pytest -v --log-cli-level=WARNING

# Salvar logs em arquivo
pytest -v --log-file=test_logs.txt --log-file-level=DEBUG
```

### Captura de Output

```bash
# Desabilitar captura (ver prints em tempo real)
pytest -v -s

# Capturar stdout mas não stderr
pytest -v --capture=no

# Ver output mesmo em testes que passaram
pytest -v -rP  # P = Passed with output
```

---

## 📊 Relatórios e Métricas

### Relatório de Cobertura HTML

```bash
# Gerar relatório
pytest --cov=src --cov-report=html -v

# Abrir no navegador (Windows)
Start-Process htmlcov/index.html

# Abrir no navegador (Linux/Mac)
open htmlcov/index.html
```

### Relatório JUnit (CI/CD)

```bash
# Gerar XML para Jenkins, GitLab CI, etc.
pytest --junitxml=test-results.xml -v
```

### Relatório de Duração

```bash
# Mostrar top 10 testes mais lentos
pytest -v --durations=10

# Mostrar todos
pytest -v --durations=0

# Apenas testes > 1s
pytest -v --durations-min=1.0
```

---

## 🔥 Cenários Comuns de Uso

### Desenvolvimento Local - Validação Rápida
```bash
# Rodar apenas unit tests antes de commit
pytest tests/unit/ -v -n auto --tb=short

# ~10 segundos
```

### Pull Request - Validação Completa
```bash
# Rodar todos os testes com cobertura
pytest -v --cov=src --cov-report=html --cov-report=term-missing

# ~2-3 minutos
```

### CI/CD Pipeline - Validação Produção
```bash
# Rodar tudo com reports para CI
pytest -v \
  --cov=src \
  --cov-report=xml \
  --cov-report=term \
  --junitxml=test-results.xml \
  --maxfail=5 \
  --tb=short

# Timeout global de 10min
# ~5-10 minutos
```

### Debug de Teste Específico
```bash
# Rodar um teste com output completo e debugger
pytest tests/unit/test_model.py::test_lstm_forward_pass -vv -s --pdb --tb=long
```

### Testar Apenas Código Modificado
```bash
# Assumindo git status mostra arquivos modificados
pytest -v $(git diff --name-only | grep "^tests/.*\.py$")
```

---

## ⚠️ Troubleshooting

### Problema 1: "No tests collected"

**Causa:** Pytest não encontrou arquivos de teste.

**Solução:**
```bash
# Verificar pytest.ini está configurado
cat pytest.ini

# Ou especificar caminho explícito
pytest tests/ -v

# Ou usar pattern específico
pytest -v --collect-only  # Ver o que seria coletado
```

---

### Problema 2: "RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long"

**Causa:** ticker_ids criado como float ao invés de long.

**Solução:** Já corrigido em:
- `cli/train.py` linha 226
- `src/ml/training/trainer.py` linha 374
- `scripts/train_unified_model.py` linhas 388-402

Verifique que está usando a última versão do código.

---

### Problema 3: "ModuleNotFoundError: No module named 'src'"

**Causa:** PYTHONPATH não inclui raiz do projeto.

**Solução:**
```bash
# Opção 1: Rodar do diretório raiz
cd /path/to/stock-prediction-lstm-api
pytest -v

# Opção 2: Instalar pacote em modo editable
pip install -e .

# Opção 3: Setar PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
pytest -v
```

---

### Problema 4: Testes E2E falhando com timeout

**Causa:** GPU não disponível ou dados não baixados.

**Solução:**
```bash
# Verificar GPU
python -c "import torch; print(torch.cuda.is_available())"

# Baixar dados primeiro
python -m cli.main data --ticker PETR4.SA VALE3.SA

# Rodar com mais timeout
pytest tests/e2e/ -v --timeout=300

# Ou rodar em CPU
CUDA_VISIBLE_DEVICES="" pytest tests/e2e/ -v
```

---

### Problema 5: "MLflow experiment not found"

**Causa:** Tracking URI não configurado.

**Solução:**
```bash
# Criar diretório
mkdir -p data/mlflow/tracking

# Setar variável de ambiente
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/tracking

# Ou adicionar em .env
echo "MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/tracking" >> .env
```

---

## 📚 Recursos Adicionais

- **Pytest Docs**: https://docs.pytest.org/
- **Coverage.py Docs**: https://coverage.readthedocs.io/
- **MLflow Testing**: https://mlflow.org/docs/latest/python_api/mlflow.tracking.html
- **PyTorch Testing**: https://pytorch.org/docs/stable/testing.html

---

## 📞 Dúvidas ou Problemas?

1. Consulte [tests/README.md](../tests/README.md) para estrutura detalhada
2. Veja [docs/PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md) para status do projeto
3. Rode com `-vv -s --tb=long` para máximo debug
4. Use `pytest --pdb` para debugger interativo

---

**Última Atualização**: 09/01/2026 00:30 BRT  
**Mantido por**: Equipe FIAP Tech Challenges
