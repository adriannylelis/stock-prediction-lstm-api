# 🚀 Quick Start Guide - Development Setup (estado atual)

## 📋 Pré-requisitos
- Python 3.11+
```bash
## 🔧 Setup rápido (Docker)

```bash
# Backend Flask (porta 5001)
docker run -p 8000:8000 -v $(pwd)/data:/app/data stock-prediction-ml

# Health e modelo
curl -s http://localhost:5001/health
curl -s http://localhost:5001/model/info

# Predição (único ticker suportado)

  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'
```

## 🐳 Notas de containerização (estado atual)
- Dockerfile único (Flask 5001) com PyTorch CPU; copia `configs/` para carregar `production_model.yaml`.
- docker-compose expõe 5001 e monta `./data` em `/app/data` (MLflow tracking local). Frontend mapeia VITE_API_URL para http://localhost:5001.
# Compose (recomendado)
docker-compose up -d
```

---

## 🎯 Workflow Recomendado

### Desenvolvimento Local
```bash
# 1. Setup inicial (1x)
./scripts/setup_ml.sh

# 2. Durante dev (toda vez que reiniciar)
./scripts/setup_backend.sh

# 3. Testes
pytest tests/unit/ -v
```

### Produção (Docker)
```bash
# Build
docker-compose build

# Deploy
docker-compose up -d

# Logs
docker-compose logs -f ml-api

# Stop
docker-compose down
```

---

## 📊 Comparação

| Método | Setup | Restart | Portabilidade | CI/CD |
|--------|-------|---------|---------------|-------|
| **Scripts bash** | ✅ Rápido | ✅ Muito rápido | ⚠️ Depende de SO | ⚠️ Requer setup |
| **Docker** | ⚠️ Lento (build) | ✅ Instantâneo | ✅ Total | ✅ Perfeito |

**Recomendação:**
- **Dev local:** Scripts bash (mais rápido para iterar)
- **Produção/CI:** Docker (consistência garantida)

---

## 🔑 Dicas Importantes para Containerização

### 1. Multi-stage builds (reduz tamanho)
```dockerfile
# Builder stage (descartado no final)
FROM python:3.13 AS builder
RUN pip install --user torch mlflow

# Runtime (apenas o necessário)
FROM python:3.13-slim
COPY --from=builder /root/.local /root/.local
```

### 2. Cache de dependências
```dockerfile
# Copiar requirements ANTES do código
COPY requirements-ml.txt .
RUN pip install -r requirements-ml.txt
# Código muda mais que deps, cache aproveitado
COPY . .
```

### 3. Volumes para dados
```yaml
volumes:
  - ./data/mlflow:/app/data/mlflow  # MLflow tracking persiste
  - ./artifacts:/app/artifacts      # Fallback models
```

### 4. .dockerignore
```
.venv/
data/mlflow/
*.pyc
__pycache__/
.git/
```

---

## ✅ Checklist de Setup

**Scripts criados:**
- [x] `setup_ml.sh` - Base ML environment
- [x] `setup_backend.sh` - Inicia API
- [ ] `setup_frontend.sh` - (se houver React/Vue)
- [ ] `Dockerfile` - Para containerização
- [ ] `docker-compose.yml` - Orquestração

**Boa prática confirmada:** ✅ SIM
- Separação modular
- Setup rápido no dia-a-dia
- Pronto para Docker quando necessário
