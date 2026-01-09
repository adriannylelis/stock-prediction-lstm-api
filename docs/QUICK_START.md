# 🚀 Quick Start Guide - Development Setup

## 📋 Pré-requisitos
- Python 3.13 (recomendado) ou 3.11+
- Git Bash (Windows) ou terminal bash (Linux/macOS)
- Node.js 18+ (para frontend)

---

## 🔧 Setup Completo (3 comandos)

### 1️⃣ ML Environment (Base)
```bash
chmod +x scripts/setup_ml.sh
./scripts/setup_ml.sh
```
**O que faz:**
- Cria `.venv` com Python selecionado
- Instala PyTorch (CPU ou GPU)
- Instala MLflow, pandas, scikit-learn
- Instala FastAPI, pytest

**Duração:** ~5-10 min (depende da internet)

---

### 2️⃣ Backend API (usa o .venv do ML)
```bash
chmod +x scripts/setup_backend.sh
./scripts/setup_backend.sh
```
**O que faz:**
- Verifica deps da API (FastAPI)
- Cria `configs/production_model.yaml` se não existir
- **Inicia o backend em http://localhost:8000**

**Duração:** ~10s + servidor rodando

---

### 3️⃣ Frontend (se houver)
```bash
cd frontend/  # ou diretório do front
npm install
npm run dev
```

---

## 📦 Estrutura de Setup (Boa Prática ✅)

```
scripts/
├── setup_ml.sh          # Base: Python + ML libs (1x)
├── setup_backend.sh     # API: Inicia servidor (N vezes)
└── setup_frontend.sh    # Front: npm install + start (opcional)
```

**Vantagens dessa separação:**
1. ✅ **Modular:** ML setup 1x, backend reinicia N vezes
2. ✅ **Rápido:** Não reinstala tudo ao restartar API
3. ✅ **Claro:** Cada script tem responsabilidade única
4. ✅ **CI/CD friendly:** Fácil de usar em pipelines

---

## 🐳 Containerização (Resumo para seu colega)

### Dockerfile - ML + API
```dockerfile
# Stage 1: Base com Python
FROM python:3.13-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y git curl

# Stage 2: ML Dependencies
FROM base AS ml-builder
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Stage 3: Runtime
FROM ml-builder AS runtime
COPY . .
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

# Start API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml - Orquestração
```yaml
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data          # Dados persistem
      - ./configs:/app/configs    # Configs externas
    environment:
      - MLFLOW_TRACKING_URI=file:./data/mlflow/tracking
    restart: unless-stopped

  mlflow-ui:
    image: python:3.13-slim
    working_dir: /mlflow
    volumes:
      - ./data/mlflow:/mlflow
    ports:
      - "5000:5000"
    command: >
      sh -c "pip install mlflow && 
             mlflow ui --backend-store-uri file:./tracking --host 0.0.0.0"
```

### Comandos Docker
```bash
# Build
docker build -t stock-prediction-ml .

# Run
docker run -p 8000:8000 -v $(pwd)/data:/app/data stock-prediction-ml

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
