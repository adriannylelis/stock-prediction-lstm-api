# 🐳 Docker Quick Start Guide

## 🚀 Início Rápido (Recomendado)

Execute o projeto completo com um único comando:

```bash
docker-compose up --build
```

**Aguarde alguns minutos** (primeira vez demora ~5-10min para baixar imagens e buildar).

Quando ver:
```
backend_1   | * Running on http://0.0.0.0:5001
frontend_1  | * Ready on http://localhost:80
```

Acesse:
- 🎨 **Frontend**: http://localhost:3000
- 🔌 **Backend API**: http://localhost:5001
- 📊 **API Health**: http://localhost:5001/health

---

## 📦 O que o Docker Compose faz?

```
docker-compose.yml
├── backend (Flask API)
│   ├── Build: Dockerfile na raiz
│   ├── Porta: 5001
│   ├── Healthcheck: curl http://localhost:5001/health
│   └── Volume: ./artifacts (modelos LSTM)
│
└── frontend (React + Nginx)
    ├── Build: frontend/Dockerfile
    ├── Porta: 3000 (mapeada para 80 interno)
    ├── Depende: backend (espera health check)
    └── Nginx: serve static files otimizado
```

---

## 🛠️ Comandos Úteis

### Iniciar Serviços
```bash
# Build e start (primeira vez)
docker-compose up --build

# Start em background
docker-compose up -d

# Apenas backend
docker-compose up backend

# Apenas frontend
docker-compose up frontend
```

### Parar Serviços
```bash
# Parar containers (mantém volumes)
docker-compose stop

# Parar e remover containers
docker-compose down

# Remover containers + volumes + imagens
docker-compose down -v --rmi all
```

### Ver Logs
```bash
# Todos os serviços
docker-compose logs -f

# Apenas backend
docker-compose logs -f backend

# Apenas frontend
docker-compose logs -f frontend

# Últimas 100 linhas
docker-compose logs --tail=100 -f
```

### Rebuild Específico
```bash
# Rebuild apenas backend
docker-compose build backend
docker-compose up -d backend

# Rebuild apenas frontend
docker-compose build frontend
docker-compose up -d frontend
```

### Verificar Status
```bash
# Ver containers rodando
docker-compose ps

# Verificar health
docker inspect stock-api-backend | grep Health -A 10
```

---

## 🐛 Troubleshooting

### Problema: "Port already in use"
```bash
# Backend (5001)
netstat -ano | findstr :5001
taskkill /PID <PID> /F

# Frontend (3000)
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Ou mudar porta no docker-compose.yml:
ports:
  - "5002:5001"  # backend
  - "3001:80"    # frontend
```

### Problema: Backend não responde
```bash
# Verificar logs
docker-compose logs backend

# Entrar no container
docker exec -it stock-api-backend bash

# Testar health manualmente
docker exec stock-api-backend curl http://localhost:5001/health
```

### Problema: Frontend não conecta no backend
```bash
# Verificar network
docker network inspect stock-prediction-lstm-api_stock-network

# Verificar variável de ambiente
docker exec stock-api-frontend env | grep VITE_API_URL

# Se necessário, ajustar frontend/.env:
VITE_API_URL=http://localhost:5001
```

### Problema: Build falha por falta de memória
```bash
# Aumentar memória do Docker Desktop
# Settings → Resources → Memory → 4GB+

# Ou buildar separadamente
docker-compose build --no-cache backend
docker-compose build --no-cache frontend
```

---

## 🔄 Workflow de Desenvolvimento

### Opção 1: Docker Compose (Recomendado para Produção)
```bash
docker-compose up
```

### Opção 2: Desenvolvimento Local (Hot Reload)
```bash
# Terminal 1 - Backend local
source venv/Scripts/activate  # Windows
python src/api/main.py

# Terminal 2 - Frontend local
cd frontend
npm run dev
```

### Opção 3: Backend Docker + Frontend Local
```bash
# Backend no Docker
docker-compose up backend

# Frontend local com hot reload
cd frontend
npm run dev
```

---

## 📊 Estrutura de Arquivos Docker

```
stock-prediction-lstm-api/
├── Dockerfile              # Backend (Flask)
├── docker-compose.yml      # Orquestração
├── requirements.txt        # Deps Python
├── artifacts/              # Modelos LSTM (volume)
│
└── frontend/
    ├── Dockerfile          # Frontend (React + Nginx)
    ├── nginx.conf          # Config Nginx
    ├── package.json
    └── src/
```

---

## 🚀 Deploy em Produção

### Docker Hub (Opcional)
```bash
# Build e tag
docker build -t seu-usuario/stock-api-backend:latest .
docker build -t seu-usuario/stock-api-frontend:latest ./frontend

# Push
docker push seu-usuario/stock-api-backend:latest
docker push seu-usuario/stock-api-frontend:latest
```

### Docker Swarm / Kubernetes
Use os arquivos Docker como base para:
- Deployment manifests
- Service definitions
- ConfigMaps para variáveis de ambiente

---

## ⚡ Otimizações Aplicadas

### Backend (Dockerfile)
✅ Multi-stage build (não aplicado, mas pode adicionar)  
✅ PyTorch CPU-only (~500MB vs 2GB CUDA)  
✅ Layer caching (requirements primeiro)  
✅ Non-root user (segurança)  
✅ Healthcheck integrado  

### Frontend (Dockerfile + Nginx)
✅ Multi-stage build (Node builder + Nginx runtime)  
✅ Nginx otimizado para SPA (try_files)  
✅ Gzip compression  
✅ Cache de assets estáticos (1 ano)  
✅ Security headers (X-Frame-Options, etc)  
✅ Tamanho final: ~20MB (vs ~300MB com Node)  

---

## 📈 Monitoramento

### Healthchecks
```bash
# Backend health
curl http://localhost:5001/health

# Frontend health
curl http://localhost:3000

# Docker health status
docker-compose ps
```

### Métricas de Recursos
```bash
# Ver uso de CPU/Memória
docker stats

# Apenas nossos containers
docker stats stock-api-backend stock-api-frontend
```

---

## 🎯 Checklist de Deployment

Antes de considerar pronto para produção:

- [ ] `docker-compose up` funciona sem erros
- [ ] Backend responde em http://localhost:5001/health
- [ ] Frontend abre em http://localhost:3000
- [ ] Predição funciona (testar com AAPL)
- [ ] Logs sem warnings críticos
- [ ] Healthchecks passando (docker-compose ps)
- [ ] Reinício automático funciona (restart: unless-stopped)
- [ ] Variáveis de ambiente corretas
- [ ] Artifacts (modelos) montados corretamente
- [ ] Network isolada criada (stock-network)

---

## 🆘 Suporte

**Problemas?** Verifique logs:
```bash
docker-compose logs -f
```

**Resetar tudo:**
```bash
docker-compose down -v
docker system prune -a
docker-compose up --build
```

---

**Status**: ✅ Docker Compose configurado e pronto para uso!
