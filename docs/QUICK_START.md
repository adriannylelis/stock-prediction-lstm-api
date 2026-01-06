# 🚀 Como Iniciar o Projeto

## Opção 1: Docker Compose (Recomendado) 🐳

A maneira mais fácil de rodar o projeto completo:

```bash
# Na raiz do projeto
docker-compose up --build
```

**Aguarde ~5-10 minutos** na primeira vez (download de imagens + build).

✅ Acesse:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5001

📖 Ver guia completo: [DOCKER_GUIDE.md](DOCKER_GUIDE.md)

---

## Opção 2: Desenvolvimento Local

### Backend (API Flask)
```bash
# 1. Ativar ambiente virtual
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Linux/Mac

# 2. Instalar dependências (se ainda não fez)
pip install -r requirements.txt

# 3. Iniciar API
cd src/api
python main.py
```

API rodará em: http://localhost:5001

### Frontend (React)
```bash
# 1. Instalar dependências (se ainda não fez)
cd frontend
npm install

# 2. Iniciar dev server
npm run dev
```

Dashboard rodará em: http://localhost:3000

---

## Troubleshooting Rápido

### ❌ "Cannot start API"
**Solução**: Use Docker!
```bash
docker-compose up backend
```

### ❌ "Port 5001 already in use"
```bash
# Windows
netstat -ano | findstr :5001
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5001 | xargs kill -9
```

### ❌ "Module not found"
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

---

## Guias Detalhados

- 📦 [README.md](README.md) - Visão geral do projeto
- 🐳 [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Guia completo Docker
- 🎨 [frontend/README.md](frontend/README.md) - Frontend específico
- 🧪 [frontend/TESTING_GUIDE.md](frontend/TESTING_GUIDE.md) - Como testar
