# ⚡ Quick Start - 5 Minutos

**Objetivo:** Treinar modelo PETR4.SA e testar API localmente em 5 minutos.

---

## 🚀 Setup Rápido

### **1. Clone e Instale Dependências** (1 min)

```bash
git clone https://github.com/adriannylelis/stock-prediction-lstm-api.git
cd stock-prediction-lstm-api

# Criar ambiente virtual
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar PyTorch CPU
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2.0"

# Instalar outras dependências
pip install -r requirements.txt
```

### **2. Treinar Modelo** (3 min)

```bash
./scripts/local_train.sh
```

**Ou manualmente:**
```bash
python -m cli train --ticker PETR4.SA --epochs 50
```

**Output esperado:**
```
✅ Training completed!
📦 Generated artifacts:
   model.pt (8.5MB)
   scaler.pkl (4KB)
```

### **3. Testar API** (1 min)

**Opção A - Docker (recomendado):**
```bash
docker-compose up backend
```

**Opção B - Python direto:**
```bash
python -m flask --app src.api.main:create_app run --port 5001
```

**Testar:**
```bash
# Health check
curl http://localhost:5001/health

# Predição
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "periods": 7}'
```

**Resposta esperada:**
```json
{
  "ticker": "PETR4.SA",
  "predictions": [
    {"date": "2026-01-12", "price": 35.42},
    {"date": "2026-01-13", "price": 35.58},
    ...
  ]
}
```

---

## 🎯 Próximos Passos

### **Deploy Automático:**
1. Configure GitHub Actions (já está pronto!)
2. Escolha plataforma: [Render.com](https://render.com) (grátis) ou [Railway.app](https://railway.app) ($5/mês)
3. Siga o guia: [docs/DEPLOY_FREE_TIER.md](docs/DEPLOY_FREE_TIER.md)

### **Desenvolvimento:**
- 📖 [Documentação Completa](docs/)
- 🏗️ [Arquitetura MLOps](docs/ARCHITECTURE_MLOPS.md)
- 🐳 [Guia Docker](docs/DOCKER_GUIDE.md)
- 🧪 [Como Rodar Testes](docs/RUN_TESTS.md)

---

## 🆘 Problemas?

**Erro: `ModuleNotFoundError`**
```bash
pip install -r requirements.txt
```

**Erro: `torch.OutOfMemoryError`**
```bash
python -m cli train --ticker PETR4.SA --batch-size 32 --epochs 50
```

**Erro: `No module named 'src'`**
```bash
# Rode a partir da raiz do projeto
cd stock-prediction-lstm-api
python -m cli train ...
```

---

✅ **Pronto! Você tem um modelo treinado e API funcionando!**
