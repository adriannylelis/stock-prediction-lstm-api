# 🚀 Guia de Deploy - Stock Prediction LSTM API

Este documento fornece instruções detalhadas para fazer deploy da API em diferentes plataformas de hosting.

---

## 📋 Requisitos

- Python 3.11+
- PyTorch 2.2.2 (CPU-only)
- NumPy < 2.0
- Flask 3.1+
- Docker (opcional)
- 2.5GB de espaço em disco (para imagem Docker)
- 512MB RAM mínimo (recomendado: 1GB)

---

## 🐳 Deploy com Docker (Recomendado)

### **Opção 1: Docker Local**

```bash
# 1. Clone o repositório
git clone https://github.com/adriannylelis/stock-prediction-lstm-api.git
cd stock-prediction-lstm-api

# 2. Build da imagem
docker build -t stock-prediction-api:latest .

# 3. Rodar container
docker run -d \
  --name stock-api \
  -p 5001:5001 \
  --restart unless-stopped \
  stock-prediction-api:latest

# 4. Verificar logs
docker logs -f stock-api

# 5. Testar API
curl http://localhost:5001/health
```

### **Opção 2: Docker Compose**

Crie um arquivo `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    image: stock-prediction-api:latest
    build: .
    container_name: stock-api
    ports:
      - "5001:5001"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
```

Comandos:
```bash
# Build e run
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar
docker-compose down
```

---

## ☁️ Deploy em Cloud Providers

### **1. Render.com (Recomendado para Docker)**

**Características:**
- ✅ Free tier disponível
- ✅ Deploy automático via GitHub
- ✅ Suporte nativo a Docker
- ✅ SSL/HTTPS automático

**Passos:**

1. **Criar conta** em [render.com](https://render.com)

2. **Novo Web Service:**
   - Click em "New +" → "Web Service"
   - Connect ao seu repositório GitHub
   - Configure:
     ```
     Name: stock-prediction-api
     Environment: Docker
     Branch: main (ou dev-adri)
     ```

3. **Configurações avançadas:**
   ```
   Docker Command: (deixar vazio - usa CMD do Dockerfile)
   Port: 5001
   Health Check Path: /health
   ```

4. **Deploy:**
   - Click em "Create Web Service"
   - Aguardar build (~10 minutos na primeira vez)
   - URL será: https://stock-prediction-api.onrender.com

**Limitações do Free Tier:**
- Container dorme após 15 minutos de inatividade
- Primeiro request pode levar 30-60s (cold start)
- 750 horas/mês grátis

---

### **2. Railway.app**

**Características:**
- ✅ $5 de crédito grátis por mês
- ✅ Deploy super rápido
- ✅ Auto-scaling
- ✅ Suporte a Docker

**Passos:**

1. **Criar conta** em [railway.app](https://railway.app)

2. **Novo Projeto:**
   - Click em "New Project"
   - Selecione "Deploy from GitHub repo"
   - Escolha seu repositório

3. **Configurações:**
   - Railway detecta automaticamente o Dockerfile
   - Variáveis de ambiente: (nenhuma necessária)
   - Port: 5001 (detectado automaticamente)

4. **Deploy:**
   - Deploy automático a cada push no GitHub
   - URL gerada automaticamente
   - Logs em tempo real

**Custo:**
- $5 de crédito grátis/mês
- ~$0.01/hora depois do crédito

---

### **3. Fly.io**

**Características:**
- ✅ Free tier com 3 VMs
- ✅ Deploy global (CDN)
- ✅ Melhor performance

**Passos:**

1. **Instalar Fly CLI:**
```bash
curl -L https://fly.io/install.sh | sh
```

2. **Login:**
```bash
fly auth login
```

3. **Inicializar app:**
```bash
fly launch
# Escolha:
# - Nome: stock-prediction-api
# - Region: São Paulo (gru) ou mais próximo
# - Skip PostgreSQL
```

4. **Deploy:**
```bash
fly deploy
```

5. **Abrir app:**
```bash
fly open
```

**Comandos úteis:**
```bash
fly logs          # Ver logs
fly status        # Status da app
fly scale count 1 # Escalar para 1 instância
```

---

### **4. Heroku (Legacy)**

**Características:**
- ⚠️ Não tem mais free tier
- ⚠️ Mínimo $7/mês
- ✅ Fácil de usar

**Passos:**

1. **Instalar Heroku CLI:**
```bash
brew install heroku/brew/heroku
```

2. **Login:**
```bash
heroku login
```

3. **Criar app:**
```bash
heroku create stock-prediction-api
heroku stack:set container  # Usar Docker
```

4. **Deploy:**
```bash
git push heroku main
```

5. **Abrir:**
```bash
heroku open
```

**Limitações:**
- Mínimo $7/mês (Eco Dyno)
- Dorme após 30 min de inatividade

---

## 🖥️ Deploy em VPS (Digital Ocean, AWS EC2, etc.)

### **Requisitos do Servidor:**
- Ubuntu 20.04+ ou Debian 11+
- 1GB RAM mínimo (2GB recomendado)
- 10GB disco
- Docker instalado

### **Passos:**

1. **Conectar via SSH:**
```bash
ssh user@seu-servidor.com
```

2. **Instalar Docker:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

3. **Clonar repositório:**
```bash
git clone https://github.com/adriannylelis/stock-prediction-lstm-api.git
cd stock-prediction-lstm-api
```

4. **Build e Run:**
```bash
sudo docker build -t stock-api .
sudo docker run -d \
  --name stock-api \
  -p 80:5001 \
  --restart unless-stopped \
  stock-api
```

5. **Configurar Nginx (opcional):**
```nginx
server {
    listen 80;
    server_name seu-dominio.com;

    location / {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🔒 Variáveis de Ambiente

A API não requer variáveis de ambiente para funcionar. Tudo é configurado via arquivos:

- Modelo: `artifacts/model_lstm_1x16.pt`
- Scaler: `artifacts/scaler_corrected.pkl`
- Config: `artifacts/model_config.json`

**Opcional (para produção):**
```bash
# Se quiser desabilitar debug mode
export FLASK_ENV=production

# Custom port (padrão: 5001)
export PORT=8080
```

---

## 📊 Monitoramento

### **Healthcheck Endpoint**

```bash
# Verificar se API está UP
curl https://sua-api.com/health

# Response esperado:
{
  "status": "healthy",
  "timestamp": "2025-12-30T...",
  "service": "stock-prediction-lstm-api"
}
```

### **Logs**

**Docker:**
```bash
docker logs -f stock-api
```

**Railway/Render:**
- Logs disponíveis no dashboard

**Fly.io:**
```bash
fly logs
```

---

## 🧪 Testar Deployment

Após fazer deploy, teste todos os endpoints:

```bash
# Substitua YOUR_URL pela URL do seu deploy
BASE_URL="https://stock-prediction-api.onrender.com"

# 1. Health check
curl $BASE_URL/health

# 2. Model info
curl $BASE_URL/model/info

# 3. Prediction
curl -X POST $BASE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

---

## ⚠️ Troubleshooting

### **Problema: Container não inicia**

**Solução:**
```bash
# Ver logs do container
docker logs stock-api

# Verificar se porta está livre
lsof -i :5001

# Rebuild forçado
docker build --no-cache -t stock-api .
```

### **Problema: NumPy version error**

**Solução:**
- Verificar se Dockerfile instala `numpy<2.0`
- Rebuild da imagem

### **Problema: Out of Memory**

**Solução:**
- Aumentar RAM do servidor (mínimo 1GB)
- Usar apenas 1 worker do Flask

### **Problema: API lenta (cold start)**

**Solução:**
- Usar health check para manter container aquecido
- Configurar keep-alive:
```bash
# Criar cronjob que pinga /health a cada 10 min
*/10 * * * * curl https://sua-api.com/health
```

---

## 📞 Suporte

- **Issues:** [GitHub Issues](https://github.com/adriannylelis/stock-prediction-lstm-api/issues)
- **Documentação:** [README.md](../README.md)
- **Plano de Implementação:** [PLANO_PESSOA_B.md](PLANO_PESSOA_B.md)

---

## 🎯 Recomendação Final

Para produção, recomendamos:

1. **Hobby/Teste:** Render.com (free tier)
2. **Produção leve:** Railway.app ($5/mês)
3. **Produção pesada:** Fly.io ou VPS com Docker
4. **Enterprise:** AWS ECS/EKS com Fargate

---

**Última atualização:** Dezembro 2025
