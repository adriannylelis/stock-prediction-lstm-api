# 🌐 Deploy Completo na Google Cloud Platform

**Stack Completa:** Backend API (Flask + LSTM) + Frontend (React + Vite)  
**Ticker:** PETR4.SA (Petrobras)  
**Custo Estimado:** ~$4-7/mês  
**Última Atualização:** 11 de janeiro de 2026

---

## 📋 Índice

1. [Visão Geral da Arquitetura](#-visão-geral-da-arquitetura)
2. [Pré-requisitos](#-pré-requisitos)
3. [Setup Inicial do GCP](#-setup-inicial-do-gcp)
4. [Configurar GitHub Secrets](#-configurar-github-secrets)
5. [Deploy Automático](#-deploy-automático)
6. [Deploy Manual (Alternativa)](#-deploy-manual-alternativa)
7. [Monitoramento e Logs](#-monitoramento-e-logs)
8. [Custos Detalhados](#-custos-detalhados)
9. [Troubleshooting](#-troubleshooting)

---

## 🏗️ Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│  GITHUB ACTIONS (CI/CD - Grátis)                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 🤖 Train Model Weekly (Domingo 00:00)                 │ │
│  │  ├─ Train LSTM on PETR4.SA                            │ │
│  │  ├─ Create GitHub Release                             │ │
│  │  └─ Trigger Deploy Workflow                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 🌐 Deploy to Google Cloud                             │ │
│  │  ├─ Download artifacts from Release                   │ │
│  │  ├─ Build Backend Docker (Cloud Build)                │ │
│  │  ├─ Deploy Backend (Cloud Run)                        │ │
│  │  ├─ Build Frontend Docker (Cloud Build)               │ │
│  │  └─ Deploy Frontend (Cloud Run)                       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  GOOGLE CLOUD PLATFORM                                      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Cloud Run - Backend API (Flask + PyTorch)          │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  📦 Container: stock-api-backend               │  │  │
│  │  │  📍 URL: https://stock-api-backend-xxx.run.app │  │  │
│  │  │  💾 Memory: 512MB                               │  │  │
│  │  │  🔢 CPU: 1 vCPU                                 │  │  │
│  │  │  📊 Endpoints:                                  │  │  │
│  │  │     - GET  /health                              │  │  │
│  │  │     - GET  /model-info                          │  │  │
│  │  │     - POST /predict                             │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         │ (API calls)                       │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Cloud Run - Frontend (React + Nginx)               │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  🎨 Container: stock-api-frontend              │  │  │
│  │  │  📍 URL: https://stock-api-frontend-xxx.run.app│  │  │
│  │  │  💾 Memory: 256MB                               │  │  │
│  │  │  🔢 CPU: 1 vCPU                                 │  │  │
│  │  │  📦 Static files servidos via Nginx            │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Container Registry (GCR)                            │  │
│  │  ├─ gcr.io/project/stock-api-backend:latest         │  │
│  │  └─ gcr.io/project/stock-api-frontend:latest        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Secret Manager (Opcional)                           │  │
│  │  └─ Credentials, API keys, etc.                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
                 👥 USUÁRIOS FINAIS
```

---

## 🔧 Pré-requisitos

### **1. Google Cloud Account**

- [ ] Conta Google Cloud criada
- [ ] Billing habilitado (cartão de crédito necessário)
- [ ] Free Tier de $300 (válido por 90 dias para novos usuários)

**Criar conta:** https://console.cloud.google.com

### **2. Ferramentas Instaladas**

```bash
# macOS
brew install --cask google-cloud-sdk
brew install jq

# Linux (Debian/Ubuntu)
curl https://sdk.cloud.google.com | bash
sudo apt-get install jq

# Verificar instalação
gcloud --version
```

### **3. Repositório GitHub**

- [ ] Repositório fork/clone de `stock-prediction-lstm-api`
- [ ] Permissões de administrador (para configurar Secrets)

---

## ⚙️ Setup Inicial do GCP

### **Opção A: Script Automático (Recomendado) ⭐**

```bash
./scripts/setup_gcloud.sh
```

Este script faz automaticamente:
1. ✅ Cria/configura projeto GCP
2. ✅ Habilita APIs necessárias
3. ✅ Cria Service Account
4. ✅ Atribui permissões
5. ✅ Gera chave JSON

**Tempo:** ~5 minutos

---

### **Opção B: Setup Manual**

<details>
<summary>Clique para expandir passos manuais</summary>

#### **1. Criar Projeto**

```bash
# Listar projetos existentes
gcloud projects list

# Criar novo projeto
gcloud projects create stock-ml-prod --name="Stock ML Production"

# Definir como projeto ativo
gcloud config set project stock-ml-prod
```

#### **2. Habilitar APIs**

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

#### **3. Habilitar Billing**

```bash
# Via console (mais fácil)
open "https://console.cloud.google.com/billing/linkedaccount?project=stock-ml-prod"

# Ou via CLI (requer billing account ID)
gcloud billing projects link stock-ml-prod --billing-account=BILLING_ACCOUNT_ID
```

#### **4. Criar Service Account**

```bash
# Criar Service Account
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions Service Account"

# Obter email
SA_EMAIL="github-actions-sa@stock-ml-prod.iam.gserviceaccount.com"

# Atribuir permissões
gcloud projects add-iam-policy-binding stock-ml-prod \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding stock-ml-prod \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding stock-ml-prod \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/cloudbuild.builds.editor"

gcloud projects add-iam-policy-binding stock-ml-prod \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/iam.serviceAccountUser"
```

#### **5. Gerar Chave JSON**

```bash
gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=$SA_EMAIL

# Visualizar conteúdo
cat gcp-key.json
```

</details>

---

## 🔐 Configurar GitHub Secrets

### **1. Acessar Configurações**

No seu repositório GitHub:
```
Settings → Secrets and variables → Actions → New repository secret
```

### **2. Adicionar Secrets**

#### **Secret 1: GCP_PROJECT_ID**
```
Name: GCP_PROJECT_ID
Value: stock-ml-prod
```

#### **Secret 2: GCP_SA_KEY**
```
Name: GCP_SA_KEY
Value: <cole o conteúdo completo do gcp-key.json>
```

**Dica - Copiar conteúdo do arquivo:**
```bash
# macOS
cat gcp-key.json | pbcopy

# Linux
cat gcp-key.json | xclip -selection clipboard

# Windows
type gcp-key.json | clip
```

#### **Secrets Opcionais:**

```
RENDER_DEPLOY_HOOK  # Se quiser manter Render como fallback
RENDER_URL          # URL do Render (se usar)
```

---

## 🚀 Deploy Automático

### **Fluxo Completo:**

1. **Treino Semanal (Automático)**
   - Todo domingo às 00:00 UTC
   - Workflow: `🤖 Train Model Weekly`
   - Cria GitHub Release com artifacts

2. **Deploy Após Treino (Automático)**
   - Trigger: Após treino bem-sucedido
   - Workflow: `🌐 Deploy to Google Cloud`
   - Deploya Backend + Frontend

### **Primeiro Deploy (Manual):**

#### **1. Executar Treino**

```
GitHub → Actions → 🤖 Train Model Weekly → Run workflow
```

Aguarde ~10 minutos. Verificar:
- ✅ Release criada em `Releases`
- ✅ `artifacts.zip` presente no Release

#### **2. Executar Deploy GCloud**

```
GitHub → Actions → 🌐 Deploy to Google Cloud → Run workflow
```

Opções:
- ✅ Deploy Backend API
- ✅ Deploy Frontend

Aguarde ~15 minutos.

#### **3. Obter URLs**

Após deploy bem-sucedido, check o **Job Summary**:

```
Frontend: https://stock-api-frontend-xxx-uc.a.run.app
Backend:  https://stock-api-backend-xxx-uc.a.run.app
```

#### **4. Testar**

```bash
# Frontend
open https://stock-api-frontend-xxx-uc.a.run.app

# Backend Health
curl https://stock-api-backend-xxx-uc.a.run.app/health

# Prediction
curl -X POST https://stock-api-backend-xxx-uc.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "periods": 7}'
```

---

## 🛠️ Deploy Manual (Alternativa)

### **Via gcloud CLI:**

```bash
# 1. Autenticar
gcloud auth login
gcloud config set project stock-ml-prod

# 2. Build Backend
gcloud builds submit --tag gcr.io/stock-ml-prod/stock-api-backend .

# 3. Deploy Backend
gcloud run deploy stock-api-backend \
  --image gcr.io/stock-ml-prod/stock-api-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --set-env-vars "FLASK_ENV=production,MODEL_TICKER=PETR4.SA"

# 4. Build Frontend
cd frontend
gcloud builds submit --tag gcr.io/stock-ml-prod/stock-api-frontend .

# 5. Deploy Frontend
gcloud run deploy stock-api-frontend \
  --image gcr.io/stock-ml-prod/stock-api-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 256Mi \
  --cpu 1
```

---

## 📊 Monitoramento e Logs

### **1. Cloud Run Dashboard**

```
https://console.cloud.google.com/run?project=stock-ml-prod
```

Métricas disponíveis:
- ✅ Request count
- ✅ Request latency
- ✅ Container instance count
- ✅ Billing data

### **2. Logs**

```bash
# Logs do Backend (última hora)
gcloud run services logs read stock-api-backend \
  --region us-central1 \
  --limit 50

# Logs do Frontend
gcloud run services logs read stock-api-frontend \
  --region us-central1 \
  --limit 50

# Tail logs (tempo real)
gcloud run services logs tail stock-api-backend --region us-central1
```

### **3. Alertas (Opcional)**

```bash
# Criar alerta para erro 500
gcloud monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="API 5xx Errors" \
  --condition-display-name="High 5xx rate" \
  --condition-threshold-value=10 \
  --condition-threshold-duration=60s
```

---

## 💰 Custos Detalhados

### **Calculadora de Custos**

| Serviço | Uso Mensal | Free Tier | Custo (USD) |
|---------|------------|-----------|-------------|
| **Cloud Run - Backend** | 100k requests, 512MB, 1 CPU | 2M requests, 180k vCPU-s, 360k GiB-s | $3-5 |
| **Cloud Run - Frontend** | 100k requests, 256MB, 1 CPU | Incluído acima | $1-2 |
| **Cloud Build** | 4 builds/mês (20min cada) | 120 min/dia grátis | $0 |
| **Container Registry** | ~1GB storage | 0.5GB grátis | $0.02 |
| **Egress** | 10GB/mês | 1GB/mês grátis (NA) | $1 |
| **Total Estimado** | | | **$4-8/mês** |

### **Cenários de Uso:**

**Baixo Tráfego (< 10k req/dia):**
- Custo: $0-2/mês (dentro do Free Tier)

**Médio Tráfego (10-50k req/dia):**
- Custo: $4-6/mês

**Alto Tráfego (100k+ req/dia):**
- Custo: $10-15/mês

### **Otimizações para Reduzir Custos:**

1. **Min Instances = 0:** Evita custos quando ocioso
2. **CPU always allocated = false:** Cobra apenas durante request
3. **Timeout baixo:** Máximo 300s (padrão: 60s)
4. **Memory otimizada:** Backend 512MB, Frontend 256MB

---

## 🔍 Troubleshooting

### **Problema: Cloud Build falha com "Permission Denied"**

**Erro:**
```
ERROR: (gcloud.builds.submit) User [...] does not have permission
```

**Solução:**
```bash
# Verificar Service Account
gcloud projects get-iam-policy stock-ml-prod \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:github-actions-sa*"

# Re-adicionar permissões
gcloud projects add-iam-policy-binding stock-ml-prod \
  --member="serviceAccount:github-actions-sa@stock-ml-prod.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.editor"
```

---

### **Problema: Deploy falha com "Billing not enabled"**

**Solução:**
1. Acessar: https://console.cloud.google.com/billing
2. Vincular cartão de crédito ao projeto
3. Habilitar Billing para `stock-ml-prod`

---

### **Problema: Backend retorna 500 - "Model not loaded"**

**Causa:** Artifacts não foram baixados durante build.

**Solução:**
1. Verificar se GitHub Release existe
2. Verificar logs do Cloud Build:
   ```bash
   gcloud builds list --limit=5
   gcloud builds log [BUILD_ID]
   ```
3. Verificar se arquivo `artifacts.zip` está no Release
4. Re-executar deploy

---

### **Problema: Frontend não conecta ao Backend**

**Causa:** URL do backend não foi configurada corretamente.

**Solução:**
1. Obter URL do backend:
   ```bash
   gcloud run services describe stock-api-backend \
     --region us-central1 \
     --format='value(status.url)'
   ```
2. Verificar nginx.conf do frontend:
   ```bash
   # Deve ter a URL correta do backend
   proxy_pass https://stock-api-backend-xxx.run.app;
   ```
3. Rebuild e re-deploy frontend

---

### **Problema: "Cold start" muito lento (>30s)**

**Solução 1 - Aumentar min instances:**
```bash
gcloud run services update stock-api-backend \
  --region us-central1 \
  --min-instances 1  # Mantém 1 instância sempre ativa
```

**Custo adicional:** ~$8-10/mês

**Solução 2 - Otimizar Docker:**
- Usar imagens base menores
- Multi-stage builds
- Reduzir dependências

---

## 🎯 Próximos Passos

### **Melhorias Recomendadas:**

1. **Custom Domain:**
   ```bash
   gcloud run domain-mappings create \
     --service stock-api-frontend \
     --domain app.stockml.com \
     --region us-central1
   ```

2. **CDN (para Frontend):**
   - Migrar frontend para Cloud Storage + Cloud CDN
   - Custo menor (~$0.50/mês)
   - Melhor performance

3. **Cloud SQL (para MLflow):**
   - Tracking Server persistente
   - Experimentos versionados

4. **Monitoring Avançado:**
   - Google Cloud Monitoring
   - Uptime checks
   - Error reporting

---

## 📚 Referências

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Pricing Calculator](https://cloud.google.com/products/calculator)
- [Best Practices](https://cloud.google.com/run/docs/best-practices)

---

✅ **Deploy completo na GCloud configurado!**

**Resumo:**
- ✅ Backend API em Cloud Run
- ✅ Frontend React em Cloud Run
- ✅ Deploy automático via GitHub Actions
- ✅ Treino semanal automatizado
- ✅ Custo: ~$4-8/mês
- ✅ Escalável e production-ready
