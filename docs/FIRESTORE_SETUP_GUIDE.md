# 🔥 Guia de Setup do Firestore

Este documento descreve como configurar o Google Cloud Firestore para o sistema de histórico de predições.

## 📋 Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Setup Inicial no GCloud](#setup-inicial-no-gcloud)
3. [Configuração Local](#configuração-local)
4. [Deployment em Produção](#deployment-em-produção)
5. [Verificação e Testes](#verificação-e-testes)
6. [Atualização de Preços Reais](#atualização-de-preços-reais)
7. [Monitoramento](#monitoramento)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 Pré-requisitos

Antes de começar, certifique-se de ter:

- ✅ Conta Google Cloud Platform ativa
- ✅ Projeto GCP criado (ex: `stock-prediction-prod`)
- ✅ `gcloud` CLI instalado ([Download](https://cloud.google.com/sdk/docs/install))
- ✅ Permissões de Owner ou Editor no projeto
- ✅ Billing habilitado no projeto

---

## 🚀 Setup Inicial no GCloud

### 📍 Onde Executar?

Você pode executar estes comandos em **2 lugares**:

**Opção 1: Terminal Local (Recomendado)** 
- No seu MacBook, no diretório do projeto
- Caminho atual: `/Users/adriannylelis/Workspace/stock-prediction-lstm-api`
- Requer: `gcloud` CLI instalado localmente

**Opção 2: Cloud Shell (Alternativa)**
- Navegador: [Console GCP](https://console.cloud.google.com) → Botão Cloud Shell (ícone `>_`)
- Já tem `gcloud` pré-instalado
- Precisa clonar repositório primeiro

### Passo 1: Autenticar no gcloud

**No seu terminal local (MacBook):**

```bash
# 1. Verificar se gcloud está instalado
gcloud --version

# Se não tiver instalado, baixar de: https://cloud.google.com/sdk/docs/install

# 2. Login no gcloud (abre navegador para autenticação)
gcloud auth login

# 3. Definir projeto padrão (substitua SEU_PROJECT_ID)
gcloud config set project SEU_PROJECT_ID

# 4. Verificar configuração
gcloud config list
```

### Passo 2: Executar Script de Setup Automático

**No seu terminal local, dentro do diretório do projeto:**

```bash
# Você já está aqui: /Users/adriannylelis/Workspace/stock-prediction-lstm-api

# 1. Verificar que o script existe
ls -la scripts/setup_firestore.sh

# 2. Executar setup (substitua SEU_PROJECT_ID pelo ID real do seu projeto)
./scripts/setup_firestore.sh SEU_PROJECT_ID

# Ou especificando região explicitamente:
./scripts/setup_firestore.sh SEU_PROJECT_ID us-central1

# Exemplo com projeto real:
# ./scripts/setup_firestore.sh stock-prediction-prod
```

**O que o script faz:**
- ✅ Habilita APIs: `firestore.googleapis.com` e `appengine.googleapis.com`
- ✅ Cria banco de dados Firestore em modo nativo
- ✅ Configura permissões IAM para service account
- ✅ Valida configurações

### Passo 3: Setup Manual (Alternativa)

Se preferir fazer manualmente:

```bash
# 1. Habilitar APIs
gcloud services enable firestore.googleapis.com
gcloud services enable appengine.googleapis.com

# 2. Criar banco de dados Firestore
gcloud firestore databases create \
  --location=us-central1 \
  --type=firestore-native

# 3. Configurar permissões IAM
SERVICE_ACCOUNT="SEU_PROJECT_ID@appspot.gserviceaccount.com"

gcloud projects add-iam-policy-binding SEU_PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/datastore.user"

# 4. Verificar
gcloud firestore databases list
```

### Passo 4: Verificar Configuração

```bash
# Listar bancos de dados Firestore
gcloud firestore databases list --project=SEU_PROJECT_ID

# Verificar região
gcloud firestore databases describe --database="(default)"
```

**Saída esperada:**
```
name: projects/SEU_PROJECT_ID/databases/(default)
locationId: us-central1
type: FIRESTORE_NATIVE
```

---

## 💻 Configuração Local

### Desenvolvimento com Emulador

Para testar localmente sem usar recursos GCP reais:

```bash
# 1. Subir emulador Firestore via Docker Compose
docker-compose up firestore backend

# 2. O backend detecta automaticamente o emulador via variável:
# FIRESTORE_EMULATOR_HOST=firestore:8080
```

### Acessar Firestore UI (Opcional)

```bash
# Firestore Emulator UI estará disponível em:
http://localhost:4000
```

### Testar Localmente

```bash
# 1. Fazer uma predição
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'

# 2. Verificar histórico salvo
curl http://localhost:5001/analytics/PETR4.SA

# 3. Ver métricas de acurácia
curl http://localhost:5001/analytics/PETR4.SA/accuracy
```

---

## 🌐 Deployment em Produção

### Opção 1: Deploy via GitHub Actions (Recomendado)

O workflow já está configurado para setup automático do Firestore:

```bash
# 1. Certifique-se que os secrets estão configurados no GitHub:
# - GCP_PROJECT_ID
# - GCP_SA_KEY (Service Account JSON)

# 2. Push para master ou trigger manualmente
git add .
git commit -m "feat: add Firestore prediction history"
git push origin master
```

O workflow `.github/workflows/deploy-gcloud.yml` irá:
- ✅ Habilitar Firestore API
- ✅ Criar banco de dados se necessário
- ✅ Configurar permissões IAM
- ✅ Deploy backend com `GOOGLE_CLOUD_PROJECT` configurado

### Opção 2: Deploy via Cloud Build

Se estiver usando `cloudbuild.yaml`:

```bash
gcloud builds submit --config=cloudbuild.yaml
```

O arquivo já está configurado com `GOOGLE_CLOUD_PROJECT=$PROJECT_ID`.

### Opção 3: Deploy Manual via gcloud

```bash
# 1. Build da imagem
gcloud builds submit --tag gcr.io/SEU_PROJECT_ID/stock-api-backend

# 2. Deploy no Cloud Run
gcloud run deploy stock-api-backend \
  --image gcr.io/SEU_PROJECT_ID/stock-api-backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=SEU_PROJECT_ID"
```

---

## ✅ Verificação e Testes

### 1. Testar Endpoints em Produção

Substitua `YOUR_BACKEND_URL` pela URL do Cloud Run:

```bash
BACKEND_URL="https://stock-api-backend-xxxx-uc.a.run.app"

# 1. Health check
curl $BACKEND_URL/health

# 2. Fazer predição (salva automaticamente no Firestore)
curl -X POST $BACKEND_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'

# 3. Buscar histórico de predições
curl $BACKEND_URL/analytics/PETR4.SA

# 4. Ver métricas de acurácia
curl $BACKEND_URL/analytics/PETR4.SA/accuracy

# 5. Ver predições pendentes (sem preço real ainda)
curl $BACKEND_URL/analytics/PETR4.SA/pending
```

### 2. Verificar Dados no Console GCP

```bash
# Abrir console Firestore
open "https://console.cloud.google.com/firestore/data?project=SEU_PROJECT_ID"
```

Você verá a coleção `predictions` com documentos contendo:
- `ticker`: "PETR4.SA"
- `predicted_at`: timestamp
- `prediction_date`: "2026-01-20"
- `predicted_price`: 38.50
- `current_price`: 38.20
- `actual_price`: null (será preenchido depois)
- `model_version`: "v1.0.0"

### 3. Executar Testes de Integração

```bash
# Localmente com emulador
docker-compose up -d firestore
export FIRESTORE_EMULATOR_HOST=localhost:8080
export GOOGLE_CLOUD_PROJECT=stock-prediction-local

pytest tests/integration/test_firestore_service.py -v
```

---

## 📊 Atualização de Preços Reais

Após alguns dias, você precisa atualizar os preços reais para calcular a acurácia do modelo.

### Executar Script Manualmente

```bash
# 1. Configurar credenciais (para produção)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
export GOOGLE_CLOUD_PROJECT="SEU_PROJECT_ID"

# 2. Atualizar preços reais
python scripts/update_actual_prices.py --ticker PETR4.SA

# 3. Simular sem salvar (dry-run)
python scripts/update_actual_prices.py --ticker PETR4.SA --dry-run

# 4. Limitar a 50 predições
python scripts/update_actual_prices.py --limit 50

# 5. Ver logs detalhados
python scripts/update_actual_prices.py --verbose
```

### Automação com Cloud Scheduler

Para executar automaticamente todos os dias:

```bash
# 1. Criar Cloud Function ou Cloud Run Job
# 2. Agendar com Cloud Scheduler

# Exemplo: Criar job que roda diariamente às 18h
gcloud scheduler jobs create http update-prices \
  --schedule="0 18 * * *" \
  --uri="https://YOUR_CLOUD_FUNCTION_URL/update-prices" \
  --http-method=POST \
  --time-zone="America/Sao_Paulo" \
  --location=us-central1
```

### Resultado Esperado

```bash
📊 Update Actual Prices - Firestore
===============================================
Processando predições pendentes para PETR4.SA...
Encontradas 3 predições pendentes

✅ Atualizado: PETR4.SA @ 2026-01-15 = R$ 38.45
✅ Atualizado: PETR4.SA @ 2026-01-16 = R$ 38.60
⏭️  Pulada: PETR4.SA @ 2026-01-20 (data futura)

📈 Resumo da Execução
===============================================
  ✅ Atualizadas com sucesso: 2
  ⏭️  Puladas (data futura):   1
  ❌ Falhas:                   0
===============================================
```

---

## 📈 Monitoramento

### Verificar Uso do Firestore

```bash
# Dashboard de uso
open "https://console.cloud.google.com/firestore/usage?project=SEU_PROJECT_ID"
```

### Limites do Free Tier

O Firestore oferece um generoso free tier:
- 📖 **50,000 leituras/dia**
- ✍️ **20,000 escritas/dia**
- 🗑️ **20,000 deleções/dia**
- 💾 **1 GB de armazenamento**

Para o caso de uso atual (predições diárias de PETR4.SA):
- ~1 escrita por predição
- ~10-30 leituras quando usuário acessa analytics
- **Estimativa:** Bem dentro do free tier

### Alertas de Custo (Opcional)

```bash
# Criar budget alert para ser notificado se ultrapassar $5/mês
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Firestore Budget Alert" \
  --budget-amount=5USD \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

---

## 🔍 Troubleshooting

### Problema: "Firestore service unavailable"

**Causa:** API não habilitada ou banco não criado

**Solução:**
```bash
# Verificar se API está habilitada
gcloud services list --enabled | grep firestore

# Se não estiver, habilitar
gcloud services enable firestore.googleapis.com

# Verificar banco de dados
gcloud firestore databases list
```

### Problema: "Permission denied"

**Causa:** Service account sem permissões

**Solução:**
```bash
SERVICE_ACCOUNT="SEU_PROJECT_ID@appspot.gserviceaccount.com"

gcloud projects add-iam-policy-binding SEU_PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/datastore.user"
```

### Problema: Emulador não conecta localmente

**Causa:** `FIRESTORE_EMULATOR_HOST` não configurado

**Solução:**
```bash
# Verificar se container está rodando
docker ps | grep firestore

# Verificar variável de ambiente
echo $FIRESTORE_EMULATOR_HOST  # Deve ser: localhost:8080

# Configurar manualmente
export FIRESTORE_EMULATOR_HOST=localhost:8080
export GOOGLE_CLOUD_PROJECT=stock-prediction-local
```

### Problema: Predições não aparecem no analytics

**Causa:** Predições não sendo salvas ou erro silencioso

**Solução:**
```bash
# 1. Verificar logs do Cloud Run
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=stock-api-backend" \
  --limit 50 \
  --format json

# 2. Buscar por erros de Firestore
gcloud logging read "resource.type=cloud_run_revision AND textPayload:firestore" \
  --limit 20

# 3. Verificar se GOOGLE_CLOUD_PROJECT está configurado
gcloud run services describe stock-api-backend \
  --region us-central1 \
  --format="value(spec.template.spec.containers[0].env)"
```

### Problema: Testes de integração falham

**Causa:** Emulador não está rodando

**Solução:**
```bash
# Subir emulador
docker-compose up -d firestore

# Aguardar 10s para emulador inicializar
sleep 10

# Verificar saúde
curl http://localhost:8080

# Rodar testes
FIRESTORE_EMULATOR_HOST=localhost:8080 \
GOOGLE_CLOUD_PROJECT=stock-prediction-test \
pytest tests/integration/test_firestore_service.py -v
```

---

## 📚 Recursos Adicionais

### Documentação Oficial

- [Firestore Quickstart](https://cloud.google.com/firestore/docs/quickstart-servers)
- [Firestore Data Model](https://cloud.google.com/firestore/docs/data-model)
- [Firestore Security Rules](https://cloud.google.com/firestore/docs/security/get-started)
- [Firestore Emulator](https://cloud.google.com/firestore/docs/emulator)
- [Firestore Pricing](https://cloud.google.com/firestore/pricing)

### Comandos Úteis

```bash
# Ver documentos de uma coleção
gcloud firestore collections list

# Fazer backup
gcloud firestore export gs://SEU_BUCKET/firestore-backup

# Importar backup
gcloud firestore import gs://SEU_BUCKET/firestore-backup

# Deletar coleção (cuidado!)
gcloud firestore indexes composite delete predictions

# Ver índices
gcloud firestore indexes list
```

### Scripts Disponíveis

| Script | Descrição |
|--------|-----------|
| `scripts/setup_firestore.sh` | Setup inicial do Firestore no GCP |
| `scripts/update_actual_prices.py` | Atualiza preços reais das predições |

---

## 🎯 Próximos Passos

Agora que o Firestore está configurado:

1. ✅ **Fazer deploy** do backend atualizado
2. ✅ **Testar endpoints** `/analytics/<ticker>`
3. ✅ **Configurar atualização** de preços reais (manual ou agendada)
4. ✅ **Monitorar uso** e custos no console GCP
5. 🔜 **Integrar frontend** para exibir histórico e métricas
6. 🔜 **Adicionar alertas** se acurácia cair abaixo de threshold
7. 🔜 **Implementar dashboard** com métricas de performance do modelo

---

## ❓ Perguntas Frequentes

**Q: Preciso criar índices manualmente?**  
A: Não. O Firestore cria índices automaticamente para queries simples. Se você fizer queries complexas, o erro indicará qual índice criar.

**Q: Como migrar dados de desenvolvimento para produção?**  
A: Use `gcloud firestore export/import` ou implemente um script de migração customizado.

**Q: Posso usar Firestore de múltiplas regiões?**  
A: Sim, mas cada projeto tem apenas um banco Firestore. Para multi-região, considere usar múltiplos projetos.

**Q: Como deletar dados antigos?**  
A: Implemente uma Cloud Function com TTL ou use a API para deletar documentos com `predicted_at` antigo.

**Q: Firestore é melhor que Cloud SQL?**  
A: Para este caso de uso (predições com queries simples), sim. É serverless, escala automaticamente e tem free tier generoso.

---

## 📞 Suporte

Se encontrar problemas:

1. Verificar [Troubleshooting](#troubleshooting) acima
2. Consultar logs do Cloud Run: `gcloud logging read`
3. Verificar [Status do GCP](https://status.cloud.google.com/)
4. Abrir issue no repositório GitHub

---

**Última atualização:** 19 de janeiro de 2026  
**Versão:** 1.0.0
