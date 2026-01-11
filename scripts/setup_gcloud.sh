#!/bin/bash

###############################################################################
# Script: Setup Google Cloud Platform
# Description: Configuração inicial do GCP para deploy completo (Backend + Frontend)
# Usage: ./scripts/setup_gcloud.sh
###############################################################################

set -e  # Exit on error

echo "☁️  Google Cloud Platform - Setup Inicial"
echo "=========================================="
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==================== VERIFICAÇÕES ====================
echo "🔍 Verificando dependências..."

# Verificar se gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI não encontrado!${NC}"
    echo ""
    echo "Instale o Google Cloud SDK:"
    echo "  macOS: brew install --cask google-cloud-sdk"
    echo "  Linux: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo -e "${GREEN}✅ gcloud CLI instalado${NC}"

# ==================== CONFIGURAÇÃO ====================
echo ""
echo "📝 Configuração do Projeto"
echo "─────────────────────────"

# Solicitar Project ID
read -p "Digite o Project ID do GCP (ou pressione Enter para criar novo): " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    # Criar novo projeto
    read -p "Digite o nome do novo projeto: " PROJECT_NAME
    PROJECT_ID=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
    
    echo ""
    echo "🆕 Criando novo projeto: $PROJECT_ID"
    gcloud projects create "$PROJECT_ID" --name="$PROJECT_NAME" || {
        echo -e "${RED}❌ Erro ao criar projeto. Tente um ID diferente.${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Projeto criado!${NC}"
fi

# Definir projeto ativo
echo ""
echo "🎯 Definindo projeto ativo: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

# ==================== HABILITAR APIs ====================
echo ""
echo "🔧 Habilitando APIs necessárias..."
echo "   (isso pode levar alguns minutos)"

APIs=(
    "run.googleapis.com"              # Cloud Run
    "cloudbuild.googleapis.com"       # Cloud Build
    "containerregistry.googleapis.com" # Container Registry
    "compute.googleapis.com"          # Compute Engine (para Cloud Run)
    "secretmanager.googleapis.com"    # Secret Manager
)

for api in "${APIs[@]}"; do
    echo "   Habilitando $api..."
    gcloud services enable "$api" --quiet
done

echo -e "${GREEN}✅ APIs habilitadas!${NC}"

# ==================== REGIÃO ====================
echo ""
echo "🌎 Configurando região"
echo "─────────────────────"
echo "Regiões recomendadas para Brasil:"
echo "  1. us-central1 (Iowa, USA) - Mais barata, latência OK"
echo "  2. southamerica-east1 (São Paulo) - Menor latência, mais cara"
echo "  3. us-east1 (South Carolina) - Balanceada"
echo ""
read -p "Escolha a região (padrão: us-central1): " REGION
REGION=${REGION:-us-central1}

echo "📍 Região selecionada: $REGION"

# ==================== SERVICE ACCOUNT ====================
echo ""
echo "🔐 Criando Service Account para GitHub Actions"
echo "───────────────────────────────────────────────"

SA_NAME="github-actions-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Verificar se já existe
if gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null; then
    echo -e "${YELLOW}⚠️  Service Account já existe. Pulando criação.${NC}"
else
    echo "Criando Service Account: $SA_NAME"
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="GitHub Actions Service Account" \
        --description="Service Account para CI/CD via GitHub Actions"
    
    echo -e "${GREEN}✅ Service Account criada!${NC}"
fi

# ==================== PERMISSÕES ====================
echo ""
echo "🔑 Atribuindo permissões..."

ROLES=(
    "roles/run.admin"                 # Deploy Cloud Run
    "roles/storage.admin"             # Cloud Storage (GCR)
    "roles/cloudbuild.builds.editor"  # Cloud Build
    "roles/iam.serviceAccountUser"    # Usar Service Account
    "roles/viewer"                    # Visualizar recursos
)

for role in "${ROLES[@]}"; do
    echo "   Atribuindo $role..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$role" \
        --quiet &>/dev/null
done

echo -e "${GREEN}✅ Permissões atribuídas!${NC}"

# ==================== KEY JSON ====================
echo ""
echo "🔑 Gerando chave JSON..."

KEY_FILE="gcp-key-${PROJECT_ID}.json"

if [ -f "$KEY_FILE" ]; then
    echo -e "${YELLOW}⚠️  Chave já existe: $KEY_FILE${NC}"
    read -p "Deseja sobrescrever? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Mantendo chave existente."
    else
        rm "$KEY_FILE"
        gcloud iam service-accounts keys create "$KEY_FILE" \
            --iam-account="$SA_EMAIL"
        echo -e "${GREEN}✅ Nova chave gerada!${NC}"
    fi
else
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SA_EMAIL"
    echo -e "${GREEN}✅ Chave gerada: $KEY_FILE${NC}"
fi

# ==================== BILLING WARNING ====================
echo ""
echo -e "${YELLOW}⚠️  ATENÇÃO: BILLING${NC}"
echo "─────────────────────────"
echo "Para usar Cloud Run, o projeto precisa ter Billing habilitado."
echo ""
read -p "Deseja abrir o console de billing? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🌐 Abrindo console..."
    open "https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
fi

# ==================== RESUMO ====================
echo ""
echo "═══════════════════════════════════════════════"
echo "✅ SETUP CONCLUÍDO COM SUCESSO!"
echo "═══════════════════════════════════════════════"
echo ""
echo "📊 Informações do Projeto:"
echo "   Project ID: $PROJECT_ID"
echo "   Região: $REGION"
echo "   Service Account: $SA_EMAIL"
echo "   Key File: $KEY_FILE"
echo ""
echo "🔐 Configurar GitHub Secrets:"
echo "   1. Vá em: https://github.com/YOUR_USERNAME/stock-prediction-lstm-api/settings/secrets/actions"
echo "   2. Adicione os seguintes secrets:"
echo ""
echo "   GCP_PROJECT_ID"
echo "   ─────────────────"
echo "   $PROJECT_ID"
echo ""
echo "   GCP_SA_KEY"
echo "   ─────────────────"
echo "   (Cole o conteúdo do arquivo $KEY_FILE)"
echo ""
echo "   Para copiar o conteúdo:"
echo "   cat $KEY_FILE | pbcopy  # macOS"
echo "   cat $KEY_FILE | xclip -selection clipboard  # Linux"
echo ""
echo "🚀 Próximos Passos:"
echo "   1. Configure os GitHub Secrets (acima)"
echo "   2. Execute o workflow: 🤖 Train Model Weekly"
echo "   3. Execute o workflow: 🌐 Deploy to Google Cloud"
echo "   4. Acesse sua aplicação!"
echo ""
echo "📚 Documentação:"
echo "   docs/GCLOUD_DEPLOY.md"
echo ""
echo "💰 Custos Estimados (uso moderado):"
echo "   Cloud Run Backend: ~\$3-5/mês"
echo "   Cloud Run Frontend: ~\$1-2/mês"
echo "   Cloud Build: Grátis (120 min build/dia)"
echo "   Total: ~\$4-7/mês"
echo ""
echo "═══════════════════════════════════════════════"

# ==================== CLEANUP ====================
echo ""
echo "🔒 IMPORTANTE: Guarde o arquivo $KEY_FILE em local seguro"
echo "⚠️  NÃO commite este arquivo no Git!"
echo ""
read -p "Pressione Enter para finalizar..."
