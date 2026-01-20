#!/bin/bash

###############################################################################
# Script de Setup do Google Cloud Firestore
# 
# Este script configura o Firestore no Google Cloud Platform:
# - Habilita a API do Firestore
# - Cria um banco de dados Firestore em modo nativo
# - Configura permissões IAM necessárias
#
# Uso:
#   ./scripts/setup_firestore.sh [PROJECT_ID] [REGION]
#
# Exemplo:
#   ./scripts/setup_firestore.sh stock-prediction-prod us-central1
#
# Requisitos:
#   - gcloud CLI instalado e autenticado
#   - Permissões de Owner ou Editor no projeto GCP
###############################################################################

set -e  # Exit on error

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funções de logging
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Validar argumentos
PROJECT_ID="${1:-}"
REGION="${2:-us-central1}"

if [ -z "$PROJECT_ID" ]; then
    log_error "PROJECT_ID não fornecido"
    echo "Uso: $0 <PROJECT_ID> [REGION]"
    echo "Exemplo: $0 stock-prediction-prod us-central1"
    exit 1
fi

# Banner
echo ""
echo "=========================================="
echo "  🔥 Firestore Setup Script"
echo "=========================================="
echo "  Project: $PROJECT_ID"
echo "  Region:  $REGION"
echo "=========================================="
echo ""

# Verificar se gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI não está instalado"
    echo "Instale em: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Verificar autenticação
log_info "Verificando autenticação do gcloud..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
    log_error "Nenhuma conta autenticada encontrada"
    echo "Execute: gcloud auth login"
    exit 1
fi

ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1)
log_success "Autenticado como: $ACTIVE_ACCOUNT"

# Configurar projeto
log_info "Configurando projeto GCP..."
gcloud config set project "$PROJECT_ID" --quiet
log_success "Projeto configurado: $PROJECT_ID"

# Habilitar APIs necessárias
log_info "Habilitando APIs do Google Cloud..."

APIS=(
    "firestore.googleapis.com"
    "appengine.googleapis.com"  # Necessário para criar instância Firestore
)

for api in "${APIS[@]}"; do
    log_info "Habilitando $api..."
    if gcloud services enable "$api" --project="$PROJECT_ID" 2>&1 | grep -q "already enabled"; then
        log_warning "$api já habilitada"
    else
        log_success "$api habilitada com sucesso"
    fi
done

# Verificar se já existe um banco Firestore
log_info "Verificando banco de dados Firestore existente..."
if gcloud firestore databases list --project="$PROJECT_ID" --format="value(name)" 2>/dev/null | grep -q "(default)"; then
    log_warning "Banco de dados Firestore (default) já existe"
    EXISTING_LOCATION=$(gcloud firestore databases describe --database="(default)" --project="$PROJECT_ID" --format="value(locationId)" 2>/dev/null || echo "unknown")
    log_info "Localização atual: $EXISTING_LOCATION"
    
    if [ "$EXISTING_LOCATION" != "$REGION" ]; then
        log_warning "⚠️  O banco existente está em $EXISTING_LOCATION, mas você especificou $REGION"
        log_warning "⚠️  Não é possível mudar a região de um banco Firestore existente"
    fi
else
    # Criar banco de dados Firestore
    log_info "Criando banco de dados Firestore em modo nativo..."
    log_info "Região: $REGION"
    
    if gcloud firestore databases create \
        --location="$REGION" \
        --type=firestore-native \
        --project="$PROJECT_ID" 2>&1; then
        log_success "Banco de dados Firestore criado com sucesso!"
    else
        log_error "Falha ao criar banco de dados Firestore"
        log_info "Isso pode acontecer se o projeto já tiver uma instância do App Engine"
        log_info "Nesse caso, o Firestore já está disponível na região do App Engine"
    fi
fi

# Configurar permissões IAM
log_info "Configurando permissões IAM..."

# Service Account do Cloud Run (se existir)
SERVICE_ACCOUNT="${PROJECT_ID}@appspot.gserviceaccount.com"

log_info "Verificando service account: $SERVICE_ACCOUNT"

# Adicionar role de Firestore User
log_info "Adicionando role roles/datastore.user..."
if gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/datastore.user" \
    --condition=None \
    --quiet 2>&1 | grep -q "Updated IAM policy"; then
    log_success "Permissões adicionadas com sucesso"
else
    log_warning "Permissões podem já existir ou service account não encontrado"
fi

# Criar índices (se necessário)
log_info "Verificando necessidade de índices compostos..."
log_info "Para predições ordenadas por data, o Firestore criará índices automaticamente"

# Resumo final
echo ""
echo "=========================================="
echo "  ✅ Setup Concluído!"
echo "=========================================="
echo ""
log_success "Firestore configurado com sucesso em: $PROJECT_ID"
log_info "Região: $REGION"
log_info "Banco: (default)"
log_info "Modo: Native"
echo ""
log_info "Próximos passos:"
echo "  1. Configure a variável GOOGLE_CLOUD_PROJECT=$PROJECT_ID"
echo "  2. Deploy sua aplicação para Cloud Run"
echo "  3. Teste os endpoints /predict e /analytics/<ticker>"
echo ""
log_info "Para testar localmente com emulador:"
echo "  docker-compose up firestore backend"
echo ""
log_info "Documentação: https://cloud.google.com/firestore/docs"
echo ""
