# Imagem base Python 3.11 slim (Debian-based, otimizada)
FROM python:3.11-slim

# Metadados
LABEL maintainer="adriannylelis"
LABEL description="Stock Prediction LSTM API - Flask REST API para previsão de preços de ações"
LABEL version="1.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema (curl para download + unzip para artifacts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro (aproveitando cache do Docker)
COPY requirements.txt .

# Instalar PyTorch CPU-only (muito menor que CUDA) com timeout aumentado
RUN pip install --timeout=300 --retries=5 torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Instalar NumPy < 2.0 para compatibilidade com PyTorch 2.2.2
RUN pip install "numpy<2.0"

# Instalar outras dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação, configs e CLI
COPY src/ ./src/
COPY cli/ ./cli/
COPY configs/ ./configs/

# ⭐ DOWNLOAD ARTIFACTS FROM GITHUB RELEASE
# Se existir artifacts/ local (dev), usa. Se não, baixa do GitHub Release (production)
ARG GITHUB_REPO="adriannylelis/stock-prediction-lstm-api"
ARG DOWNLOAD_ARTIFACTS="true"

RUN if [ "$DOWNLOAD_ARTIFACTS" = "true" ] && [ ! -f "artifacts/model.pt" ]; then \
        echo "📥 Downloading artifacts from GitHub Release..."; \
        REPO="${GITHUB_REPO:-adriannylelis/stock-prediction-lstm-api}"; \
        echo "📦 Repository: $REPO"; \
        LATEST_RELEASE=$(curl -s "https://api.github.com/repos/$REPO/releases/latest"); \
        \
        # Verificar se a resposta da API é válida \
        if [ -z "$LATEST_RELEASE" ] || echo "$LATEST_RELEASE" | grep -q "Not Found"; then \
            echo "❌ No releases found in repository."; \
            echo "⚠️  Please create a release first by running the training workflow."; \
            echo "📦 Creating empty artifacts directory for now..."; \
            mkdir -p artifacts/models/scalers; \
            exit 0; \
        fi; \
        \
        # Verificar se há assets no release \
        RELEASE_TAG=$(echo "$LATEST_RELEASE" | jq -r '.tag_name'); \
        ASSET_COUNT=$(echo "$LATEST_RELEASE" | jq '.assets | length'); \
        \
        if [ "$ASSET_COUNT" -eq 0 ] || [ "$ASSET_COUNT" = "null" ]; then \
            echo "❌ No assets found in latest release '$RELEASE_TAG'."; \
            echo "⚠️  Please ensure the training workflow completed successfully."; \
            echo "📦 Creating empty artifacts directory for now..."; \
            mkdir -p artifacts/models/scalers; \
            exit 0; \
        fi; \
        \
        # Baixar artifacts \
        DOWNLOAD_URL=$(echo "$LATEST_RELEASE" | jq -r '.assets[0].browser_download_url'); \
        echo "📌 Latest release: $RELEASE_TAG"; \
        echo "📥 Downloading from: $DOWNLOAD_URL"; \
        \
        curl -L -o artifacts.zip "$DOWNLOAD_URL" && \
        mkdir -p artifacts/models && \
        unzip -q artifacts.zip -d artifacts/models/ && \
        rm artifacts.zip && \
        echo "✅ Artifacts downloaded successfully"; \
        ls -lah artifacts/models/; \
    else \
        echo "📦 Using local artifacts (dev mode)"; \
        mkdir -p artifacts/models/scalers; \
    fi

# Criar usuário não-root para segurança e garantir permissões de escrita
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/artifacts /app/configs

# Mudar para usuário não-root
USER appuser

# Expor porta da API
EXPOSE 5001

# Healthcheck para monitoramento do container
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Comando de inicialização (Flask production server)
CMD ["python", "-m", "flask", "--app", "src.api.main:create_app()", "run", "--host=0.0.0.0", "--port=5001"]
