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

# Instalar PyTorch CPU-only (muito menor que CUDA)
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

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
        LATEST_RELEASE=$(curl -s https://api.github.com/repos/${GITHUB_REPO}/releases/latest); \
        DOWNLOAD_URL=$(echo $LATEST_RELEASE | jq -r '.assets[0].browser_download_url'); \
        RELEASE_TAG=$(echo $LATEST_RELEASE | jq -r '.tag_name'); \
        echo "📌 Latest release: $RELEASE_TAG"; \
        echo "📥 Download URL: $DOWNLOAD_URL"; \
        curl -L -o artifacts.zip "$DOWNLOAD_URL" && \
        mkdir -p artifacts && \
        unzip artifacts.zip -d artifacts/ && \
        rm artifacts.zip && \
        echo "✅ Artifacts downloaded successfully"; \
        ls -lah artifacts/; \
    else \
        echo "📦 Using local artifacts (dev mode)"; \
        mkdir -p artifacts; \
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
