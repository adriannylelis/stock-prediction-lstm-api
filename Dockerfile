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

# Instalar dependências do sistema (se necessário)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro (aproveitando cache do Docker)
COPY requirements.txt .

# Instalar PyTorch CPU-only (muito menor que CUDA)
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Instalar NumPy < 2.0 para compatibilidade com PyTorch 2.2.2
RUN pip install "numpy<2.0"

# Instalar outras dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Criar usuário não-root para segurança
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Mudar para usuário não-root
USER appuser

# Expor porta da API
EXPOSE 5001

# Healthcheck para monitoramento do container
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Comando de inicialização
CMD ["python", "-m", "src.api.main"]
