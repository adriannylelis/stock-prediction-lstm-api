
import os

# Habilitar/desabilitar rate limiting (útil para testes)
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

# Storage backend: "memory" (desenvolvimento) ou "redis://..." (produção)
RATE_LIMIT_STORAGE = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")

# Estratégia: "fixed-window" (padrão) ou "moving-window" (mais preciso)
RATE_LIMIT_STRATEGY = os.getenv("RATE_LIMIT_STRATEGY", "fixed-window")

# Identificador de rate limit: IP do cliente
RATE_LIMIT_KEY_FUNC = "default"  # Usa IP address

# Headers HTTP de resposta
RATE_LIMIT_HEADERS_ENABLED = True

# ============================================================
# LIMITES POR ENDPOINT
# ============================================================

RATE_LIMITS = {
    # Health check: Alta frequência permitida (monitoramento)
    "health": "100 per minute",
    
    # Model info: Read-only, moderado
    "model_info": "30 per minute",
    
    # Predict: CUSTOSO (Yahoo Finance + LSTM + Firestore write)
    # Limite mais restritivo para evitar sobrecarga
    "predict": "10 per minute",
    
    # Analytics: Queries Firestore, moderado
    "analytics": "30 per minute",
    "analytics_pending": "30 per minute",
    "analytics_accuracy": "30 per minute",
}

# Limite global padrão (fallback para endpoints não especificados)
DEFAULT_RATE_LIMIT = "50 per minute"

# ============================================================
# MENSAGENS DE ERRO
# ============================================================

RATE_LIMIT_MESSAGE = "Limite de requisições excedido. Tente novamente em {retry_after} segundos."

# ============================================================
# CONFIGURAÇÃO DE REDIS (OPCIONAL)
# ============================================================

# URL do Redis para produção
# Exemplo: "redis://localhost:6379/0" ou "redis://redis:6379/0" (Docker)
REDIS_URL = os.getenv("RATE_LIMIT_REDIS_URL", None)

if REDIS_URL:
    RATE_LIMIT_STORAGE = REDIS_URL


def get_rate_limit_config():
    """
    Retorna configuração completa para inicializar Flask-Limiter.
    
    Returns:
        dict: Dicionário com configurações para Flask-Limiter
    """
    return {
        "enabled": RATE_LIMIT_ENABLED,
        "storage_uri": RATE_LIMIT_STORAGE,
        "strategy": RATE_LIMIT_STRATEGY,
        "key_func": RATE_LIMIT_KEY_FUNC,
        "headers_enabled": RATE_LIMIT_HEADERS_ENABLED,
        "default_limits": [DEFAULT_RATE_LIMIT],
    }


def get_endpoint_limit(endpoint_name):
    """
    Retorna o limite configurado para um endpoint específico.
    
    Args:
        endpoint_name (str): Nome do endpoint (ex: "predict", "analytics")
    
    Returns:
        str: String de limite (ex: "10 per minute")
    """
    return RATE_LIMITS.get(endpoint_name, DEFAULT_RATE_LIMIT)
