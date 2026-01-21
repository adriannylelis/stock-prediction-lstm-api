import os

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_STORAGE = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")
RATE_LIMIT_STRATEGY = os.getenv("RATE_LIMIT_STRATEGY", "fixed-window")
RATE_LIMIT_KEY_FUNC = "default"
RATE_LIMIT_HEADERS_ENABLED = True

RATE_LIMITS = {
    "health": "100 per minute",
    "model_info": "30 per minute",
    "predict": "10 per minute",
    "analytics": "30 per minute",
    "analytics_pending": "30 per minute",
    "analytics_accuracy": "30 per minute",
}

DEFAULT_RATE_LIMIT = "50 per minute"
RATE_LIMIT_MESSAGE = "Limite de requisições excedido. Tente novamente em {retry_after} segundos."

REDIS_URL = os.getenv("RATE_LIMIT_REDIS_URL", None)
if REDIS_URL:
    RATE_LIMIT_STORAGE = REDIS_URL


def get_rate_limit_config():
    return {
        "enabled": RATE_LIMIT_ENABLED,
        "storage_uri": RATE_LIMIT_STORAGE,
        "strategy": RATE_LIMIT_STRATEGY,
        "key_func": RATE_LIMIT_KEY_FUNC,
        "headers_enabled": RATE_LIMIT_HEADERS_ENABLED,
        "default_limits": [DEFAULT_RATE_LIMIT],
    }


def get_endpoint_limit(endpoint_name):
    return RATE_LIMITS.get(endpoint_name, DEFAULT_RATE_LIMIT)
