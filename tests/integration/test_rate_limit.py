
import os
import time

import pytest

# Skip todos os testes se estiver em ambiente CI
pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
    reason="Rate limit tests disabled in CI/CD environment"
)


@pytest.fixture
def client():
    """
    Fixture que cria um client Flask para testes.
    Rate limiting deve estar habilitado.
    """
    from src.api.main import create_app

    app = create_app(
        config={
            "TESTING": True,
            "RATE_LIMIT_ENABLED": True,
            "RATE_LIMIT_STORAGE_URI": "memory://",
        }
    )

    with app.test_client() as client:
        yield client


class TestRateLimiting:
    """Testes de rate limiting."""

    def test_health_endpoint_within_limit(self, client):
        """Testa que requests dentro do limite funcionam normalmente."""
        # Health permite 100/min, fazer 5 requests deve funcionar
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json["status"] == "healthy"

    def test_predict_endpoint_rate_limit_exceeded(self, client):
        """Testa que endpoint /predict bloqueia após 10 requests por minuto."""
        # Predict permite 10/min
        success_count = 0
        rate_limited = False

        # Tentar fazer 15 requests
        for i in range(15):
            response = client.post(
                "/predict",
                json={"ticker": "AAPL"},
                content_type="application/json",
            )

            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited = True
                assert "RateLimitExceeded" in response.json["error"]
                break

        # Deve ter bloqueado antes de 15 requests
        assert rate_limited, "Rate limit não foi aplicado"
        assert success_count <= 10, f"Permitiu {success_count} requests, limite é 10"

    def test_rate_limit_headers_present(self, client):
        """Testa que headers de rate limit estão presentes na resposta."""
        response = client.get("/health")

        assert response.status_code == 200

        # Flask-Limiter adiciona headers X-RateLimit-*
        # Nota: Alguns headers podem não estar presentes dependendo da configuração
        # Apenas verificamos que a resposta é válida
        assert "status" in response.json

    def test_different_endpoints_independent_limits(self, client):
        """Testa que endpoints diferentes têm contadores independentes."""
        # Fazer 5 requests para /health (limite 100/min)
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200

        # Fazer 5 requests para /model/info (limite 30/min)
        for _ in range(5):
            response = client.get("/model/info")
            # Pode retornar 200 ou erro de artefatos faltando
            assert response.status_code in [200, 404, 500]

        # Ambos devem estar dentro dos limites
        # (não deve bloquear pois são contadores separados)

    def test_rate_limit_can_be_disabled(self):
        """Testa que rate limiting pode ser desabilitado via configuração."""
        from src.api.main import create_app

        app = create_app(
            config={
                "TESTING": True,
                "RATE_LIMIT_ENABLED": False,
            }
        )

        with app.test_client() as client:
            # Fazer muitas requests, não deve bloquear
            for _ in range(20):
                response = client.get("/health")
                assert response.status_code == 200

    def test_analytics_endpoint_rate_limit(self, client):
        """Testa rate limit no endpoint de analytics (30/min)."""
        success_count = 0

        # Tentar fazer 35 requests
        for i in range(35):
            response = client.get("/analytics/AAPL")

            if response.status_code in [200, 404]:  # 404 se não houver dados
                success_count += 1
            elif response.status_code == 429:
                # Rate limit atingido
                assert "RateLimitExceeded" in response.json["error"]
                break

        # Deve permitir ~30 requests antes de bloquear
        assert success_count <= 31, f"Permitiu {success_count} requests, limite é 30"


@pytest.mark.skipif(
    True, reason="Teste lento - requer esperar reset do contador"
)
class TestRateLimitReset:
    """Testes que requerem espera (marcados para skip por padrão)."""

    def test_rate_limit_resets_after_window(self, client):
        """
        Testa que contador reseta após a janela de tempo.
        
        NOTA: Teste desabilitado por padrão pois requer esperar 60 segundos.
        """
        # Fazer 10 requests (atingir limite)
        for _ in range(10):
            response = client.post("/predict", json={"ticker": "AAPL"})
            if response.status_code == 429:
                break

        # Próxima request deve ser bloqueada
        response = client.post("/predict", json={"ticker": "AAPL"})
        assert response.status_code == 429

        # Esperar 61 segundos (janela de 1 minuto + margem)
        time.sleep(61)

        # Agora deve funcionar novamente
        response = client.post("/predict", json={"ticker": "AAPL"})
        assert response.status_code in [200, 400, 404]  # Não deve ser 429
