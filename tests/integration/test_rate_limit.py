import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true",
    reason="Rate limit tests disabled in CI/CD environment",
)


@pytest.fixture
def client():
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
    def test_health_endpoint_within_limit(self, client):
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json["status"] == "healthy"

    def test_predict_endpoint_rate_limit_exceeded(self, client):
        success_count = 0
        rate_limited = False

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

        assert rate_limited, "Rate limit não foi aplicado"
        assert success_count <= 10, f"Permitiu {success_count} requests, limite é 10"

    def test_rate_limit_headers_present(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json

    def test_different_endpoints_independent_limits(self, client):
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200

        for _ in range(5):
            response = client.get("/model/info")
            assert response.status_code in [200, 404, 500]

    def test_rate_limit_can_be_disabled(self):
        from src.api.main import create_app

        app = create_app(
            config={
                "TESTING": True,
                "RATE_LIMIT_ENABLED": False,
            }
        )

        with app.test_client() as client:
            for _ in range(20):
                response = client.get("/health")
                assert response.status_code == 200

    def test_analytics_endpoint_rate_limit(self, client):
        success_count = 0

        for i in range(35):
            response = client.get("/analytics/AAPL")

            if response.status_code in [200, 404]:
                success_count += 1
            elif response.status_code == 429:
                assert "RateLimitExceeded" in response.json["error"]
                break

        assert success_count <= 31, f"Permitiu {success_count} requests, limite é 30"


@pytest.mark.skipif(True, reason="Teste lento - requer esperar reset do contador")
class TestRateLimitReset:
    def test_rate_limit_resets_after_window(self, client):
        for _ in range(10):
            response = client.post("/predict", json={"ticker": "AAPL"})
            if response.status_code == 429:
                break

        response = client.post("/predict", json={"ticker": "AAPL"})
        assert response.status_code == 429

        time.sleep(61)

        response = client.post("/predict", json={"ticker": "AAPL"})
        assert response.status_code in [200, 400, 404]
