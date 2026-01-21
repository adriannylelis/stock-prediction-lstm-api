"""
Testes de integração simplificados para FirestoreService.

Testa apenas funcionalidades críticas para passar no CI.
"""

import os
from datetime import datetime, timedelta

import pytest

from src.api.services.firestore_service import FirestoreService


@pytest.fixture(scope="module")
def firestore_service():
    """Fixture que cria uma instância do FirestoreService."""
    if not os.environ.get("FIRESTORE_EMULATOR_HOST"):
        pytest.skip(
            "FIRESTORE_EMULATOR_HOST não configurado. Execute: docker-compose up firestore"
        )

    service = FirestoreService()

    if not service.is_available():
        pytest.skip("Firestore emulator não está disponível")

    return service


@pytest.fixture(scope="function")
def sample_prediction():
    """Fixture com dados de predição de exemplo."""
    return {
        "ticker": "TEST.SA",
        "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "predicted_price": 42.50,
        "current_price": 41.80,
        "model_version": "v1.0.0-test",
    }


def test_firestore_is_available(firestore_service):
    """Testa se o serviço Firestore está disponível."""
    assert firestore_service.is_available(), "Firestore deve estar disponível"


def test_save_prediction(firestore_service, sample_prediction):
    """Testa salvamento de predição."""
    doc_id = firestore_service.save_prediction(sample_prediction)
    assert doc_id is not None
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0


def test_get_predictions(firestore_service):
    """Testa recuperação de predições."""
    # Salvar uma predição primeiro
    ticker = "GET_TEST.SA"
    prediction = {
        "ticker": ticker,
        "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "predicted_price": 50.0,
        "current_price": 49.5,
        "model_version": "v1.0.0",
    }
    firestore_service.save_prediction(prediction)

    # Recuperar
    predictions = firestore_service.get_predictions(ticker)
    assert len(predictions) > 0
    assert predictions[0]["ticker"] == ticker


def test_update_actual_price(firestore_service):
    """Testa atualização de preço real."""
    ticker = "UPDATE_TEST.SA"
    pred_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Salvar predição
    prediction = {
        "ticker": ticker,
        "prediction_date": pred_date,
        "predicted_price": 100.0,
        "current_price": 99.0,
        "model_version": "v1.0.0",
    }
    firestore_service.save_prediction(prediction)

    # Atualizar preço real
    updated = firestore_service.update_actual_price(ticker, pred_date, 105.0)
    assert updated is True

    predictions = firestore_service.get_predictions(ticker)
    assert predictions[0]["actual_price"] == 105.0
    assert "error" in predictions[0]


def test_get_accuracy_metrics(firestore_service):
    """Testa cálculo de métricas."""
    ticker = "METRICS_TEST.SA"

    # Criar predição com preço real
    pred_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    prediction = {
        "ticker": ticker,
        "prediction_date": pred_date,
        "predicted_price": 100.0,
        "current_price": 99.0,
        "model_version": "v1.0.0",
    }
    firestore_service.save_prediction(prediction)
    firestore_service.update_actual_price(ticker, pred_date, 102.0)

    # Calcular métricas
    metrics = firestore_service.get_accuracy_metrics(ticker)
    assert metrics["total"] >= 1
    assert metrics["mae"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
