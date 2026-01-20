"""
Testes de integração para FirestoreService.

Executa testes contra o emulador Firestore para validar:
- Salvamento de predições
- Recuperação de histórico
- Atualização de preços reais
- Cálculo de métricas de acurácia

Requisitos:
    - Firestore emulator rodando (FIRESTORE_EMULATOR_HOST configurado)
    - pytest instalado

Uso:
    pytest tests/integration/test_firestore_service.py -v
"""

import os
import pytest
from datetime import datetime, timedelta
from src.api.services.firestore_service import FirestoreService


@pytest.fixture(scope='module')
def firestore_service():
    """
    Fixture que cria uma instância do FirestoreService.
    
    O emulador deve estar rodando antes dos testes.
    """
    # Verificar se emulador está configurado
    if not os.environ.get('FIRESTORE_EMULATOR_HOST'):
        pytest.skip("FIRESTORE_EMULATOR_HOST não configurado. Execute: docker-compose up firestore")
    
    service = FirestoreService()
    
    if not service.is_available():
        pytest.skip("Firestore emulator não está disponível")
    
    return service


@pytest.fixture(scope='function')
def sample_prediction():
    """Fixture com dados de predição de exemplo."""
    return {
        'ticker': 'TEST.SA',
        'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'predicted_price': 42.50,
        'current_price': 41.80,
        'model_version': 'v1.0.0-test'
    }


class TestFirestoreServiceConnection:
    """Testes de conectividade e disponibilidade."""
    
    def test_is_available(self, firestore_service):
        """Testa se o serviço Firestore está disponível."""
        assert firestore_service.is_available(), "Firestore deve estar disponível"
    
    def test_emulator_detection(self, firestore_service):
        """Verifica se o emulador foi detectado corretamente."""
        assert os.environ.get('FIRESTORE_EMULATOR_HOST'), "FIRESTORE_EMULATOR_HOST deve estar configurado"


class TestFirestoreSavePrediction:
    """Testes de salvamento de predições."""
    
    def test_save_prediction_success(self, firestore_service, sample_prediction):
        """Testa salvamento de predição com sucesso."""
        # Salvar predição
        doc_id = firestore_service.save_prediction(sample_prediction)
        
        assert doc_id is not None, "Deve retornar um doc_id"
        assert isinstance(doc_id, str), "doc_id deve ser string"
        assert len(doc_id) > 0, "doc_id não pode ser vazio"
    
    def test_save_prediction_with_all_fields(self, firestore_service):
        """Testa salvamento com todos os campos opcionais."""
        prediction = {
            'ticker': 'FULL.SA',
            'prediction_date': '2026-01-20',
            'predicted_price': 50.00,
            'current_price': 49.50,
            'actual_price': 50.25,
            'model_version': 'v2.0.0'
        }
        
        doc_id = firestore_service.save_prediction(prediction)
        assert doc_id is not None
    
    def test_save_prediction_missing_required_field(self, firestore_service):
        """Testa que falta de campo obrigatório gera erro."""
        incomplete_prediction = {
            'ticker': 'BAD.SA',
            'predicted_price': 30.00
            # Faltando prediction_date
        }
        
        with pytest.raises(Exception):
            firestore_service.save_prediction(incomplete_prediction)


class TestFirestoreGetPredictions:
    """Testes de recuperação de predições."""
    
    def test_get_predictions_empty(self, firestore_service):
        """Testa busca de predições para ticker sem histórico."""
        predictions = firestore_service.get_predictions('NONEXISTENT.SA')
        assert predictions == [], "Deve retornar lista vazia"
    
    def test_get_predictions_after_save(self, firestore_service, sample_prediction):
        """Testa recuperação após salvamento."""
        # Salvar predição
        ticker = 'RETRIEVE.SA'
        sample_prediction['ticker'] = ticker
        firestore_service.save_prediction(sample_prediction)
        
        # Recuperar
        predictions = firestore_service.get_predictions(ticker)
        
        assert len(predictions) > 0, "Deve retornar pelo menos uma predição"
        assert predictions[0]['ticker'] == ticker
        assert 'predicted_price' in predictions[0]
        assert 'prediction_date' in predictions[0]
    
    def test_get_predictions_with_limit(self, firestore_service):
        """Testa limite de resultados."""
        ticker = 'LIMIT.SA'
        
        # Salvar múltiplas predições
        for i in range(5):
            pred = {
                'ticker': ticker,
                'prediction_date': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                'predicted_price': 40.0 + i,
                'current_price': 39.5 + i,
                'model_version': 'v1.0.0'
            }
            firestore_service.save_prediction(pred)
        
        # Buscar com limite
        predictions = firestore_service.get_predictions(ticker, limit=3)
        
        assert len(predictions) <= 3, "Deve respeitar o limite"
    
    def test_get_predictions_ordered_by_date(self, firestore_service):
        """Testa ordenação por data (mais recente primeiro)."""
        ticker = 'ORDER.SA'
        
        # Salvar predições em ordem aleatória
        dates = [
            (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
            (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
        ]
        
        for date in dates:
            pred = {
                'ticker': ticker,
                'prediction_date': date,
                'predicted_price': 45.0,
                'current_price': 44.5,
                'model_version': 'v1.0.0'
            }
            firestore_service.save_prediction(pred)
        
        # Buscar e verificar ordem
        predictions = firestore_service.get_predictions(ticker)
        
        assert len(predictions) >= 3
        # Verificar que está ordenado DESC (mais recente primeiro)
        for i in range(len(predictions) - 1):
            date1 = predictions[i].get('predicted_at')
            date2 = predictions[i+1].get('predicted_at')
            if date1 and date2:
                assert date1 >= date2, "Deve estar ordenado por data DESC"


class TestFirestoreUpdateActualPrice:
    """Testes de atualização de preço real."""
    
    def test_update_actual_price_success(self, firestore_service):
        """Testa atualização de preço real com sucesso."""
        ticker = 'UPDATE.SA'
        pred_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Salvar predição
        prediction = {
            'ticker': ticker,
            'prediction_date': pred_date,
            'predicted_price': 50.00,
            'current_price': 49.50,
            'model_version': 'v1.0.0'
        }
        firestore_service.save_prediction(prediction)
        
        # Atualizar preço real
        actual_price = 51.25
        updated = firestore_service.update_actual_price(ticker, pred_date, actual_price)
        
        assert updated is True, "Atualização deve ter sucesso"
        
        # Verificar atualização
        predictions = firestore_service.get_predictions(ticker)
        assert predictions[0]['actual_price'] == actual_price
        assert 'error' in predictions[0]
        assert 'error_percent' in predictions[0]
    
    def test_update_actual_price_calculates_error(self, firestore_service):
        """Testa que o erro é calculado corretamente."""
        ticker = 'ERROR.SA'
        pred_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        predicted = 100.00
        actual = 105.00
        
        # Salvar predição
        prediction = {
            'ticker': ticker,
            'prediction_date': pred_date,
            'predicted_price': predicted,
            'current_price': 99.00,
            'model_version': 'v1.0.0'
        }
        firestore_service.save_prediction(prediction)
        
        # Atualizar
        firestore_service.update_actual_price(ticker, pred_date, actual)
        
        # Verificar cálculo
        predictions = firestore_service.get_predictions(ticker)
        error = predictions[0]['error']
        error_percent = predictions[0]['error_percent']
        
        assert error == 5.00, f"Erro deve ser 5.00, mas foi {error}"
        assert error_percent == 5.00, f"Erro % deve ser 5.00, mas foi {error_percent}"
    
    def test_update_nonexistent_prediction(self, firestore_service):
        """Testa atualização de predição inexistente."""
        updated = firestore_service.update_actual_price(
            'NOEXIST.SA',
            '2025-01-01',
            100.0
        )
        assert updated is False, "Não deve atualizar predição inexistente"


class TestFirestoreGetPendingPredictions:
    """Testes de predições pendentes."""
    
    def test_get_pending_predictions(self, firestore_service):
        """Testa busca de predições sem preço real."""
        ticker = 'PENDING.SA'
        
        # Salvar predição sem actual_price
        prediction = {
            'ticker': ticker,
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'predicted_price': 60.00,
            'current_price': 59.50,
            'model_version': 'v1.0.0'
        }
        firestore_service.save_prediction(prediction)
        
        # Buscar pendentes
        pending = firestore_service.get_pending_predictions(ticker)
        
        assert len(pending) > 0, "Deve ter predições pendentes"
        assert all(p.get('actual_price') is None for p in pending)


class TestFirestoreGetAccuracyMetrics:
    """Testes de métricas de acurácia."""
    
    def test_get_accuracy_metrics_empty(self, firestore_service):
        """Testa métricas quando não há dados."""
        metrics = firestore_service.get_accuracy_metrics('NODATA.SA')
        
        assert metrics['total'] == 0
        assert metrics['mae'] is None
        assert metrics['mape'] is None
        assert metrics['rmse'] is None
    
    def test_get_accuracy_metrics_with_data(self, firestore_service):
        """Testa cálculo de métricas com dados reais."""
        ticker = 'METRICS.SA'
        
        # Criar predições com erros conhecidos
        predictions = [
            {'predicted': 100.0, 'actual': 102.0},  # erro: 2.0
            {'predicted': 100.0, 'actual': 98.0},   # erro: 2.0
            {'predicted': 100.0, 'actual': 101.0},  # erro: 1.0
        ]
        
        for i, pred in enumerate(predictions):
            prediction = {
                'ticker': ticker,
                'prediction_date': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                'predicted_price': pred['predicted'],
                'current_price': pred['predicted'] - 1,
                'model_version': 'v1.0.0'
            }
            firestore_service.save_prediction(prediction)
            firestore_service.update_actual_price(
                ticker,
                prediction['prediction_date'],
                pred['actual']
            )
        
        # Calcular métricas
        metrics = firestore_service.get_accuracy_metrics(ticker)
        
        assert metrics['total'] == 3
        assert metrics['mae'] is not None
        assert metrics['mae'] > 0
        assert metrics['mape'] is not None
        assert metrics['rmse'] is not None


class TestFirestoreDataIntegrity:
    """Testes de integridade de dados."""
    
    def test_prediction_has_timestamp(self, firestore_service, sample_prediction):
        """Verifica que predição tem timestamp de criação."""
        firestore_service.save_prediction(sample_prediction)
        
        predictions = firestore_service.get_predictions(sample_prediction['ticker'])
        assert 'predicted_at' in predictions[0]
    
    def test_prediction_preserves_all_fields(self, firestore_service):
        """Verifica que todos os campos são preservados."""
        prediction = {
            'ticker': 'PRESERVE.SA',
            'prediction_date': '2026-01-25',
            'predicted_price': 75.50,
            'current_price': 74.80,
            'model_version': 'v1.2.3'
        }
        
        firestore_service.save_prediction(prediction)
        retrieved = firestore_service.get_predictions('PRESERVE.SA')[0]
        
        assert retrieved['ticker'] == prediction['ticker']
        assert retrieved['prediction_date'] == prediction['prediction_date']
        assert retrieved['predicted_price'] == prediction['predicted_price']
        assert retrieved['current_price'] == prediction['current_price']
        assert retrieved['model_version'] == prediction['model_version']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
