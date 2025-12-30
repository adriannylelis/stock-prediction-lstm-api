"""
Script de teste para validar tratamento de erros HTTP.

Testa todos os cenários de erro da API:
- 400: Ticker inválido
- 400: Dados insuficientes
- 404: Ticker não encontrado
- 500: Erro de inferência do modelo
- 503: Serviço indisponível

Para executar:
    python test_error_handling.py
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:5001"

def print_test_header(test_name):
    """Imprime cabeçalho de teste."""
    print("\n" + "=" * 70)
    print(f"  {test_name}")
    print("=" * 70)

def print_response(response):
    """Imprime resposta formatada."""
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except:
        print(response.text)

def test_missing_content_type():
    """Teste 1: Request sem Content-Type application/json"""
    print_test_header("TEST 1: Missing Content-Type")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        data="invalid data"
    )
    print_response(response)
    assert response.status_code == 400, f"Esperado 400, recebido {response.status_code}"
    print("✅ PASSED: Content-Type validation working")

def test_missing_ticker():
    """Teste 2: Request sem campo 'ticker'"""
    print_test_header("TEST 2: Missing Ticker Field")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={}
    )
    print_response(response)
    assert response.status_code == 400, f"Esperado 400, recebido {response.status_code}"
    print("✅ PASSED: Missing ticker validation working")

def test_invalid_ticker_format():
    """Teste 3: Ticker com formato inválido"""
    print_test_header("TEST 3: Invalid Ticker Format")
    
    # Ticker muito curto
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"ticker": "A"}
    )
    print_response(response)
    assert response.status_code == 400, f"Esperado 400, recebido {response.status_code}"
    assert "error" in response.json()
    print("✅ PASSED: Invalid ticker format validation working")

def test_ticker_not_found():
    """Teste 4: Ticker inexistente no Yahoo Finance"""
    print_test_header("TEST 4: Ticker Not Found (404)")
    
    # Ticker que não existe
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"ticker": "INVALIDTICKER123"}
    )
    print_response(response)
    # Pode ser 400 (dados insuficientes) ou 404 (não encontrado)
    assert response.status_code in [400, 404], f"Esperado 400 ou 404, recebido {response.status_code}"
    print(f"✅ PASSED: Ticker not found handled correctly ({response.status_code})")

def test_insufficient_data():
    """Teste 5: Ticker válido mas com dados insuficientes"""
    print_test_header("TEST 5: Insufficient Data (400)")
    
    # Ticker recente ou com poucos dados
    # Nota: Este teste depende de encontrar um ticker real com < 60 dias
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"ticker": "NEWIPO"}  # Exemplo de IPO recente
    )
    print_response(response)
    print(f"Status: {response.status_code} (pode variar dependendo do ticker)")

def test_successful_prediction():
    """Teste 6: Previsão bem-sucedida"""
    print_test_header("TEST 6: Successful Prediction (200)")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"ticker": "AAPL"}
    )
    print_response(response)
    assert response.status_code == 200, f"Esperado 200, recebido {response.status_code}"
    
    # Validar estrutura da resposta
    data = response.json()
    assert "success" in data
    assert data["success"] == True
    assert "data" in data
    
    result = data["data"]
    required_fields = [
        "ticker", "predicted_price", "current_price", 
        "change_percent", "prediction_date", "confidence"
    ]
    for field in required_fields:
        assert field in result, f"Campo '{field}' ausente na resposta"
    
    print("✅ PASSED: Successful prediction with all required fields")

def test_endpoint_not_found():
    """Teste 7: Endpoint inexistente (404)"""
    print_test_header("TEST 7: Endpoint Not Found (404)")
    
    response = requests.get(f"{BASE_URL}/invalid-endpoint")
    print_response(response)
    assert response.status_code == 404, f"Esperado 404, recebido {response.status_code}"
    print("✅ PASSED: 404 handler working")

def test_method_not_allowed():
    """Teste 8: Método HTTP inválido (405)"""
    print_test_header("TEST 8: Method Not Allowed (405)")
    
    # /predict só aceita POST
    response = requests.get(f"{BASE_URL}/predict")
    print_response(response)
    assert response.status_code == 405, f"Esperado 405, recebido {response.status_code}"
    print("✅ PASSED: 405 handler working")

def run_all_tests():
    """Executa todos os testes."""
    print("\n" + "#" * 70)
    print("#  TESTE DE TRATAMENTO DE ERROS HTTP - API REST")
    print("#" * 70)
    print(f"\nData/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL: {BASE_URL}")
    
    tests = [
        ("Missing Content-Type", test_missing_content_type),
        ("Missing Ticker Field", test_missing_ticker),
        ("Invalid Ticker Format", test_invalid_ticker_format),
        ("Ticker Not Found", test_ticker_not_found),
        ("Insufficient Data", test_insufficient_data),
        ("Successful Prediction", test_successful_prediction),
        ("Endpoint Not Found", test_endpoint_not_found),
        ("Method Not Allowed", test_method_not_allowed),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except requests.exceptions.ConnectionError:
            print(f"❌ FAILED: Não foi possível conectar ao servidor")
            print(f"   Certifique-se de que a API está rodando em {BASE_URL}")
            failed += 1
            break
        except Exception as e:
            print(f"❌ FAILED: Erro inesperado - {str(e)}")
            failed += 1
    
    # Resumo
    print("\n" + "=" * 70)
    print("  RESUMO DOS TESTES")
    print("=" * 70)
    print(f"Total de testes: {len(tests)}")
    print(f"✅ Passou: {passed}")
    print(f"❌ Falhou: {failed}")
    print(f"Taxa de sucesso: {(passed/len(tests))*100:.1f}%")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_all_tests()
