"""
Testes automatizados para a API REST.

Executa testes de integração em todos os endpoints da API,
validando responses, status codes e tempo de resposta.

Para executar:
    python tests/test_api.py
"""

import requests
import time
from datetime import datetime
from typing import Dict, Any

BASE_URL = "http://localhost:5001"

# Cores para output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class APITester:
    """Classe para executar testes da API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
        self.total_time = 0
    
    def print_header(self, message: str):
        """Imprime cabeçalho de seção."""
        print(f"\n{BLUE}{'='*70}")
        print(f"  {message}")
        print(f"{'='*70}{RESET}\n")
    
    def print_test(self, test_name: str):
        """Imprime nome do teste."""
        print(f"{YELLOW}[TEST]{RESET} {test_name}...", end=" ")
    
    def print_pass(self, elapsed_ms: float = None):
        """Imprime resultado de teste passou."""
        if elapsed_ms:
            print(f"{GREEN}✓ PASSED{RESET} ({elapsed_ms:.2f}ms)")
        else:
            print(f"{GREEN}✓ PASSED{RESET}")
        self.passed += 1
    
    def print_fail(self, reason: str):
        """Imprime resultado de teste falhou."""
        print(f"{RED}✗ FAILED{RESET} - {reason}")
        self.failed += 1
    
    def test_health_endpoint(self):
        """Testa endpoint GET /health."""
        self.print_test("GET /health")
        
        try:
            start = time.time()
            response = requests.get(f"{self.base_url}/health")
            elapsed = (time.time() - start) * 1000
            self.total_time += elapsed
            
            # Validações
            assert response.status_code == 200, f"Status code {response.status_code}"
            data = response.json()
            assert "status" in data, "Campo 'status' ausente"
            assert data["status"] == "healthy", f"Status não é 'healthy': {data['status']}"
            assert "timestamp" in data, "Campo 'timestamp' ausente"
            
            self.print_pass(elapsed)
        except AssertionError as e:
            self.print_fail(str(e))
        except Exception as e:
            self.print_fail(f"Erro: {str(e)}")
    
    def test_model_info_endpoint(self):
        """Testa endpoint GET /model/info."""
        self.print_test("GET /model/info")
        
        try:
            start = time.time()
            response = requests.get(f"{self.base_url}/model/info")
            elapsed = (time.time() - start) * 1000
            self.total_time += elapsed
            
            # Validações
            assert response.status_code == 200, f"Status code {response.status_code}"
            data = response.json()
            
            required_fields = ["architecture", "lookback", "metrics", "features"]
            for field in required_fields:
                assert field in data, f"Campo '{field}' ausente"
            
            assert isinstance(data["features"], list), "Campo 'features' não é lista"
            assert len(data["features"]) > 0, "Lista 'features' está vazia"
            
            self.print_pass(elapsed)
        except AssertionError as e:
            self.print_fail(str(e))
        except Exception as e:
            self.print_fail(f"Erro: {str(e)}")
    
    def test_prediction_valid_ticker(self, ticker: str):
        """Testa predição com ticker válido."""
        self.print_test(f"POST /predict - {ticker} (válido)")
        
        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/predict",
                json={"ticker": ticker},
                headers={"Content-Type": "application/json"}
            )
            elapsed = (time.time() - start) * 1000
            self.total_time += elapsed
            
            # Validações
            assert response.status_code == 200, f"Status code {response.status_code}"
            data = response.json()
            
            assert "success" in data, "Campo 'success' ausente"
            assert data["success"] == True, "Campo 'success' não é True"
            assert "data" in data, "Campo 'data' ausente"
            
            result = data["data"]
            required_fields = [
                "ticker", "predicted_price", "current_price",
                "change_percent", "prediction_date", "confidence"
            ]
            for field in required_fields:
                assert field in result, f"Campo '{field}' ausente em data"
            
            assert result["ticker"] == ticker, f"Ticker esperado {ticker}, recebido {result['ticker']}"
            assert isinstance(result["predicted_price"], (int, float)), "predicted_price não é número"
            assert isinstance(result["current_price"], (int, float)), "current_price não é número"
            assert result["confidence"] in ["high", "medium", "low"], f"Confidence inválido: {result['confidence']}"
            
            # Validar latência
            assert elapsed < 10000, f"Latência muito alta: {elapsed:.2f}ms"
            
            self.print_pass(elapsed)
        except AssertionError as e:
            self.print_fail(str(e))
        except Exception as e:
            self.print_fail(f"Erro: {str(e)}")
    
    def test_prediction_invalid_ticker(self):
        """Testa predição com ticker inválido."""
        self.print_test("POST /predict - Ticker inválido (A)")
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"ticker": "A"},
                headers={"Content-Type": "application/json"}
            )
            
            # Validações
            assert response.status_code == 400, f"Status code esperado 400, recebido {response.status_code}"
            data = response.json()
            
            assert "error" in data, "Campo 'error' ausente"
            assert "message" in data, "Campo 'message' ausente"
            assert data["status"] == 400, "Campo 'status' não é 400"
            
            self.print_pass()
        except AssertionError as e:
            self.print_fail(str(e))
        except Exception as e:
            self.print_fail(f"Erro: {str(e)}")
    
    def test_prediction_missing_ticker(self):
        """Testa predição sem campo ticker."""
        self.print_test("POST /predict - Campo ticker ausente")
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={},
                headers={"Content-Type": "application/json"}
            )
            
            # Validações
            assert response.status_code == 400, f"Status code esperado 400, recebido {response.status_code}"
            data = response.json()
            
            assert "error" in data, "Campo 'error' ausente"
            assert "message" in data, "Campo 'message' ausente"
            
            self.print_pass()
        except AssertionError as e:
            self.print_fail(str(e))
        except Exception as e:
            self.print_fail(f"Erro: {str(e)}")
    
    def test_prediction_wrong_method(self):
        """Testa método HTTP incorreto."""
        self.print_test("GET /predict (método incorreto)")
        
        try:
            response = requests.get(f"{self.base_url}/predict")
            
            # Validações
            assert response.status_code == 405, f"Status code esperado 405, recebido {response.status_code}"
            
            self.print_pass()
        except AssertionError as e:
            self.print_fail(str(e))
        except Exception as e:
            self.print_fail(f"Erro: {str(e)}")
    
    def test_endpoint_not_found(self):
        """Testa endpoint inexistente."""
        self.print_test("GET /invalid-endpoint (404)")
        
        try:
            response = requests.get(f"{self.base_url}/invalid-endpoint")
            
            # Validações
            assert response.status_code == 404, f"Status code esperado 404, recebido {response.status_code}"
            
            self.print_pass()
        except AssertionError as e:
            self.print_fail(str(e))
        except Exception as e:
            self.print_fail(f"Erro: {str(e)}")
    
    def print_summary(self):
        """Imprime resumo dos testes."""
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        avg_time = self.total_time / self.passed if self.passed > 0 else 0
        
        print(f"\n{BLUE}{'='*70}")
        print("  RESUMO DOS TESTES")
        print(f"{'='*70}{RESET}")
        print(f"Total de testes: {total}")
        print(f"{GREEN}✓ Passou: {self.passed}{RESET}")
        print(f"{RED}✗ Falhou: {self.failed}{RESET}")
        print(f"Taxa de sucesso: {success_rate:.1f}%")
        print(f"Tempo total: {self.total_time:.2f}ms")
        print(f"Tempo médio: {avg_time:.2f}ms")
        print(f"{BLUE}{'='*70}{RESET}\n")
        
        return self.failed == 0


def main():
    """Executa todos os testes."""
    print(f"\n{BLUE}{'#'*70}")
    print("#  TESTES DE INTEGRAÇÃO - API REST")
    print(f"{'#'*70}{RESET}")
    print(f"\nData/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL: {BASE_URL}")
    
    tester = APITester(BASE_URL)
    
    # Verificar se API está acessível
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"\n{RED}✗ ERRO: Não foi possível conectar à API em {BASE_URL}{RESET}")
        print(f"{YELLOW}Certifique-se de que a API está rodando.{RESET}\n")
        return False
    
    # Executar testes
    tester.print_header("TESTES DE ENDPOINTS BÁSICOS")
    tester.test_health_endpoint()
    tester.test_model_info_endpoint()
    
    tester.print_header("TESTES DE PREDIÇÃO - CASOS VÁLIDOS")
    tester.test_prediction_valid_ticker("AAPL")
    tester.test_prediction_valid_ticker("PETR4.SA")
    tester.test_prediction_valid_ticker("VALE3.SA")
    
    tester.print_header("TESTES DE PREDIÇÃO - CASOS DE ERRO")
    tester.test_prediction_invalid_ticker()
    tester.test_prediction_missing_ticker()
    tester.test_prediction_wrong_method()
    
    tester.print_header("TESTES DE ENDPOINTS INEXISTENTES")
    tester.test_endpoint_not_found()
    
    # Resumo
    success = tester.print_summary()
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
