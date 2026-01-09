"""Test API prediction with Production model."""
import json

import requests

API_URL = "http://localhost:5000"

print("\n" + "="*60)
print("Testing Stock Prediction API")
print("="*60)

# Test 1: Health check
print("\n1️⃣ Health Check")
print("-" * 60)
response = requests.get(f"{API_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 2: Model info
print("\n2️⃣ Model Info")
print("-" * 60)
response = requests.get(f"{API_URL}/model/info")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 3: Prediction for PETR4
print("\n3️⃣ Prediction - PETR4.SA")
print("-" * 60)
response = requests.post(
    f"{API_URL}/predict",
    json={"ticker": "PETR4.SA"},
    headers={"Content-Type": "application/json"}
)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 4: Prediction for VALE3
print("\n4️⃣ Prediction - VALE3.SA")
print("-" * 60)
response = requests.post(
    f"{API_URL}/predict",
    json={"ticker": "VALE3.SA"},
    headers={"Content-Type": "application/json"}
)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 5: Invalid ticker
print("\n5️⃣ Invalid Ticker Test")
print("-" * 60)
response = requests.post(
    f"{API_URL}/predict",
    json={"ticker": "INVALID"},
    headers={"Content-Type": "application/json"}
)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

print("\n" + "="*60)
print("✅ API Tests Complete!")
print("="*60 + "\n")
