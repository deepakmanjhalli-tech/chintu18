"""
test_app.py
===========
Automated test suite to verify all endpoints, validation logic,
and ML predictions of the Crop Recommendation System.
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def test_home():
    print("[TEST 1] Testing Homepage (GET /)...")
    req = urllib.request.Request(f"{BASE_URL}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode('utf-8')
        assert "Basic Crop Recommendation System" in html
        assert "cropForm" in html
    print("         PASS: Homepage loaded with 200 OK and expected HTML elements.")

def test_health():
    print("[TEST 2] Testing Health Endpoint (GET /health)...")
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get('status') == 'healthy'
        assert data.get('model_loaded') is True
    print("         PASS: Health check reports model loaded and healthy.")

def test_predict_valid_rice():
    print("[TEST 3] Testing Valid Prediction (Rice)...")
    payload = {
        'N': 80, 'P': 48, 'K': 40,
        'temperature': 24.0, 'humidity': 82.5,
        'ph': 6.5, 'rainfall': 240.0
    }
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data['success'] is True
        assert data['crop_name'].lower() == 'rice'
        assert 'description' in data['crop_info']
    print(f"         PASS: Recommended Crop: {data['crop_name']} (Matches expected Rice).")

def test_predict_valid_coffee():
    print("[TEST 4] Testing Valid Prediction (Coffee)...")
    payload = {
        'N': 101, 'P': 28, 'K': 30,
        'temperature': 25.5, 'humidity': 58.0,
        'ph': 6.8, 'rainfall': 158.0
    }
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data['success'] is True
        assert data['crop_name'].lower() == 'coffee'
    print(f"         PASS: Recommended Crop: {data['crop_name']} (Matches expected Coffee).")

def test_predict_validation_errors():
    print("[TEST 5] Testing Validation Errors (Missing & Out of Range)...")
    # Payload with missing 'rainfall' and out-of-range 'N' and 'ph'
    payload = {
        'N': 999,      # Max is 140
        'P': 40,
        'K': 40,
        'temperature': 24.0,
        'humidity': 80.0,
        'ph': 14.5     # Max is 10.0
        # missing 'rainfall'
    }
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        urllib.request.urlopen(req)
        assert False, "Expected 400 error but request succeeded"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        resp_data = json.loads(e.read().decode('utf-8'))
        assert resp_data['success'] is False
        assert len(resp_data['errors']) >= 3
        print(f"         PASS: Correctly rejected with 400 Bad Request and {len(resp_data['errors'])} error messages:")
        for err in resp_data['errors']:
            print(f"               - {err}")

if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING AUTOMATED VERIFICATION TESTS")
    print("=" * 60)
    test_home()
    test_health()
    test_predict_valid_rice()
    test_predict_valid_coffee()
    test_predict_validation_errors()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
