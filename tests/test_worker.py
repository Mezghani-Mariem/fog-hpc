import pytest
from fastapi.testclient import TestClient
from fog_node.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_monte_carlo():
    r = client.post("/jobs", json={"kind": "monte_carlo", "size": 10000, "seed": 42})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "completed"
    assert 3.0 < data["result"]["pi_estimate"] < 3.3

def test_matrix():
    r = client.post("/jobs", json={"kind": "matrix", "size": 100, "seed": 42})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "completed"
    assert data["result"]["dimension"] == 100

def test_invalid_kind():
    r = client.post("/jobs", json={"kind": "unknown", "size": 10, "seed": 42})
    assert r.status_code == 422

def test_matrix_too_large():
    r = client.post("/jobs", json={"kind": "matrix", "size": 1000, "seed": 42})
    assert r.status_code == 422