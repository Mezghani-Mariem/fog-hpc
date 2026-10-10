import httpx
import pytest
from fastapi.testclient import TestClient

import scheduler.main as scheduler


def test_health_returns_configured_node_count(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "NODES",
        ["http://fog-1.local:8001", "http://fog-2.local:8001"],
    )

    client = TestClient(scheduler.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "configured_nodes": 2}


def test_nodes_marks_unavailable_workers_without_crashing(monkeypatch):
    configured_nodes = ["http://fog-1.local:8001", "http://fog-2.local:8001"]
    monkeypatch.setattr(scheduler, "NODES", configured_nodes)

    class UnavailableAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, timeout):
            raise httpx.ConnectError("worker unavailable", request=httpx.Request("GET", url))

    monkeypatch.setattr(scheduler.httpx, "AsyncClient", UnavailableAsyncClient)

    client = TestClient(scheduler.app)
    response = client.get("/nodes")

    assert response.status_code == 200
    assert response.json() == [
        {"url": "http://fog-1.local:8001", "reachable": False, "available": False},
        {"url": "http://fog-2.local:8001", "reachable": False, "available": False},
    ]


def test_jobs_dispatches_valid_monte_carlo_job_to_available_worker(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "NODES",
        ["http://busy-worker.local:8001", "http://available-worker.local:8001"],
    )
    monkeypatch.setattr(scheduler, "history", {})
    monkeypatch.setattr(scheduler, "uuid4", lambda: "scheduler-job-1")

    calls = []
    worker_payload = {
        "job_id": "worker-job-1",
        "status": "completed",
        "node_id": "fog-2",
        "result": {"pi_estimate": 3.1416},
        "execution_seconds": 0.0123,
    }

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class DispatchAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, timeout):
            calls.append(("GET", url, timeout, None))
            if url == "http://busy-worker.local:8001/status":
                return FakeResponse(
                    200,
                    {
                        "node_id": "fog-1",
                        "available": False,
                        "running_jobs": 2,
                        "max_running": 2,
                        "cpu_percent": 90.0,
                        "ram_percent": 70.0,
                    },
                )
            return FakeResponse(
                200,
                {
                    "node_id": "fog-2",
                    "available": True,
                    "running_jobs": 0,
                    "max_running": 2,
                    "cpu_percent": 15.0,
                    "ram_percent": 35.0,
                },
            )

        async def post(self, url, json, timeout):
            calls.append(("POST", url, timeout, json))
            return FakeResponse(200, worker_payload)

    monkeypatch.setattr(scheduler.httpx, "AsyncClient", DispatchAsyncClient)

    client = TestClient(scheduler.app)
    response = client.post(
        "/jobs",
        json={"kind": "monte_carlo", "size": 10000, "seed": 42},
    )

    assert response.status_code == 200
    response_json = response.json()
    assert calls == [
        ("GET", "http://busy-worker.local:8001/status", 2, None),
        ("GET", "http://available-worker.local:8001/status", 2, None),
        (
            "POST",
            "http://available-worker.local:8001/jobs",
            scheduler.TIMEOUT,
            {"kind": "monte_carlo", "size": 10000, "seed": 42},
        ),
    ]
    assert response_json["elapsed_seconds"] >= 0
    response_json_without_elapsed = {
        key: value for key, value in response_json.items() if key != "elapsed_seconds"
    }
    assert response_json_without_elapsed == {
        "job_id": "scheduler-job-1",
        "status": "completed",
        "worker_url": "http://available-worker.local:8001",
        "worker_result": worker_payload,
    }
    assert scheduler.history["scheduler-job-1"] == response_json


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "unknown", "size": 10, "seed": 42},
        {"kind": "matrix", "size": 701, "seed": 42},
        {"kind": "monte_carlo", "size": 0, "seed": 42},
        {"kind": "monte_carlo", "size": 100, "seed": -1},
    ],
)
def test_jobs_reject_invalid_requests_before_polling_workers(monkeypatch, payload):
    async def fail_if_polled(client):
        pytest.fail("invalid jobs should be rejected before polling workers")

    monkeypatch.setattr(scheduler, "poll_nodes", fail_if_polled)

    client = TestClient(scheduler.app)
    response = client.post("/jobs", json=payload)

    assert response.status_code == 422
