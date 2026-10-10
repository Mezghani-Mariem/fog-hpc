import asyncio

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import scheduler.main as scheduler


@pytest.fixture(autouse=True)
def reset_round_robin_state(monkeypatch):
    monkeypatch.setattr(scheduler, "_round_robin_index", 0)
    monkeypatch.setattr(scheduler, "_in_flight", {})
    monkeypatch.setattr(scheduler, "_node_labels_by_url", {})
    monkeypatch.setattr(scheduler, "_reservation_labels_seen", set())
    monkeypatch.setattr(scheduler, "_selection_lock", asyncio.Lock())
    scheduler.WORKERS_REACHABLE.set(0)
    scheduler.WORKERS_AVAILABLE.set(0)
    for label in ("fog", "fog-1", "fog-2", "fog-3", "configured-1", "configured-2", "unknown"):
        scheduler.RESERVATIONS.labels(label).set(0)


def worker_status(
    url,
    *,
    reachable=True,
    available=True,
    running_jobs=0,
    max_running=2,
    cpu_percent=10.0,
    node_id="fog",
):
    return {
        "url": url,
        "reachable": reachable,
        "available": available,
        "node_id": node_id,
        "running_jobs": running_jobs,
        "max_running": max_running,
        "cpu_percent": cpu_percent,
        "ram_percent": 50.0,
    }


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://test.local")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("HTTP error", request=request, response=response)

    def json(self):
        return self._payload


def metric_value(metric, *labels):
    if labels:
        return metric.labels(*labels)._value.get()
    return metric._value.get()


def test_select_node_prefers_fewer_running_jobs():
    selected = scheduler.select_node(
        [
            worker_status("http://fog-1.local:8001", running_jobs=3, cpu_percent=20.0),
            worker_status("http://fog-2.local:8001", running_jobs=0, cpu_percent=60.0),
        ]
    )

    assert selected["url"] == "http://fog-2.local:8001"


def test_select_node_uses_lower_cpu_when_running_jobs_are_equal():
    selected = scheduler.select_node(
        [
            worker_status("http://fog-1.local:8001", running_jobs=0, cpu_percent=90.0),
            worker_status("http://fog-2.local:8001", running_jobs=0, cpu_percent=20.0),
        ]
    )

    assert selected["url"] == "http://fog-2.local:8001"


def test_select_node_excludes_unreachable_and_unavailable_workers():
    selected = scheduler.select_node(
        [
            worker_status("http://fog-1.local:8001", reachable=False, available=True),
            worker_status("http://fog-2.local:8001", reachable=True, available=False),
            worker_status("http://fog-3.local:8001", reachable=True, available=True),
        ]
    )

    assert selected["url"] == "http://fog-3.local:8001"


def test_select_node_rotates_equal_load_workers_fairly():
    statuses = [
        worker_status("http://fog-1.local:8001", running_jobs=0, cpu_percent=20.0),
        worker_status("http://fog-2.local:8001", running_jobs=0, cpu_percent=20.0),
    ]

    selections = [scheduler.select_node(statuses)["url"] for _ in range(4)]

    assert selections == [
        "http://fog-1.local:8001",
        "http://fog-2.local:8001",
        "http://fog-1.local:8001",
        "http://fog-2.local:8001",
    ]


def test_reserve_node_skips_worker_when_effective_capacity_is_reached():
    async def scenario():
        scheduler._in_flight["http://fog-1.local:8001"] = 1
        selected = await scheduler.reserve_node(
            [
                worker_status(
                    "http://fog-1.local:8001",
                    running_jobs=1,
                    max_running=2,
                    cpu_percent=10.0,
                ),
                worker_status(
                    "http://fog-2.local:8001",
                    running_jobs=0,
                    max_running=2,
                    cpu_percent=50.0,
                ),
            ]
        )

        assert selected["url"] == "http://fog-2.local:8001"
        assert scheduler._in_flight == {
            "http://fog-1.local:8001": 1,
            "http://fog-2.local:8001": 1,
        }

    asyncio.run(scenario())


def test_release_node_does_not_create_negative_reservations():
    async def scenario():
        await scheduler.release_node("http://fog-1.local:8001")

        assert scheduler._in_flight == {}
        assert metric_value(scheduler.RESERVATIONS, "configured-1") == 0

    asyncio.run(scenario())


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


def test_metrics_endpoint_exposes_scheduler_metrics():
    client = TestClient(scheduler.app)
    response = client.get("/metrics")

    assert response.status_code == 200
    text = response.text
    expected_metric_names = [
        "foghpc_scheduler_jobs_total",
        "foghpc_scheduler_dispatch_seconds",
        "foghpc_scheduler_jobs_received_total",
        "foghpc_scheduler_job_outcomes_total",
        "foghpc_scheduler_dispatch_attempts_total",
        "foghpc_scheduler_workers_reachable",
        "foghpc_scheduler_workers_available",
        "foghpc_scheduler_in_flight_reservations",
    ]
    for name in expected_metric_names:
        assert name in text


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
    assert metric_value(scheduler.WORKERS_REACHABLE) == 0
    assert metric_value(scheduler.WORKERS_AVAILABLE) == 0


def test_nodes_updates_worker_availability_gauges(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "NODES",
        [
            "http://fog-1.local:8001",
            "http://fog-2.local:8001",
            "http://fog-3.local:8001",
        ],
    )

    class MixedStatusClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, timeout):
            if "fog-3" in url:
                raise httpx.ConnectError("worker unavailable", request=httpx.Request("GET", url))
            if "fog-2" in url:
                return FakeResponse(
                    200,
                    worker_status(
                        "http://fog-2.local:8001",
                        node_id="fog-2",
                        available=False,
                        running_jobs=2,
                    ),
                )
            return FakeResponse(
                200,
                worker_status("http://fog-1.local:8001", node_id="fog-1"),
            )

    monkeypatch.setattr(scheduler.httpx, "AsyncClient", MixedStatusClient)

    client = TestClient(scheduler.app)
    response = client.get("/nodes")

    assert response.status_code == 200
    assert metric_value(scheduler.WORKERS_REACHABLE) == 2
    assert metric_value(scheduler.WORKERS_AVAILABLE) == 1


def test_jobs_dispatches_valid_monte_carlo_job_to_available_worker(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "NODES",
        ["http://busy-worker.local:8001", "http://available-worker.local:8001"],
    )
    monkeypatch.setattr(scheduler, "history", {})
    monkeypatch.setattr(scheduler, "uuid4", lambda: "scheduler-job-1")
    received_before = metric_value(scheduler.JOBS_RECEIVED)
    outcome_before = metric_value(scheduler.JOB_OUTCOMES, "completed")
    attempt_before = metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-2", "completed")
    legacy_before = metric_value(scheduler.SCHEDULED, "fog-2", "completed")
    reservation_before = metric_value(scheduler.RESERVATIONS, "fog-2")

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
            assert metric_value(scheduler.RESERVATIONS, "fog-2") == reservation_before + 1
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
    assert scheduler._in_flight == {}
    assert metric_value(scheduler.RESERVATIONS, "fog-2") == reservation_before
    assert metric_value(scheduler.JOBS_RECEIVED) == received_before + 1
    assert metric_value(scheduler.JOB_OUTCOMES, "completed") == outcome_before + 1
    assert metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-2", "completed") == attempt_before + 1
    assert metric_value(scheduler.SCHEDULED, "fog-2", "completed") == legacy_before + 1


def test_job_skips_worker_unavailable_during_polling_and_uses_next_worker(monkeypatch):
    monkeypatch.setattr(
        scheduler,
        "NODES",
        ["http://fog-1.local:8001", "http://fog-2.local:8001"],
    )
    monkeypatch.setattr(scheduler, "history", {})
    monkeypatch.setattr(scheduler, "uuid4", lambda: "scheduler-job-offline")
    post_urls = []

    class OfflineThenAvailableClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, timeout):
            if "fog-1" in url:
                raise httpx.ConnectError("worker unavailable", request=httpx.Request("GET", url))
            return FakeResponse(
                200,
                worker_status("http://fog-2.local:8001", node_id="fog-2"),
            )

        async def post(self, url, json, timeout):
            post_urls.append(url)
            return FakeResponse(
                200,
                {
                    "job_id": "worker-job-2",
                    "status": "completed",
                    "node_id": "fog-2",
                    "result": {"pi_estimate": 3.14},
                    "execution_seconds": 0.01,
                },
            )

    monkeypatch.setattr(scheduler.httpx, "AsyncClient", OfflineThenAvailableClient)

    client = TestClient(scheduler.app)
    response = client.post(
        "/jobs",
        json={"kind": "monte_carlo", "size": 10000, "seed": 42},
    )

    assert response.status_code == 200
    assert response.json()["worker_url"] == "http://fog-2.local:8001"
    assert post_urls == ["http://fog-2.local:8001/jobs"]
    assert scheduler.history["scheduler-job-offline"]["status"] == "completed"
    assert scheduler._in_flight == {}


def test_concurrent_jobs_distribute_across_equal_available_workers(monkeypatch):
    async def scenario():
        monkeypatch.setattr(
            scheduler,
            "NODES",
            ["http://fog-1.local:8001", "http://fog-2.local:8001"],
        )
        post_urls = []
        both_posts_started = asyncio.Event()

        class ConcurrentAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def get(self, url, timeout):
                node_id = "fog-1" if "fog-1" in url else "fog-2"
                return FakeResponse(
                    200,
                    worker_status(url.removesuffix("/status"), node_id=node_id),
                )

            async def post(self, url, json, timeout):
                post_urls.append(url)
                if len(post_urls) == 2:
                    both_posts_started.set()
                await both_posts_started.wait()
                node_id = "fog-1" if "fog-1" in url else "fog-2"
                return FakeResponse(
                    200,
                    {
                        "job_id": f"{node_id}-job",
                        "status": "completed",
                        "node_id": node_id,
                        "result": {"pi_estimate": 3.14},
                        "execution_seconds": 0.01,
                    },
                )

        monkeypatch.setattr(scheduler.httpx, "AsyncClient", ConcurrentAsyncClient)

        job = scheduler.Job(kind="monte_carlo", size=10000, seed=42)
        await asyncio.gather(scheduler.submit(job), scheduler.submit(job))

        assert sorted(post_urls) == [
            "http://fog-1.local:8001/jobs",
            "http://fog-2.local:8001/jobs",
        ]
        assert scheduler._in_flight == {}

    asyncio.run(scenario())


def test_reservation_released_after_worker_503_and_next_worker_is_tried(monkeypatch):
    async def scenario():
        monkeypatch.setattr(
            scheduler,
            "NODES",
            ["http://fog-1.local:8001", "http://fog-2.local:8001"],
        )
        monkeypatch.setattr(scheduler, "history", {})
        monkeypatch.setattr(scheduler, "uuid4", lambda: "scheduler-job-fallback")
        received_before = metric_value(scheduler.JOBS_RECEIVED)
        completed_before = metric_value(scheduler.JOB_OUTCOMES, "completed")
        rejected_before = metric_value(scheduler.JOB_OUTCOMES, "rejected")
        rejected_attempt_before = metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "rejected")
        completed_attempt_before = metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-2", "completed")
        post_urls = []

        class CapacityThenSuccessClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def get(self, url, timeout):
                node_id = "fog-1" if "fog-1" in url else "fog-2"
                return FakeResponse(
                    200,
                    worker_status(url.removesuffix("/status"), node_id=node_id),
                )

            async def post(self, url, json, timeout):
                post_urls.append(url)
                if "fog-1" in url:
                    assert scheduler._in_flight == {"http://fog-1.local:8001": 1}
                    assert metric_value(scheduler.RESERVATIONS, "fog-1") == 1
                    return FakeResponse(503, {"detail": "Worker at capacity"})
                assert scheduler._in_flight == {"http://fog-2.local:8001": 1}
                assert metric_value(scheduler.RESERVATIONS, "fog-1") == 0
                assert metric_value(scheduler.RESERVATIONS, "fog-2") == 1
                return FakeResponse(
                    200,
                    {
                        "job_id": "worker-job-2",
                        "status": "completed",
                        "node_id": "fog-2",
                        "result": {"pi_estimate": 3.14},
                        "execution_seconds": 0.01,
                    },
                )

        monkeypatch.setattr(scheduler.httpx, "AsyncClient", CapacityThenSuccessClient)

        result = await scheduler.submit(scheduler.Job(kind="monte_carlo", size=10000, seed=42))

        assert post_urls == [
            "http://fog-1.local:8001/jobs",
            "http://fog-2.local:8001/jobs",
        ]
        assert result["worker_url"] == "http://fog-2.local:8001"
        assert scheduler.history["scheduler-job-fallback"]["status"] == "completed"
        assert scheduler._in_flight == {}
        assert metric_value(scheduler.RESERVATIONS, "fog-1") == 0
        assert metric_value(scheduler.RESERVATIONS, "fog-2") == 0
        assert metric_value(scheduler.JOBS_RECEIVED) == received_before + 1
        assert metric_value(scheduler.JOB_OUTCOMES, "completed") == completed_before + 1
        assert metric_value(scheduler.JOB_OUTCOMES, "rejected") == rejected_before
        assert metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "rejected") == rejected_attempt_before + 1
        assert metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-2", "completed") == completed_attempt_before + 1

    asyncio.run(scenario())


def test_all_workers_returning_capacity_503_records_rejected_job(monkeypatch):
    async def scenario():
        monkeypatch.setattr(
            scheduler,
            "NODES",
            ["http://fog-1.local:8001", "http://fog-2.local:8001"],
        )
        monkeypatch.setattr(scheduler, "history", {})
        monkeypatch.setattr(scheduler, "uuid4", lambda: "scheduler-job-capacity")
        received_before = metric_value(scheduler.JOBS_RECEIVED)
        rejected_before = metric_value(scheduler.JOB_OUTCOMES, "rejected")
        fog1_attempt_before = metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "rejected")
        fog2_attempt_before = metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-2", "rejected")
        post_urls = []

        class AllCapacityClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def get(self, url, timeout):
                node_id = "fog-1" if "fog-1" in url else "fog-2"
                return FakeResponse(
                    200,
                    worker_status(url.removesuffix("/status"), node_id=node_id),
                )

            async def post(self, url, json, timeout):
                post_urls.append(url)
                return FakeResponse(503, {"detail": "Worker at capacity"})

        monkeypatch.setattr(scheduler.httpx, "AsyncClient", AllCapacityClient)

        with pytest.raises(HTTPException) as exc:
            await scheduler.submit(scheduler.Job(kind="monte_carlo", size=10000, seed=42))

        assert exc.value.status_code == 503
        assert exc.value.detail == "Workers rejected task due to capacity"
        assert post_urls == [
            "http://fog-1.local:8001/jobs",
            "http://fog-2.local:8001/jobs",
        ]
        assert scheduler.history["scheduler-job-capacity"] == {
            "job_id": "scheduler-job-capacity",
            "status": "rejected",
            "error": "Workers rejected task due to capacity",
        }
        assert scheduler._in_flight == {}
        assert metric_value(scheduler.JOBS_RECEIVED) == received_before + 1
        assert metric_value(scheduler.JOB_OUTCOMES, "rejected") == rejected_before + 1
        assert metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "rejected") == fog1_attempt_before + 1
        assert metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-2", "rejected") == fog2_attempt_before + 1

    asyncio.run(scenario())


def test_reservation_released_after_timeout(monkeypatch):
    async def scenario():
        monkeypatch.setattr(scheduler, "NODES", ["http://fog-1.local:8001"])
        monkeypatch.setattr(scheduler, "history", {})
        monkeypatch.setattr(scheduler, "uuid4", lambda: "scheduler-job-timeout")
        received_before = metric_value(scheduler.JOBS_RECEIVED)
        unknown_before = metric_value(scheduler.JOB_OUTCOMES, "unknown")
        unknown_attempt_before = metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "unknown")
        post_urls = []

        class TimeoutClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def get(self, url, timeout):
                return FakeResponse(200, worker_status("http://fog-1.local:8001", node_id="fog-1"))

            async def post(self, url, json, timeout):
                post_urls.append(url)
                assert metric_value(scheduler.RESERVATIONS, "fog-1") == 1
                raise httpx.TimeoutException("request timed out")

        monkeypatch.setattr(scheduler.httpx, "AsyncClient", TimeoutClient)

        with pytest.raises(HTTPException) as exc:
            await scheduler.submit(scheduler.Job(kind="monte_carlo", size=10000, seed=42))

        assert exc.value.status_code == 504
        assert post_urls == ["http://fog-1.local:8001/jobs"]
        assert scheduler.history["scheduler-job-timeout"]["status"] == "unknown"
        assert "timeout" in scheduler.history["scheduler-job-timeout"]["error"].lower()
        assert scheduler._in_flight == {}
        assert metric_value(scheduler.RESERVATIONS, "fog-1") == 0
        assert metric_value(scheduler.JOBS_RECEIVED) == received_before + 1
        assert metric_value(scheduler.JOB_OUTCOMES, "unknown") == unknown_before + 1
        assert metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "unknown") == unknown_attempt_before + 1

    asyncio.run(scenario())


def test_worker_http_500_records_failed_job_without_retry(monkeypatch):
    async def scenario():
        monkeypatch.setattr(scheduler, "NODES", ["http://fog-1.local:8001"])
        monkeypatch.setattr(scheduler, "history", {})
        monkeypatch.setattr(scheduler, "uuid4", lambda: "scheduler-job-worker-500")
        received_before = metric_value(scheduler.JOBS_RECEIVED)
        failed_before = metric_value(scheduler.JOB_OUTCOMES, "failed")
        failed_attempt_before = metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "failed")
        post_urls = []

        class Worker500Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def get(self, url, timeout):
                return FakeResponse(200, worker_status("http://fog-1.local:8001", node_id="fog-1"))

            async def post(self, url, json, timeout):
                post_urls.append(url)
                assert metric_value(scheduler.RESERVATIONS, "fog-1") == 1
                return FakeResponse(500, {"detail": "worker failed"})

        monkeypatch.setattr(scheduler.httpx, "AsyncClient", Worker500Client)

        with pytest.raises(HTTPException) as exc:
            await scheduler.submit(scheduler.Job(kind="monte_carlo", size=10000, seed=42))

        assert exc.value.status_code == 502
        assert post_urls == ["http://fog-1.local:8001/jobs"]
        assert scheduler.history["scheduler-job-worker-500"] == {
            "job_id": "scheduler-job-worker-500",
            "status": "failed",
            "error": "Worker returned HTTP 500",
        }
        assert scheduler._in_flight == {}
        assert metric_value(scheduler.RESERVATIONS, "fog-1") == 0
        assert metric_value(scheduler.JOBS_RECEIVED) == received_before + 1
        assert metric_value(scheduler.JOB_OUTCOMES, "failed") == failed_before + 1
        assert metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "failed") == failed_attempt_before + 1

    asyncio.run(scenario())


def test_dispatch_transport_error_records_unknown_job_without_retry(monkeypatch):
    async def scenario():
        monkeypatch.setattr(scheduler, "NODES", ["http://fog-1.local:8001"])
        monkeypatch.setattr(scheduler, "history", {})
        monkeypatch.setattr(scheduler, "uuid4", lambda: "scheduler-job-transport")
        received_before = metric_value(scheduler.JOBS_RECEIVED)
        unknown_before = metric_value(scheduler.JOB_OUTCOMES, "unknown")
        unknown_attempt_before = metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "unknown")
        failed_before = metric_value(scheduler.JOB_OUTCOMES, "failed")
        post_urls = []

        class TransportErrorClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def get(self, url, timeout):
                return FakeResponse(200, worker_status("http://fog-1.local:8001", node_id="fog-1"))

            async def post(self, url, json, timeout):
                post_urls.append(url)
                assert metric_value(scheduler.RESERVATIONS, "fog-1") == 1
                raise httpx.ReadError(
                    "connection reset while reading response",
                    request=httpx.Request("POST", url),
                )

        monkeypatch.setattr(scheduler.httpx, "AsyncClient", TransportErrorClient)

        with pytest.raises(HTTPException) as exc:
            await scheduler.submit(scheduler.Job(kind="monte_carlo", size=10000, seed=42))

        assert exc.value.status_code == 502
        assert post_urls == ["http://fog-1.local:8001/jobs"]
        assert scheduler.history["scheduler-job-transport"]["status"] == "unknown"
        assert scheduler.history["scheduler-job-transport"]["worker_url"] == "http://fog-1.local:8001"
        assert "ambiguous worker communication failure" in scheduler.history["scheduler-job-transport"]["error"].lower()
        assert scheduler._in_flight == {}
        assert metric_value(scheduler.RESERVATIONS, "fog-1") == 0
        assert metric_value(scheduler.JOBS_RECEIVED) == received_before + 1
        assert metric_value(scheduler.JOB_OUTCOMES, "unknown") == unknown_before + 1
        assert metric_value(scheduler.DISPATCH_ATTEMPTS, "fog-1", "unknown") == unknown_attempt_before + 1
        assert metric_value(scheduler.JOB_OUTCOMES, "failed") == failed_before

    asyncio.run(scenario())


def test_reservation_released_after_task_cancellation(monkeypatch):
    async def scenario():
        monkeypatch.setattr(scheduler, "NODES", ["http://fog-1.local:8001"])
        post_started = asyncio.Event()
        keep_post_open = asyncio.Event()

        class HangingClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def get(self, url, timeout):
                return FakeResponse(200, worker_status("http://fog-1.local:8001", node_id="fog-1"))

            async def post(self, url, json, timeout):
                post_started.set()
                await keep_post_open.wait()
                return FakeResponse(
                    200,
                    {
                        "job_id": "worker-job-1",
                        "status": "completed",
                        "node_id": "fog-1",
                        "result": {"pi_estimate": 3.14},
                        "execution_seconds": 0.01,
                    },
                )

        monkeypatch.setattr(scheduler.httpx, "AsyncClient", HangingClient)

        task = asyncio.create_task(
            scheduler.submit(scheduler.Job(kind="monte_carlo", size=10000, seed=42))
        )
        await asyncio.wait_for(post_started.wait(), timeout=1)
        assert scheduler._in_flight == {"http://fog-1.local:8001": 1}
        assert metric_value(scheduler.RESERVATIONS, "fog-1") == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert scheduler._in_flight == {}
        assert metric_value(scheduler.RESERVATIONS, "fog-1") == 0

    asyncio.run(scenario())


def test_jobs_returns_503_when_no_worker_is_available(monkeypatch):
    monkeypatch.setattr(scheduler, "history", {})
    monkeypatch.setattr(scheduler, "uuid4", lambda: "scheduler-job-no-worker")
    received_before = metric_value(scheduler.JOBS_RECEIVED)
    rejected_before = metric_value(scheduler.JOB_OUTCOMES, "rejected")

    async def no_available_workers(client):
        return [
            worker_status("http://fog-1.local:8001", reachable=False, available=False),
            worker_status("http://fog-2.local:8001", reachable=True, available=False),
        ]

    monkeypatch.setattr(scheduler, "poll_nodes", no_available_workers)

    client = TestClient(scheduler.app)
    response = client.post(
        "/jobs",
        json={"kind": "monte_carlo", "size": 10000, "seed": 42},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "No available worker"}
    assert scheduler.history["scheduler-job-no-worker"] == {
        "job_id": "scheduler-job-no-worker",
        "status": "rejected",
        "error": "No available worker",
    }
    assert scheduler._in_flight == {}
    assert metric_value(scheduler.JOBS_RECEIVED) == received_before + 1
    assert metric_value(scheduler.JOB_OUTCOMES, "rejected") == rejected_before + 1


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
    received_before = metric_value(scheduler.JOBS_RECEIVED)

    async def fail_if_polled(client):
        pytest.fail("invalid jobs should be rejected before polling workers")

    monkeypatch.setattr(scheduler, "poll_nodes", fail_if_polled)

    client = TestClient(scheduler.app)
    response = client.post("/jobs", json=payload)

    assert response.status_code == 422
    assert metric_value(scheduler.JOBS_RECEIVED) == received_before
