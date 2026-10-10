"""Central scheduler. MVP: synchronous dispatch of independent jobs."""
import asyncio
import os
import time
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

NODES = [x.strip().rstrip('/') for x in os.getenv('FOG_NODES', 'http://localhost:8001').split(',') if x.strip()]
TIMEOUT = float(os.getenv('JOB_TIMEOUT_SECONDS', '120'))
app = FastAPI(title='Fog-HPC Scheduler')
# For an MVP: in-memory history. Restarting loses the history.
history = {}
SCHEDULED = Counter('foghpc_scheduler_jobs_total', 'Jobs dispatched', ['node', 'outcome'])
LATENCY = Histogram('foghpc_scheduler_dispatch_seconds', 'Dispatch duration')
_round_robin_index = 0
_in_flight = {}
_selection_lock = asyncio.Lock()

class Job(BaseModel):
    kind: str
    size: int = Field(ge=1, le=2_000_000)
    seed: int = Field(default=42, ge=0, le=2**32-1)

@app.get('/health')
def health():
    return {'status': 'ok', 'configured_nodes': len(NODES)}

async def poll_nodes(client):
    results = []
    for url in NODES:
        try:
            resp = await client.get(f'{url}/status', timeout=2)
            resp.raise_for_status()
            data = resp.json()
            results.append({'url': url, **data, 'reachable': True})
        except (httpx.HTTPError, ValueError):
            results.append({'url': url, 'reachable': False, 'available': False})
    return results

def _effective_running_jobs(node, reservations=None):
    reservations = reservations or {}
    return node.get('running_jobs', 999) + reservations.get(node.get('url'), 0)

def _node_score(node, reservations=None):
    return (_effective_running_jobs(node, reservations), node.get('cpu_percent', 100))

def select_nodes(statuses, reservations=None, excluded_urls=None):
    global _round_robin_index

    reservations = reservations or {}
    excluded_urls = excluded_urls or set()
    eligible = []
    for node in statuses:
        url = node.get('url')
        if url in excluded_urls or not node.get('reachable') or not node.get('available'):
            continue
        max_running = node.get('max_running')
        if max_running is not None and _effective_running_jobs(node, reservations) >= max_running:
            continue
        eligible.append(node)
    if not eligible:
        return []

    grouped = {}
    for node in eligible:
        grouped.setdefault(_node_score(node, reservations), []).append(node)

    ordered = []
    for score in sorted(grouped):
        candidates = grouped[score]
        if len(candidates) > 1:
            start = _round_robin_index % len(candidates)
            _round_robin_index += 1
            candidates = candidates[start:] + candidates[:start]
        ordered.extend(candidates)
    return ordered

def select_node(statuses, reservations=None, excluded_urls=None):
    ordered = select_nodes(statuses, reservations, excluded_urls)
    if not ordered:
        return None
    return ordered[0]

async def reserve_node(statuses, excluded_urls=None):
    async with _selection_lock:
        node = select_node(statuses, _in_flight, excluded_urls)
        if not node:
            return None
        url = node['url']
        _in_flight[url] = _in_flight.get(url, 0) + 1
        return node

async def release_node(url):
    async with _selection_lock:
        count = _in_flight.get(url, 0)
        if count <= 1:
            _in_flight.pop(url, None)
        else:
            _in_flight[url] = count - 1

async def release_node_safely(url):
    release_task = asyncio.create_task(release_node(url))
    try:
        await asyncio.shield(release_task)
    except asyncio.CancelledError:
        await release_task
        raise

@app.get('/nodes')
async def nodes():
    async with httpx.AsyncClient() as client:
        return await poll_nodes(client)

@app.post('/jobs')
async def submit(job: Job):
    if job.kind not in ('matrix', 'monte_carlo') or (job.kind == 'matrix' and job.size > 700):
        raise HTTPException(422, detail='Unsupported kind or size (matrix <= 700)')
    job_id = str(uuid4())
    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        status = await poll_nodes(client)
        rejected_urls = set()
        # Try a different node only when it explicitly rejects the job (HTTP 503).
        # Timeout is ambiguous: task may still execute, so never blindly retry.
        while True:
            node = await reserve_node(status, rejected_urls)
            if not node:
                detail = 'Workers rejected task due to capacity' if rejected_urls else 'No available worker'
                raise HTTPException(503, detail=detail)
            url = node['url']
            try:
                r = await client.post(f"{url}/jobs", json=job.model_dump(), timeout=TIMEOUT)
                if r.status_code == 503:
                    rejected_urls.add(url)
                    continue
                r.raise_for_status()
                payload = r.json()
                item = {'job_id': job_id, 'status': 'completed', 'worker_url': url,
                        'worker_result': payload, 'elapsed_seconds': round(time.perf_counter()-start, 4)}
                history[job_id] = item
                SCHEDULED.labels(payload['node_id'], 'completed').inc()
                LATENCY.observe(time.perf_counter()-start)
                return item
            except httpx.TimeoutException:
                history[job_id] = {'job_id': job_id, 'status': 'unknown', 'worker_url': url,
                                   'error': 'Request timeout. Do not retry automatically.'}
                SCHEDULED.labels(node.get('node_id','unknown'), 'unknown').inc()
                raise HTTPException(504, detail=history[job_id])
            except (httpx.HTTPError, ValueError) as exc:
                history[job_id] = {'job_id': job_id, 'status': 'failed', 'error': str(exc)}
                raise HTTPException(502, detail=history[job_id])
            finally:
                await release_node_safely(url)

@app.get('/jobs/{job_id}')
def get_job(job_id: str):
    if job_id not in history:
        raise HTTPException(404, detail='Unknown job')
    return history[job_id]

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
