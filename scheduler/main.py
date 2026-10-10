"""Central scheduler. MVP: synchronous dispatch of independent jobs."""
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
        eligible = [n for n in status if n.get('reachable') and n.get('available')]
        # Lower running jobs first, then lower CPU utilization.
        eligible.sort(key=lambda n: (n.get('running_jobs', 999), n.get('cpu_percent', 100)))
        if not eligible:
            raise HTTPException(503, detail='No available worker')
        # Try a different node only when it explicitly rejects the job (HTTP 503).
        # Timeout is ambiguous: task may still execute, so never blindly retry.
        for node in eligible:
            try:
                r = await client.post(f"{node['url']}/jobs", json=job.model_dump(), timeout=TIMEOUT)
                if r.status_code == 503:
                    continue
                r.raise_for_status()
                payload = r.json()
                item = {'job_id': job_id, 'status': 'completed', 'worker_url': node['url'],
                        'worker_result': payload, 'elapsed_seconds': round(time.perf_counter()-start, 4)}
                history[job_id] = item
                SCHEDULED.labels(payload['node_id'], 'completed').inc()
                LATENCY.observe(time.perf_counter()-start)
                return item
            except httpx.TimeoutException:
                history[job_id] = {'job_id': job_id, 'status': 'unknown', 'worker_url': node['url'],
                                   'error': 'Request timeout. Do not retry automatically.'}
                SCHEDULED.labels(node.get('node_id','unknown'), 'unknown').inc()
                raise HTTPException(504, detail=history[job_id])
            except (httpx.HTTPError, ValueError) as exc:
                history[job_id] = {'job_id': job_id, 'status': 'failed', 'error': str(exc)}
                raise HTTPException(502, detail=history[job_id])
    raise HTTPException(503, detail='Workers rejected task due to capacity')

@app.get('/jobs/{job_id}')
def get_job(job_id: str):
    if job_id not in history:
        raise HTTPException(404, detail='Unknown job')
    return history[job_id]

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
