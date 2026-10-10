"""Fog Node worker"""
import os
import time
import asyncio
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

NODE_ID = os.getenv("NODE_ID", "fog-1")
MAX_RUNNING = int(os.getenv("MAX_RUNNING", "2"))
if MAX_RUNNING < 1:
    raise ValueError("MAX_RUNNING must be >= 1")

app = FastAPI(title=f"Fog Node {NODE_ID}")

JOBS = Counter("foghpc_worker_jobs_total", "Worker job outcomes", ["node", "outcome"])
RUNNING = Gauge("foghpc_worker_running_jobs", "Current running jobs", ["node"])
DURATION = Histogram("foghpc_worker_job_duration_seconds", "Job duration", ["node", "kind"])

jobs = {}
executor = ThreadPoolExecutor(max_workers=MAX_RUNNING)

class JobRequest(BaseModel):
    kind: str
    size: int = Field(ge=1, le=2_000_000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

def compute(job: JobRequest):
    rng = np.random.default_rng(job.seed)
    if job.kind == "monte_carlo":
        pts = rng.random((job.size, 2))
        inside = np.sum(pts * pts, axis=1) <= 1
        return {"pi_estimate": float(4.0 * np.mean(inside))}
    if job.kind == "matrix":
        m = rng.random((job.size, job.size))
        return {"checksum": float(np.sum(m @ m)), "dimension": job.size}
    raise ValueError("Unknown kind")

def _run_compute(job: JobRequest):
    start = time.perf_counter()
    result = compute(job)
    elapsed = time.perf_counter() - start
    return result, elapsed

@app.get("/health")
def health():
    return {"status": "ok", "node_id": NODE_ID}

@app.get("/status")
def status():
    running = sum(1 for j in jobs.values() if j["status"] == "running")
    return {
        "node_id": NODE_ID,
        "available": running < MAX_RUNNING,
        "running_jobs": running,
        "max_running": MAX_RUNNING,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": psutil.virtual_memory().percent,
    }

@app.post("/jobs")
async def submit(job: JobRequest):
    if job.kind not in ("monte_carlo", "matrix"):
        raise HTTPException(422, detail="Unsupported kind")
    if job.kind == "matrix" and job.size > 700:
        raise HTTPException(422, detail="Matrix size must be <= 700")
    if job.kind == "monte_carlo" and job.size > 2_000_000:
        raise HTTPException(422, detail="Too many samples")

    running = sum(1 for j in jobs.values() if j["status"] == "running")
    if running >= MAX_RUNNING:
        raise HTTPException(503, detail="Worker at capacity")

    job_id = str(uuid4())
    jobs[job_id] = {"job_id": job_id, "status": "queued", "kind": job.kind, "size": job.size}
    RUNNING.labels(NODE_ID).inc()
    loop = asyncio.get_running_loop()
    try:
        result, elapsed = await loop.run_in_executor(executor, _run_compute, job)
        jobs[job_id].update({
            "status": "completed",
            "result": result,
            "execution_seconds": round(elapsed, 4),
        })
        JOBS.labels(NODE_ID, "completed").inc()
        DURATION.labels(NODE_ID, job.kind).observe(elapsed)
        return {
            "job_id": job_id,
            "status": "completed",
            "node_id": NODE_ID,
            "result": result,
            "execution_seconds": round(elapsed, 4),
        }
    except Exception as exc:
        jobs[job_id].update({"status": "failed", "error": str(exc)})
        JOBS.labels(NODE_ID, "failed").inc()
        raise HTTPException(500, detail=str(exc))
    finally:
        RUNNING.labels(NODE_ID).dec()

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, detail="Unknown job")
    return jobs[job_id]

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)