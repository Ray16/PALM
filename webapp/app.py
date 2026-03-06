"""FastAPI web server for the Data Splitting Agent.

Endpoints:
  GET  /                       → serve the single-page UI
  POST /api/upload             → upload data file, returns columns list
  POST /api/run/{job_id}       → start splitting job in background
  GET  /api/stream/{job_id}    → SSE stream of progress events
  GET  /api/download/{job_id}  → download results as ZIP
  GET  /api/status/{job_id}    → poll job status (JSON)

Run:
    bash PALM/webapp/run.sh
    # or directly:
    uvicorn PALM.webapp.app:app --host 0.0.0.0 --port 8080
"""

import asyncio
import json
import os
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────
WEBAPP_DIR = Path(__file__).parent
PALM_DIR   = WEBAPP_DIR.parent

from ..config import (
    EntityConfig, FilterConfig, PipelineConfig, SplittingConfig,
)
from ..pipeline import run_pipeline

# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="Data Splitting Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

JOBS_DIR = Path(os.environ.get("DATASAIL_JOBS_DIR", "/tmp/datasail_webapp"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Thread-safe job registry
_lock: threading.Lock = threading.Lock()
_jobs: Dict[str, dict] = {}


# ── Job helpers ───────────────────────────────────────────────────────────

def _init_job(job_id: str, **kw):
    with _lock:
        _jobs[job_id] = {"status": "uploaded", "progress": 0,
                         "log": [], "error": None, **kw}


def _update(job_id: str, **kw):
    with _lock:
        _jobs[job_id].update(kw)


def _append_log(job_id: str, msg: str, progress: Optional[int] = None):
    msg = msg.strip()
    if not msg:
        return
    with _lock:
        _jobs[job_id]["log"].append(msg)
        if progress is not None:
            _jobs[job_id]["progress"] = progress


def _snapshot(job_id: str) -> dict:
    with _lock:
        if job_id not in _jobs:
            raise HTTPException(404, "Job not found")
        return dict(_jobs[job_id])


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((WEBAPP_DIR / "static" / "index.html").read_text())


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Accept a data file, return job_id + detected column names."""
    job_id   = str(uuid.uuid4())
    job_dir  = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    data      = await file.read()
    file_path = job_dir / file.filename
    file_path.write_bytes(data)

    columns, fmt = [], "unknown"
    try:
        import pandas as pd
        from ..loaders import _detect_format
        fmt = _detect_format(str(file_path))
        if fmt == "csv":
            df = pd.read_csv(file_path, nrows=0)
            columns = [c for c in df.columns if c != "_row_id"]
        elif fmt == "json":
            df = pd.read_json(file_path)
            columns = [c for c in df.columns if c != "_row_id"]
    except Exception:
        pass

    _init_job(job_id, file_path=str(file_path), file_name=file.filename)
    return {"job_id": job_id, "filename": file.filename,
            "columns": columns, "format": fmt}


# ── Run request model ─────────────────────────────────────────────────────

class EntitySpec(BaseModel):
    name: str
    type: str                    # molecule | material | biomolecule | gene
    column: str
    feature_sets: List[str] = []


class SplitSpec(BaseModel):
    techniques: List[str] = ["R", "I2"]
    ratio: str = "80/20"         # "70/30" | "80/20" | "90/10"


class RunRequest(BaseModel):
    e: EntitySpec
    f: EntitySpec
    splitting: SplitSpec
    dataset_name: str = "dataset"


@app.post("/api/run/{job_id}")
async def run_job(job_id: str, req: RunRequest):
    """Start the splitting pipeline as a background thread."""
    snap = _snapshot(job_id)
    if snap["status"] == "running":
        raise HTTPException(400, "Job is already running")

    _update(job_id, status="running", progress=0, log=[], error=None)
    threading.Thread(target=_run_task, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status": "started"}


def _build_config(job_id: str, req: RunRequest) -> PipelineConfig:
    snap    = _snapshot(job_id)
    job_dir = JOBS_DIR / job_id
    out_dir = job_dir / "output"
    out_dir.mkdir(exist_ok=True)

    ratio_map = {"70/30": [7, 3], "80/20": [8, 2], "90/10": [9, 1]}
    splits = ratio_map.get(req.splitting.ratio, [8, 2])

    return PipelineConfig(
        input_file=snap["file_path"],
        output_dir=str(out_dir),
        dataset_name=req.dataset_name or job_id[:8],
        e=EntityConfig(
            name=req.e.name, type=req.e.type,
            extract_column=req.e.column,
            feature_sets=req.e.feature_sets,
        ),
        f=EntityConfig(
            name=req.f.name, type=req.f.type,
            extract_column=req.f.column,
            feature_sets=req.f.feature_sets,
        ),
        splitting=SplittingConfig(
            techniques=req.splitting.techniques,
            splits=splits,
            names=["train", "test"],
        ),
    )


def _run_task(job_id: str, req: RunRequest):
    try:
        config = _build_config(job_id, req)

        def cb(pct: int, msg: str):
            _append_log(job_id, msg, progress=pct)

        run_pipeline(config, progress_callback=cb)
        _update(job_id, status="completed", progress=100)
        _append_log(job_id, "✓ Done! Results are ready for download.", progress=100)

    except Exception as exc:
        _update(job_id, status="failed", error=str(exc))
        _append_log(job_id, f"✗ Error: {exc}")


# ── SSE progress stream ───────────────────────────────────────────────────

@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    """Server-Sent Events endpoint — pushes progress to the browser."""

    async def generator():
        last_len = 0
        while True:
            with _lock:
                if job_id not in _jobs:
                    yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                    return
                job      = dict(_jobs[job_id])
                new_logs = job["log"][last_len:]
                last_len = len(job["log"])

            yield f"data: {json.dumps({'progress': job['progress'], 'status': job['status'], 'log': new_logs, 'error': job['error']})}\n\n"

            if job["status"] in ("completed", "failed"):
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Status poll (fallback for browsers that don't support SSE well) ────────

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    return _snapshot(job_id)


# ── Download results ──────────────────────────────────────────────────────

@app.get("/api/download/{job_id}")
async def download(job_id: str):
    snap = _snapshot(job_id)
    if snap["status"] != "completed":
        raise HTTPException(400, "Job not yet completed")

    out_dir  = JOBS_DIR / job_id / "output"
    zip_path = JOBS_DIR / job_id / "results.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(out_dir))

    name = snap.get("file_name", job_id).rsplit(".", 1)[0]
    return FileResponse(
        zip_path,
        filename=f"datasail_{name}.zip",
        media_type="application/zip",
    )
