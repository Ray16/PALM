"""FastAPI web server for the PALM Data Splitting Agent.

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
import logging
import os
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="PALM Data Splitting Agent")

# CORS: configurable via env var, defaults to permissive for local dev
_allowed_origins = os.environ.get("PALM_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins, allow_methods=["*"], allow_headers=["*"],
)

JOBS_DIR = Path(os.environ.get("DATASAIL_JOBS_DIR", "/tmp/datasail_webapp"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Upload size limit (default 500 MB)
MAX_UPLOAD_BYTES = int(os.environ.get("PALM_MAX_UPLOAD_MB", "500")) * 1024 * 1024

# Job cleanup: delete jobs older than this many hours (default 24)
JOB_MAX_AGE_HOURS = int(os.environ.get("PALM_JOB_MAX_AGE_HOURS", "24"))

# Simple per-IP rate limiting
_rate_lock = threading.Lock()
_rate_map: Dict[str, list] = {}  # ip -> list of timestamps
RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_MAX = int(os.environ.get("PALM_RATE_LIMIT", "30"))  # max requests per window

# Thread-safe job registry
_lock: threading.Lock = threading.Lock()
_jobs: Dict[str, dict] = {}


# ── Rate limiting ────────────────────────────────────────────────────────

def _check_rate_limit(client_ip: str):
    """Raise HTTPException if client exceeds rate limit."""
    now = time.time()
    with _rate_lock:
        timestamps = _rate_map.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        if len(timestamps) >= RATE_LIMIT_MAX:
            raise HTTPException(429, "Rate limit exceeded. Please try again later.")
        timestamps.append(now)
        _rate_map[client_ip] = timestamps


# ── Job cleanup ──────────────────────────────────────────────────────────

def _cleanup_old_jobs():
    """Delete job directories older than JOB_MAX_AGE_HOURS."""
    if not JOBS_DIR.is_dir():
        return
    cutoff = time.time() - JOB_MAX_AGE_HOURS * 3600
    cleaned = 0
    for entry in JOBS_DIR.iterdir():
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
            if mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                with _lock:
                    _jobs.pop(entry.name, None)
                cleaned += 1
        except OSError:
            pass
    if cleaned:
        logger.info(f"Cleaned up {cleaned} old job(s)")


@app.on_event("startup")
async def _startup_cleanup():
    """Run cleanup on startup and schedule periodic cleanup."""
    _cleanup_old_jobs()

    async def _periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # every hour
            _cleanup_old_jobs()

    asyncio.create_task(_periodic_cleanup())


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
    return HTMLResponse(
        (WEBAPP_DIR / "static" / "index.html").read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), request: Request = None):
    """Accept a data file (or ZIP archive of a directory), return job_id + detected column names."""
    # Rate limit
    client_ip = request.client.host if request and request.client else "unknown"
    _check_rate_limit(client_ip)

    job_id   = str(uuid.uuid4())
    job_dir  = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    data      = await file.read()

    # Enforce upload size limit
    if len(data) > MAX_UPLOAD_BYTES:
        shutil.rmtree(job_dir, ignore_errors=True)
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            413,
            f"File too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Maximum upload size is {max_mb} MB. "
            f"Try reducing your dataset or compressing it."
        )

    # Sanitize filename to prevent path traversal
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(400, "Invalid filename")
    file_path = job_dir / safe_name
    file_path.write_bytes(data)

    # If ZIP, extract into a subdirectory and use that as the input path
    if safe_name.lower().endswith(".zip"):
        extract_dir = job_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(400, "Invalid ZIP file")

        # Filter out Mac metadata and hidden files
        top_entries = [e for e in extract_dir.iterdir()
                       if not e.name.startswith((".", "_"))]

        if len(top_entries) == 1 and top_entries[0].is_dir():
            # Single top-level directory: use it directly
            file_path = top_entries[0]
        elif len(top_entries) == 1 and top_entries[0].is_file():
            # Single file: use it directly (e.g., single SDF in a ZIP)
            file_path = top_entries[0]
        else:
            # Multiple files/dirs: use the extract directory
            file_path = extract_dir

    columns, fmt, hints, preview, stats = [], "unknown", None, None, None
    load_error = None
    try:
        from ..loaders import _detect_format, load_data, build_upload_hints
        fmt = _detect_format(str(file_path))
        df = load_data(str(file_path))
        columns = [c for c in df.columns if not c.startswith("_")]
        try:
            hints = build_upload_hints(df, fmt)
        except Exception:
            pass
        # Build data preview (first 10 rows) and stats
        try:
            display_cols = [c for c in df.columns if not c.startswith("_")]
            preview_df = df[display_cols].head(10)
            # Truncate long string values for display
            for col in preview_df.columns:
                if preview_df[col].dtype == object:
                    preview_df[col] = preview_df[col].astype(str).str[:80]
            preview = {
                "columns": list(preview_df.columns),
                "rows": preview_df.values.tolist(),
            }
            stats = {
                "total_rows": len(df),
                "total_columns": len(display_cols),
            }
            # Per-column stats for detected entity columns
            if hints and hints.get("column_types"):
                col_stats = {}
                for col, info in hints["column_types"].items():
                    if info.get("type"):
                        vals = df[col].dropna().astype(str)
                        col_stats[col] = {
                            "unique": int(vals.nunique()),
                            "type": info["type"],
                        }
                stats["columns"] = col_stats
        except (ValueError, KeyError, TypeError) as exc:
            logger.debug(f"Preview error: {exc}")
    except FileNotFoundError as exc:
        load_error = f"File not found: {exc}. Please check the uploaded file."
    except ValueError as exc:
        load_error = f"Could not parse the file: {exc}. Please check the format."
    except ImportError as exc:
        load_error = (
            f"Missing dependency for this file format: {exc}. "
            f"Please check that required packages (rdkit, ase, biopython) are installed."
        )
    except Exception as exc:
        logger.warning(f"Upload load error: {exc}", exc_info=True)
        load_error = f"Unexpected error loading file: {exc}"

    _init_job(job_id, file_path=str(file_path), file_name=safe_name)
    resp = {"job_id": job_id, "filename": file.filename,
            "columns": columns, "format": fmt, "hints": hints,
            "preview": preview, "stats": stats}
    if load_error:
        resp["load_error"] = load_error
    return resp


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
    e1: EntitySpec
    e2: Optional[EntitySpec] = None
    splitting: SplitSpec
    dataset_name: str = "dataset"


@app.post("/api/run/{job_id}")
async def run_job(job_id: str, req: RunRequest, request: Request = None):
    """Start the splitting pipeline as a background thread."""
    # Rate limit
    client_ip = request.client.host if request and request.client else "unknown"
    _check_rate_limit(client_ip)

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

    ratio_map = {
        "70/30": [7, 3], "80/20": [8, 2], "90/10": [9, 1],
        "70/15/15": [70, 15, 15], "80/10/10": [80, 10, 10],
    }
    splits = ratio_map.get(req.splitting.ratio, [8, 2])

    e2_config = None
    if req.e2 is not None:
        e2_config = EntityConfig(
            name=req.e2.name, type=req.e2.type,
            extract_column=req.e2.column,
            feature_sets=req.e2.feature_sets,
        )

    return PipelineConfig(
        input_file=snap["file_path"],
        output_dir=str(out_dir),
        dataset_name=req.dataset_name or job_id[:8],
        e1=EntityConfig(
            name=req.e1.name, type=req.e1.type,
            extract_column=req.e1.column,
            feature_sets=req.e1.feature_sets,
        ),
        e2=e2_config,
        splitting=SplittingConfig(
            techniques=req.splitting.techniques,
            splits=splits,
            names=["train", "val", "test"] if len(splits) == 3 else ["train", "test"],
        ),
    )


def _run_task(job_id: str, req: RunRequest):
    import traceback as _tb
    try:
        config = _build_config(job_id, req)

        def cb(pct: int, msg: str):
            _append_log(job_id, msg, progress=pct)

        run_pipeline(config, progress_callback=cb)
        _update(job_id, status="completed", progress=100)
        _append_log(job_id, "✓ Done! Results are ready for download.", progress=100)

    except KeyError as exc:
        logger.warning(f"Job {job_id} failed: missing column {exc}", exc_info=True)
        msg = f"Column {exc} not found in your data. Please check your entity column configuration."
        _update(job_id, status="failed", error=msg)
        _append_log(job_id, f"✗ {msg}")
    except ValueError as exc:
        logger.warning(f"Job {job_id} failed: {exc}", exc_info=True)
        _update(job_id, status="failed", error=str(exc))
        _append_log(job_id, f"✗ Configuration error: {exc}")
    except ImportError as exc:
        logger.warning(f"Job {job_id} failed: missing dependency {exc}", exc_info=True)
        msg = f"Missing package: {exc}. Please install the required dependencies."
        _update(job_id, status="failed", error=msg)
        _append_log(job_id, f"✗ {msg}")
    except RuntimeError as exc:
        logger.warning(f"Job {job_id} failed: {exc}", exc_info=True)
        _update(job_id, status="failed", error=str(exc))
        _append_log(job_id, f"✗ Splitting failed: {exc}")
    except Exception as exc:
        tb = _tb.format_exc()
        logger.error(f"Job {job_id} pipeline error:\n{tb}")
        _update(job_id, status="failed", error=str(exc))
        _append_log(job_id, f"✗ Unexpected error: {exc}")


# ── SSE progress stream ───────────────────────────────────────────────────

@app.get("/api/stream/{job_id}")
async def stream(job_id: str):
    """Server-Sent Events endpoint — pushes progress to the browser."""

    async def generator():
        last_len = 0
        max_idle = 600  # timeout after 10 minutes of no completion
        elapsed = 0.0
        poll_interval = 0.3
        while elapsed < max_idle:
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
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        yield f"data: {json.dumps({'error': 'stream timeout', 'status': 'failed'})}\n\n"

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


@app.get("/api/files/{job_id}")
async def list_files(job_id: str):
    """Return a tree of output files grouped by split technique."""
    snap = _snapshot(job_id)
    if snap["status"] != "completed":
        raise HTTPException(400, "Job not yet completed")

    out_dir = JOBS_DIR / job_id / "output"
    split_dir = out_dir / "split_result"
    techniques = []

    if split_dir.is_dir():
        for entry in sorted(split_dir.iterdir()):
            if entry.is_dir():
                files = sorted(
                    str(f.relative_to(split_dir))
                    for f in entry.rglob("*") if f.is_file()
                )
                techniques.append({"name": entry.name, "files": files})

    # Include plot URLs
    plot_dir = out_dir / "plots"
    plots = []
    if plot_dir.is_dir():
        for p in sorted(plot_dir.iterdir()):
            if p.suffix == ".png":
                plots.append(p.name)

    return {"techniques": techniques, "plots": plots}


@app.get("/api/metrics/{job_id}")
async def get_metrics(job_id: str):
    """Return split quality metrics for all techniques."""
    snap = _snapshot(job_id)
    if snap["status"] != "completed":
        raise HTTPException(400, "Job not yet completed")

    metrics_dir = JOBS_DIR / job_id / "output" / "metrics"
    if not metrics_dir.is_dir():
        return {"metrics": {}}

    all_metrics = {}
    for f in sorted(metrics_dir.iterdir()):
        if f.suffix == ".json":
            all_metrics[f.stem] = json.loads(f.read_text())

    return {"metrics": all_metrics}


@app.get("/api/plot/{job_id}/{filename}")
async def get_plot(job_id: str, filename: str):
    """Serve a split visualization plot."""
    # Sanitize filename
    safe = Path(filename).name
    plot_path = JOBS_DIR / job_id / "output" / "plots" / safe
    if not plot_path.is_file():
        raise HTTPException(404, "Plot not found")
    return FileResponse(plot_path, media_type="image/png")


@app.get("/api/download/{job_id}/{technique}")
async def download_technique(job_id: str, technique: str):
    """Download a single technique's split results as a ZIP."""
    snap = _snapshot(job_id)
    if snap["status"] != "completed":
        raise HTTPException(400, "Job not yet completed")

    # Sanitize technique name to prevent path traversal
    safe_technique = Path(technique).name
    if not safe_technique or safe_technique != technique:
        raise HTTPException(400, f"Invalid technique name: {technique}")

    technique_dir = JOBS_DIR / job_id / "output" / "split_result" / safe_technique
    # Verify the resolved path is under JOBS_DIR
    try:
        technique_dir.resolve().relative_to(JOBS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid technique path")

    if not technique_dir.is_dir():
        raise HTTPException(404, f"Technique folder not found: {safe_technique}")

    zip_path = JOBS_DIR / job_id / f"{safe_technique}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(technique_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(technique_dir))

    return FileResponse(
        zip_path,
        filename=f"{safe_technique}.zip",
        media_type="application/zip",
    )
