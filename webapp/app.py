"""FastAPI web server for the PALM Data Splitting Agent.

Datasets (uploads / demos) and runs (one per "Run splitting") are kept by
:mod:`PALM.webapp.jobs` — persisted, queued, cancellable. Main endpoints:

  POST /api/upload | /api/demo/{id}     → new dataset (columns, preview, hints)
  POST /api/map/{dataset_id}            → t-SNE map of one entity column
  POST /api/run/{dataset_id}            → queue a run      → {run_id}
  GET  /api/runs                        → this browser's runs (X-Client-Id)
  GET  /api/runs/{run_id}[/stream]      → status + log [SSE]
  POST /api/runs/{run_id}/cancel        · DELETE /api/runs/{run_id}
  GET  /api/runs/{run_id}/{metrics,files,splitmap/{axis},download[/{tech}]}

Run (from data_splitting/, so both `webapp` and `PALM` are importable):
    bash PALM/webapp/run.sh
    # or directly:
    uvicorn PALM.webapp.app:app --host 0.0.0.0 --port 8080
"""

import asyncio
import base64
import secrets
from contextlib import asynccontextmanager
from contextvars import ContextVar
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
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────
WEBAPP_DIR = Path(__file__).parent

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from . import embedding
from . import jobs
from . import worker

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────
_shutdown = threading.Event()
_owner = ContextVar("webapp_owner", default=None)


@asynccontextmanager
async def lifespan(app):
    _shutdown.clear()
    store.load()
    store.cleanup(JOB_MAX_AGE_HOURS)
    store.start_workers()

    async def cleanup():
        while True:
            await asyncio.sleep(3600)
            await asyncio.to_thread(store.cleanup, JOB_MAX_AGE_HOURS)

    housekeeping = asyncio.create_task(cleanup())
    try:
        yield
    finally:
        _shutdown.set()
        housekeeping.cancel()
        await asyncio.gather(housekeeping, return_exceptions=True)
        await asyncio.to_thread(store.stop_workers)
        with _maps_lock:
            threads = list(_map_threads)
        for thread in threads:
            await asyncio.to_thread(thread.join, 2)


app = FastAPI(title="PALM Data Splitting Agent", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)


@app.middleware("http")
async def access_control(request: Request, call_next):
    password = os.environ.get("PALM_AUTH_PASSWORD")
    if password and request.url.path != "/healthz":
        username = os.environ.get("PALM_AUTH_USER", "palm")
        try:
            scheme, encoded = request.headers.get("authorization", "").split(" ", 1)
            credentials = base64.b64decode(encoded, validate=True).decode("utf-8")
            user, supplied = credentials.split(":", 1)
            allowed = (scheme.lower() == "basic" and
                       secrets.compare_digest(user.encode(), username.encode()) and
                       secrets.compare_digest(supplied.encode(), password.encode()))
        except (ValueError, UnicodeError):
            allowed = False
        if not allowed:
            return JSONResponse({"detail": "Sign in to use this private app"}, status_code=401,
                                headers={"WWW-Authenticate": 'Basic realm="Data Splitting", charset="UTF-8"'})
    token = _owner.set(_client(request))
    try:
        response = await call_next(request)
        if request.headers.get("x-client-id"):
            response.set_cookie("palm_client", _client(request), httponly=True,
                                samesite="lax", secure=request.url.scheme == "https")
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "private, no-store"
        return response
    finally:
        _owner.reset(token)

# CORS: configurable via env var, defaults to permissive for local dev
_allowed_origins = [x.strip() for x in os.environ.get("PALM_CORS_ORIGINS", "*").split(",") if x.strip()]
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


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/healthz")
async def health():
    return {"status": "ok"}


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

    safe_name = Path((file.filename or "").replace("\\", "/")).name
    if not safe_name or safe_name in (".", "..") or "\0" in safe_name:
        raise HTTPException(400, "Invalid filename")
    job_id, job_dir = store.new_dataset_dir()
    file_path = job_dir / safe_name
    try:
        size = 0
        with file_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Maximum upload size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
                await asyncio.to_thread(output.write, chunk)
        if not size:
            raise HTTPException(400, "The uploaded file is empty")
        if safe_name.lower().endswith(".zip"):
            from .uploads import extract_dataset_zip
            try:
                file_path = await asyncio.to_thread(
                    extract_dataset_zip, file_path, job_dir / "extracted",
                    int(os.environ.get("PALM_MAX_EXTRACT_MB", "500")) * 1024 * 1024,
                    int(os.environ.get("PALM_MAX_ARCHIVE_FILES", "5000")))
            except (ValueError, zipfile.BadZipFile) as exc:
                raise HTTPException(400, str(exc)) from exc
        return await asyncio.to_thread(_inspect_upload, job_id, file_path, safe_name,
                                       file.filename, _client(request))
    except BaseException:
        await asyncio.to_thread(shutil.rmtree, job_dir, True)
        raise
    finally:
        await file.close()


def _inspect_upload(job_id: str, file_path: Path, safe_name: str, display_name: str,
                    client: str, demo: Optional[dict] = None) -> dict:
    """Detect format/columns/hints for a stored input file and register the dataset."""
    columns, fmt, hints, preview, stats = [], "unknown", None, None, None
    load_error = None
    try:
        from PALM.benchmarks.datasail_pipeline.loaders import _detect_format, load_data, build_upload_hints
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

    resp = {"job_id": job_id, "dataset_id": job_id, "filename": display_name,
            "columns": columns, "format": fmt, "hints": hints,
            "preview": preview, "stats": stats}
    if load_error:
        resp["load_error"] = load_error
    if demo is not None:
        resp["demo"] = demo
    store.add_dataset(job_id, file_path, safe_name, client, resp, demo,
                      display_name=demo["title"] if demo else display_name)
    return resp


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """The upload/demo inspection result (columns, preview, hints) for re-use."""
    return _dataset_or_404(dataset_id)["inspect"]


# ── Run request model ─────────────────────────────────────────────────────

class EntitySpec(BaseModel):
    name: str
    type: str                    # molecule | material | biomolecule | gene
    column: str
    feature_sets: List[str] = []


class SplitSpec(BaseModel):
    techniques: List[str] = ["R", "I2"]
    ratio: str = "80/20"         # "70/30" | "80/20" | "90/10"
    # Accept splits that leave rows unassigned (inherent to double-cold I2/C2,
    # which drop interactions whose two entities land in different splits).
    allow_incomplete: bool = False


class RunRequest(BaseModel):
    e1: EntitySpec
    e2: Optional[EntitySpec] = None
    splitting: SplitSpec
    dataset_name: str = "dataset"


def _safe_dataset_name(name: Optional[str], job_id: str) -> str:
    """The dataset name ends up in output paths: keep it to [A-Za-z0-9_-]."""
    import re
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_")[:60]
    return clean or job_id[:8]


def _execute_run(run: dict, progress):
    """Run the pipeline for one run in a worker process; raise a user-facing error."""
    ds = store.get_dataset(run["dataset_id"])
    out_dir = store.output_dir(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = {"kind": "run", "request": run["request"], "input_file": ds["file_path"],
            "output_dir": str(out_dir)}
    try:
        worker.run_task(task, progress, log_path=store.run_dir(run) / "worker.log")
    except jobs.RunCancelled:
        raise
    except KeyError as exc:
        logger.warning(f"Run {run['id']} failed: missing column {exc}")
        raise RuntimeError(f"Column {exc} not found in your data. "
                           f"Please check your entity column configuration.") from exc
    except ValueError as exc:
        logger.warning(f"Run {run['id']} failed: {exc}")
        raise RuntimeError(f"Configuration error: {exc}") from exc
    except ImportError as exc:
        logger.warning(f"Run {run['id']} failed: missing dependency {exc}")
        raise RuntimeError(f"Missing package: {exc}. "
                           f"Please install the required dependencies.") from exc
    except worker.WorkerError as exc:
        logger.error(f"Run {run['id']} worker died: {exc}")
        raise RuntimeError(str(exc)) from exc
    except RuntimeError as exc:
        logger.warning(f"Run {run['id']} failed: {exc}")
        raise RuntimeError(f"Splitting failed: {exc}") from exc
    except Exception as exc:
        logger.error(f"Run {run['id']} pipeline error", exc_info=True)
        raise RuntimeError(f"Unexpected error: {exc}") from exc


store = jobs.Store(JOBS_DIR, _execute_run)


def _client(request: Optional[Request]) -> str:
    """Anonymous per-browser id (header set by the UI); scopes the run list."""
    cid = request.headers.get("x-client-id") if request else None
    if not cid and request and request.method == "GET" and request.url.path.startswith("/api/runs/"):
        cid = request.query_params.get("client_id")
    cid = cid or (request.cookies.get("palm_client") if request else None) or "anonymous"
    return cid[:64]


def _dataset_or_404(dataset_id: str) -> dict:
    ds = store.get_dataset(dataset_id)
    if ds is None or (_owner.get() is not None and ds["client"] != _owner.get()):
        raise HTTPException(404, "Dataset not found (it may have expired)")
    return ds


def _run_or_404(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None or (_owner.get() is not None and run["client"] != _owner.get()):
        raise HTTPException(404, "Run not found (it may have expired)")
    return run


def _completed_run(run_id: str) -> dict:
    run = _run_or_404(run_id)
    if run["status"] != "completed":
        raise HTTPException(400, "Run not yet completed")
    return run


def _run_summary(run: dict) -> dict:
    return {k: v for k, v in run.items() if k not in ("log", "client")}


# ── Runs ──────────────────────────────────────────────────────────────────

@app.post("/api/run/{dataset_id}")
async def submit_run(dataset_id: str, req: RunRequest, request: Request = None):
    """Queue a splitting run on an uploaded dataset (any number per dataset)."""
    client_ip = request.client.host if request and request.client else "unknown"
    _check_rate_limit(client_ip)
    ds = _dataset_or_404(dataset_id)
    if req.splitting.ratio not in worker.RATIOS:
        raise HTTPException(400, f"Unknown split ratio {req.splitting.ratio!r}; "
                                 f"choose one of {', '.join(worker.RATIOS)}")
    if not req.splitting.techniques:
        raise HTTPException(400, "Choose at least one splitting method")
    body = req.model_dump()
    name = _safe_dataset_name(req.dataset_name, dataset_id)
    body["dataset_name"] = name
    label = ds.get("display_name") or ds["file_name"]
    try:
        run = store.submit(dataset_id, body, _client(request), label)
    except jobs.QueueFull as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "5"}) from exc
    return {"run_id": run["id"], "status": run["status"], "dataset_name": name}


@app.get("/api/runs")
async def list_runs(request: Request = None):
    return {"runs": [_run_summary(r) for r in store.list_runs(_client(request))],
            "max_concurrent": jobs.MAX_CONCURRENT_RUNS}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run = _run_or_404(run_id)
    return {k: v for k, v in run.items() if k != "client"}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    _run_or_404(run_id)
    return _run_summary(store.cancel(run_id))


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    _run_or_404(run_id)
    try:
        store.delete(run_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"deleted": run_id}


@app.get("/api/runs/{run_id}/stream")
async def stream(run_id: str):
    """Server-Sent Events: the full log so far, then progress until the run ends."""
    _run_or_404(run_id)

    async def generator():
        sent = 0
        deadline = time.time() + 12 * 3600
        while time.time() < deadline:
            snap = store.log_since(run_id, sent)
            if snap is None:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                return
            sent = snap.pop("n_log")
            yield f"data: {json.dumps(snap)}\n\n"
            if snap["status"] in jobs.FINISHED:
                return
            await asyncio.sleep(0.4)
        yield f"data: {json.dumps({'error': 'stream timeout'})}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Run results ───────────────────────────────────────────────────────────

_zip_lock = threading.Lock()


def _cached_zip(src: Path, zip_path: Path):
    # Completed outputs are immutable; build each archive once, atomically.
    with _zip_lock:
        if zip_path.is_file():
            return
        temporary = zip_path.with_suffix(".zip.tmp")
        _zip_dir(src, temporary)
        temporary.replace(zip_path)


def _zip_dir(src: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(src))


@app.get("/api/runs/{run_id}/download")
async def download(run_id: str):
    run = _completed_run(run_id)
    out_dir = store.output_dir(run)
    zip_path = store.run_dir(run) / "results.zip"
    await asyncio.to_thread(_cached_zip, out_dir, zip_path)
    return FileResponse(zip_path, filename=f"splits_{run['request']['dataset_name']}_{run_id}.zip",
                        media_type="application/zip")


@app.get("/api/runs/{run_id}/files")
async def list_files(run_id: str):
    """Output files grouped by split technique, plus the pipeline's PNGs."""
    run = _completed_run(run_id)
    out_dir = store.output_dir(run)
    split_dir = out_dir / "split_result"
    techniques = []
    if split_dir.is_dir():
        for entry in sorted(split_dir.iterdir()):
            if entry.is_dir() and entry.name != "failed":
                files = sorted(str(f.relative_to(split_dir))
                               for f in entry.rglob("*") if f.is_file())
                techniques.append({"name": entry.name, "files": files})
    plot_dir = out_dir / "plots"
    plots = sorted(p.name for p in plot_dir.iterdir() if p.suffix == ".png") \
        if plot_dir.is_dir() else []
    return {"techniques": techniques, "plots": plots}


@app.get("/api/runs/{run_id}/metrics")
async def get_metrics(run_id: str):
    run = _completed_run(run_id)
    metrics_dir = store.output_dir(run) / "metrics"
    out = {}
    if metrics_dir.is_dir():
        for f in sorted(metrics_dir.iterdir()):
            if f.suffix == ".json":
                out[f.stem] = json.loads(f.read_text())
    return {"metrics": out, "timings": _technique_timings(run)}


def _technique_timings(run: dict) -> Dict[str, float]:
    """Wall-clock seconds per technique, from the pipeline's "<t> finished in Ns" log lines."""
    import re
    techniques = set(run["request"]["splitting"]["techniques"])
    full = store.get_run(run["id"]) or run
    out = {}
    for line in full.get("log", []):
        m = re.fullmatch(r"\s*(\S+) finished in ([0-9.]+)s\s*", line)
        if m and m.group(1) in techniques:
            out[m.group(1)] = float(m.group(2))
    return out


@app.get("/api/runs/{run_id}/plot/{filename}")
async def get_plot(run_id: str, filename: str):
    run = _run_or_404(run_id)
    plot_path = store.output_dir(run) / "plots" / Path(filename).name
    if not plot_path.is_file():
        raise HTTPException(404, "Plot not found")
    return FileResponse(plot_path, media_type="image/png")


@app.get("/api/runs/{run_id}/download/{technique}")
async def download_technique(run_id: str, technique: str):
    """One technique's split folder as a ZIP."""
    run = _completed_run(run_id)
    safe = Path(technique).name
    if not safe or safe != technique:
        raise HTTPException(400, f"Invalid technique name: {technique}")
    split_root = store.output_dir(run) / "split_result"
    technique_dir = split_root / safe
    try:
        technique_dir.resolve().relative_to(split_root.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid technique path")
    if not technique_dir.is_dir():
        raise HTTPException(404, f"Technique folder not found: {safe}")
    zip_path = store.run_dir(run) / f"{safe}.zip"
    await asyncio.to_thread(_cached_zip, technique_dir, zip_path)
    return FileResponse(zip_path, filename=f"{safe}.zip", media_type="application/zip")


# ── Server info ───────────────────────────────────────────────────────────

_gpu_available = None


@app.get("/api/info")
async def info():
    """What the UI needs to know about this server."""
    global _gpu_available
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        _gpu_available = False
    if _gpu_available is None:
        try:
            import torch
            _gpu_available = bool(torch.cuda.is_available())
        except ImportError:
            _gpu_available = False
    return {"gpu": _gpu_available, "job_max_age_hours": JOB_MAX_AGE_HOURS,
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "tsne_max_points": embedding.MAX_POINTS,
            "max_concurrent_jobs": worker.MAX_CONCURRENT_JOBS,
            "private_access": bool(os.environ.get("PALM_AUTH_PASSWORD"))}


# ── Static vendor assets ──────────────────────────────────────────────────

@app.get("/static/vendor/plotly.min.js")
async def plotly_js():
    """Serve the plotly.js bundled with the Python plotly package (no CDN)."""
    import plotly
    path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if not path.is_file():
        raise HTTPException(404, "plotly.min.js not available")
    return FileResponse(path, media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=86400"})


# ── Demo datasets ─────────────────────────────────────────────────────────

DEMO_DIR = WEBAPP_DIR / "demo_data"


def _demo_manifest() -> Dict[str, dict]:
    path = DEMO_DIR / "demos.json"
    if not path.is_file():
        return {}
    return {d["id"]: d for d in json.loads(path.read_text())
            if (DEMO_DIR / d["file"]).is_file()}


@app.get("/api/demos")
async def list_demos():
    return {"demos": list(_demo_manifest().values())}


@app.post("/api/demo/{demo_id}")
async def load_demo(demo_id: str, request: Request = None):
    """Start a job from a bundled demo dataset (same response as /api/upload)."""
    client_ip = request.client.host if request and request.client else "unknown"
    _check_rate_limit(client_ip)
    demo = _demo_manifest().get(demo_id)
    if demo is None:
        raise HTTPException(404, f"Unknown demo: {demo_id}")
    job_id, job_dir = store.new_dataset_dir()
    file_path = job_dir / demo["file"]
    shutil.copyfile(DEMO_DIR / demo["file"], file_path)
    return await asyncio.to_thread(_inspect_upload, job_id, file_path, demo["file"],
                                   demo["file"], _client(request), demo=demo)


# ── Embedding maps (t-SNE) ────────────────────────────────────────────────

class MapRequest(BaseModel):
    column: str
    type: str
    feature_sets: List[str] = []


_maps_lock = threading.Lock()
_map_threads = set()
_map_datasets = {}
_maps: Dict[str, dict] = {}        # map key -> {"status", "message", "error"}
# Maps are computed in worker processes, this many at a time (featurization
# with a language model and t-SNE are both heavy).
_MAP_SLOTS = threading.Semaphore(max(1, int(os.environ.get("WEBAPP_MAX_CONCURRENT_MAPS", "1"))))


def _map_task(key: str, file_path: str, req: MapRequest):
    def progress(pct, msg):
        if _shutdown.is_set():
            raise jobs.RunCancelled()
        if msg:
            with _maps_lock:
                _maps[key]["message"] = msg
    try:
        with _MAP_SLOTS:
            progress(None, "")
            if embedding.cached_map(key) is None:
                worker.run_task({"kind": "map", "file_path": file_path, "column": req.column,
                                 "type": req.type, "feature_sets": req.feature_sets}, progress)
        with _maps_lock:
            _maps[key].update(status="completed", message="Done")
    except KeyError as exc:
        with _maps_lock:
            _maps[key].update(status="failed", error=f"Column {exc} not found")
    except Exception as exc:
        logger.warning(f"Map {key} failed: {exc}")
        with _maps_lock:
            _maps[key].update(status="failed", error=str(exc))
    finally:
        with _maps_lock:
            _map_threads.discard(threading.current_thread())


@app.post("/api/map/{job_id}")
async def start_map(job_id: str, req: MapRequest):
    """Start (or reuse) the 2-D embedding map of one entity column."""
    snap = _dataset_or_404(job_id)
    feature_sets = embedding.resolve_feature_sets(req.type, req.feature_sets)
    req = MapRequest(column=req.column, type=req.type, feature_sets=feature_sets)
    key = await asyncio.to_thread(embedding.map_key, snap["file_path"], req.column, req.type, feature_sets)
    with _maps_lock:
        _map_datasets.setdefault(key, set()).add(job_id)
    if await asyncio.to_thread(embedding.cached_map, key) is not None:
        return {"key": key, "status": "completed"}
    with _maps_lock:
        state = _maps.get(key)
        if state is None or state["status"] == "failed":
            limit = max(1, int(os.environ.get("WEBAPP_MAX_PENDING_MAPS", "8")))
            if sum(s["status"] == "running" for s in _maps.values()) >= limit:
                raise HTTPException(429, "The map queue is full. Please try again shortly.")
            _maps[key] = {"status": "running", "message": "Queued...", "error": None}
            thread = threading.Thread(target=_map_task, args=(key, snap["file_path"], req), daemon=True)
            _map_threads.add(thread)
            thread.start()
        return {"key": key, **_maps[key]}


@app.get("/api/map/{job_id}/{key}")
async def get_map(job_id: str, key: str):
    dataset = _dataset_or_404(job_id)
    if len(key) != 16 or any(c not in "0123456789abcdef" for c in key):
        raise HTTPException(404, "Map not found")
    result = await asyncio.to_thread(embedding.cached_map, key)
    if result is not None:
        expected = await asyncio.to_thread(embedding.map_key, dataset["file_path"],
                                          result["column"], result["type"], result["feature_sets"])
        if expected != key:
            raise HTTPException(404, "Map not found")
        return {"status": "completed", "map": result}
    with _maps_lock:
        state = _maps.get(key) if job_id in _map_datasets.get(key, set()) else None
    if state is None:
        raise HTTPException(404, "Map not found")
    return state


@app.get("/api/runs/{run_id}/splitmap/{axis}")
async def split_map(run_id: str, axis: str):
    """Split labels of a finished run, aligned to that entity's embedding map.

    Returns the map key (fetch/compute it via /api/map) and, once the map
    exists, one label list per valid technique.
    """
    run = _completed_run(run_id)
    ds = _dataset_or_404(run["dataset_id"])
    req = run["request"]
    ent = req.get(axis)
    if axis not in ("e1", "e2") or not ent:
        raise HTTPException(404, f"No entity '{axis}' in this run")
    feature_sets = embedding.resolve_feature_sets(ent["type"], ent.get("feature_sets"))
    key = await asyncio.to_thread(embedding.map_key, ds["file_path"], ent["column"], ent["type"], feature_sets)
    map_request = {"column": ent["column"], "type": ent["type"], "feature_sets": feature_sets}
    m = await asyncio.to_thread(embedding.cached_map, key)
    if m is None:
        return {"key": key, "status": "missing", "dataset_id": ds["id"],
                "map_request": map_request}
    out_dir = store.output_dir(run)
    name = req["dataset_name"]
    comp_path = out_dir / "metrics" / f"comparison_{name}.json"
    techniques = []
    if comp_path.is_file():
        status = json.loads(comp_path.read_text()).get("status", {})
        techniques = [t for t, st in status.items() if st == "valid"]
    labels = await asyncio.to_thread(
        embedding.split_labels, ds["file_path"], ent["column"], m["ids"],
        out_dir, name, techniques)
    return {"key": key, "status": "completed", "dataset_id": ds["id"], "labels": labels}


# Remaining static assets (app.js, app.css). Mounted last so the explicit
# /static/vendor route above takes precedence.
from fastapi.staticfiles import StaticFiles  # noqa: E402

app.mount("/static", StaticFiles(directory=WEBAPP_DIR / "static"), name="static")
