"""Datasets and splitting runs: persisted on disk, queued, cancellable.

A *dataset* is one upload (or demo copy); a *run* is one press of "Run
splitting" on a dataset — a dataset can have any number of runs, each with its
own request and output directory:

    JOBS_DIR/<dataset_id>/dataset.json                 inspection result + metadata
    JOBS_DIR/<dataset_id>/<input file or extracted/>
    JOBS_DIR/<dataset_id>/runs/<run_id>/run.json       request, status, progress, log
    JOBS_DIR/<dataset_id>/runs/<run_id>/output/        pipeline output

Both are mirrored to disk, so the run list survives server restarts (including
``run.sh --dev`` reloads). On start-up, runs that were queued are queued again;
runs that were executing are marked ``interrupted`` (their thread died with the
old process).

Runs execute on a small pool of worker threads (``WEBAPP_MAX_CONCURRENT_RUNS``,
default 2) so a burst of submissions queues instead of oversubscribing the node.
Each thread hands its run to a worker *process* (:mod:`PALM.webapp.worker`), so a
run can never stall the server or another run. A queued run can be cancelled
outright; a running one is stopped within about a second (the executor's
periodic ``progress(None, "")`` raises :class:`RunCancelled`, and the worker
process is killed).

Each record carries the submitting browser's anonymous client id, so the run
list shows a browser only its own runs. Ids are random UUIDs; this is
separation, not authentication.
"""

import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
FINISHED = ("completed", "failed", "cancelled", "interrupted")
MAX_CONCURRENT_RUNS = max(1, int(os.environ.get("WEBAPP_MAX_CONCURRENT_RUNS", "2")))
_PERSIST_EVERY_S = 2.0          # throttle run.json rewrites while a run streams logs


class RunCancelled(Exception):
    """Raised from a run's progress callback once cancellation is requested."""


class RunInterrupted(Exception):
    """The service is shutting down; preserve an honest interrupted status."""


class QueueFull(Exception):
    """The bounded run queue cannot accept another submission yet."""


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str))
    tmp.replace(path)


class Store:
    def __init__(self, root: Path, executor: Callable[[dict, Callable], None]):
        """``executor(run, progress)`` runs one job; ``progress(pct, msg)`` reports."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._executor = executor
        self._lock = threading.RLock()
        self.datasets: Dict[str, dict] = {}
        self.runs: Dict[str, dict] = {}
        self._cancel: set = set()
        self._last_persist: Dict[str, float] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._stopping = threading.Event()
        self.max_pending = max(1, int(os.environ.get("WEBAPP_MAX_PENDING_RUNS", "32")))
        self.max_client_pending = max(1, int(os.environ.get("WEBAPP_MAX_CLIENT_RUNS", "4")))

    # ── paths ──────────────────────────────────────────────────────────────
    def dataset_dir(self, dataset_id: str) -> Path:
        return self.root / dataset_id

    def run_dir(self, run: dict) -> Path:
        return self.root / run["dataset_id"] / "runs" / run["id"]

    def output_dir(self, run: dict) -> Path:
        return self.run_dir(run) / "output"

    # ── start-up ───────────────────────────────────────────────────────────
    def load(self):
        """Re-read every dataset/run from disk and requeue unfinished queued runs."""
        if any(t.is_alive() for t in self._workers):
            raise RuntimeError("Stop workers before reloading their queue")
        self._queue = queue.Queue()
        self.datasets.clear()
        self.runs.clear()
        requeue = []
        for meta in sorted(self.root.glob("*/dataset.json")):
            try:
                ds = json.loads(meta.read_text())
            except (OSError, ValueError):
                continue
            self.datasets[ds["id"]] = ds
            for rj in sorted((meta.parent / "runs").glob("*/run.json")):
                try:
                    run = json.loads(rj.read_text())
                except (OSError, ValueError):
                    continue
                if run["status"] == "running":
                    run.update(status="interrupted", finished=time.time(),
                               error="The server restarted while this run was executing.")
                    run["log"].append("✗ Interrupted: the server restarted during this run.")
                    _write_json(rj, run)
                elif run["status"] == "queued":
                    requeue.append(run)
                self.runs[run["id"]] = run
        for run in sorted(requeue, key=lambda r: r["created"]):
            self._queue.put(run["id"])
        logger.info("Loaded %d dataset(s), %d run(s); requeued %d",
                    len(self.datasets), len(self.runs), len(requeue))

    def start_workers(self):
        self._stopping.clear()
        self._workers = [t for t in self._workers if t.is_alive()]
        while len(self._workers) < MAX_CONCURRENT_RUNS:
            t = threading.Thread(target=self._worker, daemon=True,
                                 name=f"run-worker-{len(self._workers)}")
            t.start()
            self._workers.append(t)

    def stop_workers(self, timeout: float = 10):
        """Reap active worker processes and leave queued runs persisted for restart."""
        self._stopping.set()
        for _ in self._workers:
            self._queue.put(None)
        deadline = time.monotonic() + timeout
        for thread in self._workers:
            thread.join(max(0, deadline - time.monotonic()))

    # ── datasets ───────────────────────────────────────────────────────────
    def new_dataset_dir(self) -> (str, Path):
        dataset_id = str(uuid.uuid4())
        d = self.dataset_dir(dataset_id)
        d.mkdir(parents=True)
        return dataset_id, d

    def add_dataset(self, dataset_id: str, file_path: Path, file_name: str,
                    client: str, inspect: dict, demo: Optional[dict] = None,
                    display_name: Optional[str] = None) -> dict:
        ds = {"id": dataset_id, "file_path": str(file_path), "file_name": file_name,
              "display_name": display_name or file_name,
              "client": client, "created": time.time(), "inspect": inspect,
              "demo_id": demo["id"] if demo else None}
        with self._lock:
            self.datasets[dataset_id] = ds
        _write_json(self.dataset_dir(dataset_id) / "dataset.json", ds)
        return ds

    def get_dataset(self, dataset_id: str) -> Optional[dict]:
        with self._lock:
            return self.datasets.get(dataset_id)

    # ── runs ───────────────────────────────────────────────────────────────
    def submit(self, dataset_id: str, request: dict, client: str, label: str) -> dict:
        run = {"id": uuid.uuid4().hex[:12], "dataset_id": dataset_id, "client": client,
               "request": request, "label": label, "status": "queued", "progress": 0,
               "phase": "Queued", "log": [], "error": None, "created": time.time(),
               "started": None, "finished": None}
        with self._lock:
            active = [r for r in self.runs.values() if r["status"] in ACTIVE]
            if len(active) >= self.max_pending:
                raise QueueFull("The run queue is full. Please wait for a run to finish.")
            if sum(r["client"] == client for r in active) >= self.max_client_pending:
                raise QueueFull("You already have the maximum number of active runs. "
                                "Wait for one to finish or cancel it before submitting another.")
            self.runs[run["id"]] = run
            self._persist(run, force=True)
        self._queue.put(run["id"])
        return dict(run)

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._lock:
            run = self.runs.get(run_id)
            return None if run is None else {**run, "log": list(run["log"])}

    def list_runs(self, client: Optional[str]) -> List[dict]:
        with self._lock:
            runs = [r for r in self.runs.values() if client is None or r["client"] == client]
            out = [{k: v for k, v in r.items() if k != "log"} for r in runs]
        position = self._queue_positions()
        for r in out:
            r["queue_position"] = position.get(r["id"])
        return sorted(out, key=lambda r: r["created"], reverse=True)

    def log_since(self, run_id: str, start: int):
        with self._lock:
            run = self.runs.get(run_id)
            if run is None:
                return None
            return {"status": run["status"], "progress": run["progress"],
                    "phase": run["phase"], "error": run["error"],
                    "log": run["log"][start:], "n_log": len(run["log"]),
                    "queue_position": self._queue_positions().get(run_id)}

    def cancel(self, run_id: str) -> dict:
        with self._lock:
            run = self.runs[run_id]
            if run["status"] == "queued":
                run.update(status="cancelled", finished=time.time(), phase="Cancelled")
                run["log"].append("✗ Cancelled before it started.")
                self._persist(run, force=True)
            elif run["status"] == "running":
                self._cancel.add(run_id)
                run["phase"] = "Cancelling…"
                run["log"].append("… Cancellation requested; stopping at the next step.")
            return dict(run)

    def delete(self, run_id: str):
        with self._lock:
            run = self.runs.get(run_id)
            if run is None:
                return
            if run["status"] in ACTIVE:
                raise ValueError("Cancel the run before deleting it")
            self.runs.pop(run_id)
        shutil.rmtree(self.run_dir(run), ignore_errors=True)

    # ── execution ──────────────────────────────────────────────────────────
    def _queue_positions(self) -> Dict[str, int]:
        with self._lock:
            waiting = sorted((r for r in self.runs.values() if r["status"] == "queued"),
                             key=lambda r: r["created"])
        return {r["id"]: i + 1 for i, r in enumerate(waiting)}

    def _worker(self):
        while True:
            run_id = self._queue.get()
            try:
                if run_id is None or self._stopping.is_set():
                    return
                with self._lock:
                    run = self.runs.get(run_id)
                    if run is None or run["status"] != "queued":
                        continue
                    run.update(status="running", started=time.time(), phase="Starting")
                    self._persist(run, force=True)
                self._execute(run)
            finally:
                self._queue.task_done()

    def _execute(self, run: dict):
        run_id = run["id"]

        def progress(pct: int, msg: str):
            if self._stopping.is_set():
                raise RunInterrupted()
            if run_id in self._cancel:
                raise RunCancelled()
            msg = (msg or "").strip()
            with self._lock:
                if pct is not None:
                    run["progress"] = int(pct)
                if msg:
                    run["log"].append(msg)
                    if msg[:1] == "[" and "/5]" in msg[:6]:
                        run["phase"] = msg.split("]", 1)[1].strip()
                self._persist(run)

        try:
            self._executor(run, progress)
            final = {"status": "completed", "progress": 100, "phase": "Complete"}
            tail = "✓ Done! Results are ready."
        except RunCancelled:
            final = {"status": "cancelled", "phase": "Cancelled"}
            tail = "✗ Cancelled."
        except RunInterrupted:
            final = {"status": "interrupted", "phase": "Interrupted",
                     "error": "The server stopped while this run was executing."}
            tail = "✗ Interrupted: the server stopped during this run."
        except Exception as exc:          # executor already formats the message
            final = {"status": "failed", "phase": "Failed", "error": str(exc)}
            tail = f"✗ {exc}"
        with self._lock:
            self._cancel.discard(run_id)
            run.update(final, finished=time.time())
            run["log"].append(tail)
            self._persist(run, force=True)

    def _persist(self, run: dict, force: bool = False):
        now = time.time()
        if not force and now - self._last_persist.get(run["id"], 0) < _PERSIST_EVERY_S:
            return
        self._last_persist[run["id"]] = now
        try:
            _write_json(self.run_dir(run) / "run.json", run)
        except OSError as exc:
            logger.warning("Could not persist run %s: %s", run["id"], exc)

    # ── housekeeping ───────────────────────────────────────────────────────
    def cleanup(self, max_age_hours: float) -> int:
        """Delete datasets (and their runs) with no activity for ``max_age_hours``."""
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        with self._lock:
            by_ds: Dict[str, list] = {}
            for r in self.runs.values():
                by_ds.setdefault(r["dataset_id"], []).append(r)
            stale = []
            for ds_id, ds in self.datasets.items():
                runs = by_ds.get(ds_id, [])
                if any(r["status"] in ACTIVE for r in runs):
                    continue
                last = max([ds["created"]] + [r["finished"] or r["created"] for r in runs])
                if last < cutoff:
                    stale.append(ds_id)
            for ds_id in stale:
                self.datasets.pop(ds_id, None)
                for r in by_ds.get(ds_id, []):
                    self.runs.pop(r["id"], None)
        for ds_id in stale:
            shutil.rmtree(self.dataset_dir(ds_id), ignore_errors=True)
            removed += 1
        # orphan directories (e.g. failed uploads that never registered)
        for entry in self.root.iterdir():
            if entry.is_dir() and entry.name not in self.datasets:
                try:
                    if entry.stat().st_mtime < cutoff:
                        shutil.rmtree(entry, ignore_errors=True)
                except OSError:
                    pass
        return removed
