"""Heavy work (splitting runs, t-SNE maps) in a child process, not a server thread.

Running the pipeline on threads of the server process is unsafe: two runs (or a
run and a map) in one interpreter can starve each other of the GIL — e.g. one
thread inside sklearn/threadpoolctl's ``dl_iterate_phdr`` callback, another in
numpy — and the event loop starves with them, so the whole site stops
answering. Each job therefore runs as ``python -m PALM.webapp.worker`` in its own
session (process group):

  * the server only reads JSON-line messages from the child's stdout, so it
    stays responsive whatever the child does;
  * cancelling kills the process group at once instead of waiting for the
    current solver call to finish;
  * a crash or out-of-memory kill fails that one job, not the server.

Protocol: the parent writes one JSON task to the child's stdin; the child
writes ``{"progress": [pct, msg]}`` lines and finally ``{"done": true}`` or
``{"error": {"type", "args", "traceback"}}`` to the original stdout. Anything
the pipeline prints (Python or C level) goes to stderr, which the parent sends
to a log file.
"""

import json
import inspect
from functools import wraps
import logging
import os
import selectors
import signal
import subprocess
import sys
import time
import traceback
import threading
from pathlib import Path
from typing import Callable, Optional

MAX_CONCURRENT_JOBS = max(1, int(os.environ.get('WEBAPP_MAX_CONCURRENT_JOBS', '2')))
_WORK_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)

ROOT_DIR = Path(__file__).resolve().parents[2]          # directory containing the PALM checkout

RATIOS = {
    "70/30": [7, 3], "80/20": [8, 2], "90/10": [9, 1],
    "70/15/15": [70, 15, 15], "80/10/10": [80, 10, 10],
}

# Exceptions re-raised in the parent with their own type, so the server can
# word its error messages by cause; anything else arrives as RuntimeError.
_KNOWN_ERRORS = {e.__name__: e for e in (KeyError, ValueError, ImportError,
                                          FileNotFoundError, RuntimeError, MemoryError)}


class WorkerError(Exception):
    """The child process died without reporting a result."""


# ── child side ────────────────────────────────────────────────────────────

def build_config(request: dict, input_file: str, output_dir: str):
    """PipelineConfig for one run request (the dict the API validated)."""
    from PALM.benchmarks.datasail_pipeline.config import (
        EntityConfig, PipelineConfig, SplittingConfig,
    )

    def entity(e):
        return EntityConfig(name=e["name"], type=e["type"], extract_column=e["column"],
                            feature_sets=list(e.get("feature_sets") or []))

    s = request["splitting"]
    splits = RATIOS.get(s.get("ratio"), [8, 2])
    return PipelineConfig(
        input_file=input_file,
        output_dir=output_dir,
        dataset_name=request["dataset_name"],
        e1=entity(request["e1"]),
        e2=entity(request["e2"]) if request.get("e2") else None,
        splitting=SplittingConfig(
            techniques=list(s["techniques"]),
            splits=splits,
            names=["train", "val", "test"] if len(splits) == 3 else ["train", "test"],
            allow_incomplete=bool(s.get("allow_incomplete", False)),
            threads=max(1, int(os.environ.get("WEBAPP_WORKER_THREADS", "2"))),
        ),
    )


class _ForwardTimings(logging.Handler):
    """Forward DataSAIL's per-technique "<t> finished in Ns" log lines as progress.

    The pipeline reports timings of its added techniques (hypergraph, lowrank)
    through the progress callback but DataSAIL's only through logging.
    """

    def __init__(self, report):
        super().__init__(logging.INFO)
        self._report = report

    def emit(self, record):
        msg = record.getMessage().strip()
        if " finished in " in msg or " FAILED after " in msg:
            self._report(None, msg)


def _configure_datasail_threads():
    """Apply the worker budget to DataSAIL's documented solver helper default.

    DataSAIL 1.4 exposes num_threads on solve(), but its C1/C2 callers omit it
    and inherit fourteen threads. All imported references share this function;
    changing its default here affects only this isolated compute process.
    """
    from datasail.solver.utils import solve
    threads = max(1, int(os.environ.get("WEBAPP_WORKER_THREADS", "2")))
    positional = [p.name for p in inspect.signature(solve).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    defaults = list(solve.__defaults__ or ())
    index = positional.index("num_threads") - (len(positional) - len(defaults))
    if index < 0:
        raise RuntimeError("The installed DataSAIL solver has no configurable thread default")
    defaults[index] = threads
    solve.__defaults__ = tuple(defaults)


def _matrix_callables(kwargs):
    """Use DataSAIL's supported callable form for custom matrix inputs.

    DataSAIL 1.4 + SciPy 1.17 tests distance against a set of retired metric
    names. Tuples containing lists/arrays are unhashable in that check. A
    zero-argument callable is an official MATRIX_INPUT and returns the same
    names/matrix without copying or changing the values.
    """
    kwargs = dict(kwargs)
    for key in ("e_dist", "f_dist", "e_sim", "f_sim"):
        value = kwargs.get(key)
        if isinstance(value, tuple) and len(value) == 2:
            kwargs[key] = lambda value=value: value
    return kwargs


def _configure_datasail_runtime():
    _configure_datasail_threads()
    from PALM.benchmarks.datasail_pipeline import splitting
    original = splitting.datasail
    if getattr(original, "_webapp_matrix_inputs", False):
        return

    @wraps(original)
    def compatible(*args, **kwargs):
        return original(*args, **_matrix_callables(kwargs))

    compatible._webapp_matrix_inputs = True
    splitting.datasail = compatible


def _task_run(task, report):
    from PALM.benchmarks.datasail_pipeline.pipeline import run_pipeline
    from PALM.webapp import methods  # noqa: F401  (registers "hypergraph" and "lowrank")
    from PALM.webapp.reduction import install_pipeline_cache
    install_pipeline_cache()
    _configure_datasail_runtime()

    split_log = logging.getLogger("PALM.benchmarks.datasail_pipeline.splitting")
    split_log.addHandler(_ForwardTimings(report))
    if split_log.getEffectiveLevel() > logging.INFO:
        split_log.setLevel(logging.INFO)
    Path(task["output_dir"]).mkdir(parents=True, exist_ok=True)
    cfg = build_config(task["request"], task["input_file"], task["output_dir"])
    run_pipeline(cfg, progress_callback=lambda pct, msg: report(pct, msg))


def _task_map(task, report):
    from PALM.webapp import embedding
    embedding.compute_map(task["file_path"], task["column"], task["type"],
                          task["feature_sets"], progress=lambda msg: report(None, msg))


TASKS = {"run": _task_run, "map": _task_map}


def _child_main():
    # Keep the protocol channel private: fd 1 becomes stderr for everything
    # else (print(), logging, C extensions), the original stdout is ours.
    channel = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj):
        channel.write(json.dumps(obj, default=str) + "\n")
        channel.flush()

    try:
        task = json.loads(sys.stdin.readline())
        TASKS[task["kind"]](task, lambda pct, msg: send({"progress": [pct, msg]}))
        send({"done": True})
    except BaseException as exc:      # report everything, incl. KeyboardInterrupt
        traceback.print_exc()           # -> stderr -> the job's worker.log
        send({"error": {"type": type(exc).__name__,
                        "args": [str(a) for a in exc.args],
                        "traceback": traceback.format_exc()}})
        sys.exit(1)


# ── parent side ───────────────────────────────────────────────────────────

def _kill(proc: subprocess.Popen, grace: float = 5.0):
    """Terminate the child's whole process group (SIGTERM, then SIGKILL)."""
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, grace)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _rebuild(err: dict) -> BaseException:
    cls = _KNOWN_ERRORS.get(err.get("type"), RuntimeError)
    args = err.get("args") or [err.get("type", "error")]
    if cls is RuntimeError and err.get("type") not in _KNOWN_ERRORS:
        args = [f"{err.get('type')}: {' '.join(args)}"]
    exc = cls(*args)
    exc.worker_traceback = err.get("traceback")
    return exc


def _run_task(task: dict, on_progress: Callable[[Optional[int], str], None],
             log_path: Optional[Path] = None, tick: float = 0.5,
             env: Optional[dict] = None) -> None:
    """Run ``task`` in a child process; return when it is done.

    ``on_progress(pct, msg)`` gets every progress message, and is also called
    as ``on_progress(None, "")`` about every ``tick`` seconds; if it raises
    (e.g. the job was cancelled) the child is killed and the exception
    propagates. A failure in the child is re-raised here.
    """
    child_env = dict(os.environ if env is None else env)
    threads = str(max(1, int(child_env.get("WEBAPP_WORKER_THREADS", "2"))))
    for name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        child_env[name] = threads
    child_env.setdefault("CUDA_VISIBLE_DEVICES", "")
    child_env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    timeout = float(child_env.get("WEBAPP_MAP_TIMEOUT_SECONDS" if task["kind"] == "map"
                                  else "WEBAPP_RUN_TIMEOUT_SECONDS", "3600"))
    started = time.monotonic()
    log = open(log_path, "ab") if log_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "PALM.webapp.worker"], cwd=str(ROOT_DIR),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            start_new_session=True, env=child_env)
    finally:
        if log_path:
            log.close()                     # the child holds its own descriptor
    sel = None
    try:
        proc.stdin.write((json.dumps(task) + "\n").encode())
        proc.stdin.close()
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        buf = b""
        last_tick = time.monotonic()
        while True:
            if timeout > 0 and time.monotonic() - started > timeout:
                raise WorkerError(f"This task reached the {timeout:g}-second time limit. "
                                  "Try fewer methods or a smaller dataset.")
            if sel.select(timeout=tick):
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:               # child closed stdout: it exited
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    msg = json.loads(line)
                    if "progress" in msg:
                        on_progress(*msg["progress"])
                    elif "done" in msg:
                        return
                    elif "error" in msg:
                        raise _rebuild(msg["error"])
            if time.monotonic() - last_tick >= tick:
                last_tick = time.monotonic()
                on_progress(None, "")
        code = proc.wait()
        raise WorkerError(
            f"The worker process exited unexpectedly (exit code {code}"
            f"{', killed by signal ' + str(-code) if code < 0 else ''}"
            f"{', possibly out of memory' if code == -signal.SIGKILL else ''}).")
    finally:
        _kill(proc)
        if sel is not None:
            sel.close()
        proc.stdout.close()


def run_task(task: dict, on_progress: Callable[[Optional[int], str], None],
             log_path: Optional[Path] = None, tick: float = 0.5,
             env: Optional[dict] = None) -> None:
    """Runs and maps share one compute budget; waiting jobs remain cancellable."""
    announced = False
    while not _WORK_SLOTS.acquire(timeout=tick):
        on_progress(None, "Waiting for another job to finish…" if not announced else "")
        announced = True
    try:
        return _run_task(task, on_progress, log_path=log_path, tick=tick, env=env)
    finally:
        _WORK_SLOTS.release()


if __name__ == "__main__":
    _child_main()
