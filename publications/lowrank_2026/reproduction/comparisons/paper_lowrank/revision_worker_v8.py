"""Atomic, resumable CPU/GPU case worker for the publication revision.

The launcher supplies disjoint affinity/thread caps and GPU reservations. Native
producers stay unchanged, so completed rows keep their original provenance.
Claims carry worker and child process-start identities. Final artifacts, not
exit status alone, establish completion. Two attempts per case, three consecutive
actual failures per worker. Cooldown/contention does not consume an attempt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


def atomic(path, data):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def start_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (FileNotFoundError, ProcessLookupError):
        return None


def complete(job):
    root = Path(job["output"])
    if job["study"] == "benchmark_cell":
        path = root / job['filename']
        if not path.exists(): return False
        row = json.loads(path.read_text())
        return (row.get('status') in ('ok','unavailable') and
                (row['status'] != 'ok' or path.with_suffix('.npy').exists()))
    if job["study"] == "benchmark":
        rows = list(root.glob("*__eps*__s*.json"))
        if len(rows) != 44:
            return False
        for path in rows:
            row = json.loads(path.read_text())
            if row.get("status") == "ok" and not path.with_suffix(".npy").exists():
                return False
            if row.get("status") not in ("ok", "unavailable"):
                return False
        return True
    path = root / "result.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get("status") == "unavailable":
        return job["study"] == "umap" and record.get("n", 0) * (record.get("n", 0) - 1) // 2 * 8 > 16 * 1024**3
    if record.get("status") != "ok":
        return False
    if job["study"] == "umap":
        return len(record.get("rows", [])) == 21 and all((root / f"{m}__fold{f}.npy").exists()
            for m in ("umap_cluster", "plain", "combined") for f in range(7))
    if job["study"] == "datasail_matched_v2":
        return (len(record.get("rows", [])) == 14 and
                all((root / f"{m}__eps{eps}__s{s}.npy").exists() for eps in (.01,.05) for m in ('datasail','plain','combined') for s in ([0] if m=='datasail' else [0,1,2])))
    if job["study"] == "runtime":
        return len(record.get("rows", [])) == 7 and (root / "datasail.npy").exists()
    if job["study"] == "scaling":
        return record.get("n") == job["n"] and (root / "assignment_codes.npy").exists()
    return False


class TerminationRequested(BaseException):
    pass


def run(manifest_path, kind, worker_id):
    manifest = json.loads(manifest_path.read_text())
    queue = manifest_path.parent
    for folder in ("claims", "attempts", "done", "logs", "workers", "partial_archive"):
        (queue / folder).mkdir(exist_ok=True)
    for path, digest in manifest["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("producer changed since queue was prepared: " + path)
    owner = dict(host=socket.gethostname(), pid=os.getpid(), start_ticks=start_ticks(os.getpid()),
        affinity=sorted(os.sched_getaffinity(0)), worker_id=worker_id, kind=kind)
    atomic(queue / "workers" / (worker_id + ".json"), dict(owner, status="running"))
    failures = 0
    child = None
    claim = None

    def terminate(signum, _frame):
        nonlocal child
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if claim is not None:
            shutil.rmtree(claim)
        atomic(queue / "workers" / (worker_id + ".json"), dict(owner, status="terminated"))
        raise SystemExit(128 + signum)

    def request_termination(signum, _frame):
        # Never call Popen.wait from a signal handler: the interrupted wait may
        # hold its private lock. Unwind it first, then clean up outside the handler.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise TerminationRequested(signum)

    signal.signal(signal.SIGTERM, request_termination)
    signal.signal(signal.SIGINT, request_termination)
    try:
        for job in manifest["jobs"]:
            if job["kind"] != kind:
                continue
            if complete(job):
                atomic(queue / "done" / (job["id"] + ".json"), dict(job_id=job["id"], verified_at=time.time(), evidence=job["output"]))
                continue
            attempt_path = queue / "attempts" / (job["id"] + ".json")
            history = json.loads(attempt_path.read_text()) if attempt_path.exists() else dict(attempts=[])
            if len(history["attempts"]) >= 2:
                continue
            if history["attempts"] and time.time() - history["attempts"][-1]["finished"] < 60:
                continue
            claim = queue / "claims" / job["id"]
            try:
                claim.mkdir()
            except FileExistsError:
                claim = None
                continue  # ownership is resolved by the coordinator, never by age
            atomic(claim / "owner.json", dict(owner, job_id=job["id"]))
            if complete(job):
                shutil.rmtree(claim)
                claim = None
                continue
            # Preserve partial chunks; completed benchmark rows remain directly reusable.
            partial = Path(job["output"]) / "progress.json"
            if partial.exists():
                shutil.copy2(partial, queue / "partial_archive" / (job["id"] + f".{time.time_ns()}.json"))
            failed = Path(job["output"]) / "result.json"
            if failed.exists() and json.loads(failed.read_text()).get("status") == "failed":
                shutil.move(failed, queue / "partial_archive" / (job["id"] + f".failed.{time.time_ns()}.json"))
            attempt = len(history["attempts"]) + 1
            log = queue / "logs" / (job["id"] + f".attempt{attempt}.log")
            started = time.time()
            with log.open("a") as out:
                child = subprocess.Popen(job["argv"], cwd=manifest["cwd"], stdout=out, stderr=subprocess.STDOUT)
                atomic(claim / "owner.json", dict(owner, job_id=job["id"],
                    child_pid=child.pid, child_start_ticks=start_ticks(child.pid), argv=job["argv"], log=str(log)))
                print(worker_id, "START", job["id"], child.pid, flush=True)
                exit_code = child.wait()
            ok = complete(job)
            history["attempts"].append(dict(started=started, finished=time.time(), exit_code=exit_code,
                                          output_verified=ok, worker=owner, log=str(log)))
            atomic(attempt_path, history)
            shutil.rmtree(claim)
            claim = None
            child = None
            if ok:
                failures = 0
                atomic(queue / "done" / (job["id"] + ".json"), dict(job_id=job["id"], verified_at=time.time(), evidence=job["output"]))
            else:
                failures += 1
            print(worker_id, "FINISH", job["id"], "verified", ok, flush=True)
            if failures >= 3:
                break
    except TerminationRequested as event:
        terminate(event.args[0], None)

    # Exit with the actual remaining state, distinguishing live peer claims.
    remaining = [j["id"] for j in manifest["jobs"] if j["kind"] == kind and not complete(j)]
    held = [j for j in remaining if (queue / "claims" / j).exists()]
    atomic(queue / "workers" / (worker_id + ".json"), dict(owner, status="exited",
        remaining=remaining, held_by_peers=held, consecutive_failures=failures))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    p.add_argument("--id", required=True)
    a = p.parse_args()
    run(a.manifest, a.kind, a.id)
