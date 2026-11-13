"""Worker processes: concurrent runs keep the server responsive, cancel kills, errors propagate."""
import os
import threading
import time
from pathlib import Path

import pytest

from PALM.webapp import jobs, worker

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_data"
MOL_FEATURES = ["rdkit_descriptors", "composition", "physicochemical", "morgan_fingerprint"]


def _request(name, techniques):
    return {"dataset_name": name,
            "e1": {"name": "molecule", "type": "molecule", "column": "smiles",
                   "feature_sets": MOL_FEATURES},
            "e2": None,
            "splitting": {"techniques": techniques, "ratio": "80/20",
                          "allow_incomplete": False}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    import PALM.webapp.app as app_mod
    monkeypatch.setattr(jobs, "MAX_CONCURRENT_RUNS", 2)
    s = jobs.Store(tmp_path, app_mod._execute_run)
    monkeypatch.setattr(app_mod, "store", s)
    s.start_workers()
    return s


def _add_demo(store, demo):
    ds_id, d = store.new_dataset_dir()
    path = d / f"{demo}.csv"
    path.write_bytes((DEMO_DIR / f"{demo}.csv").read_bytes())
    store.add_dataset(ds_id, path, path.name, "me", {"columns": []})
    return ds_id


def _wait(pred, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _worker_children():
    """PIDs of live `python -m PALM.webapp.worker` processes started by this process."""
    me, out = os.getpid(), []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        state = stat.rsplit(")", 1)[1].split()[0]
        if ppid == me and b"webapp.worker" in cmd and state != "Z":
            out.append(int(pid))
    return out


def test_concurrent_runs_keep_parent_responsive(store):
    """The scenario that froze the in-thread server: two runs splitting at once.

    With the pipeline on server threads, ESOL + BBBP with hypergraph + lowrank
    starved the interpreter within ~15 s (a heartbeat thread stopped). Now the
    parent's threads must keep running on time while both runs complete.
    """
    a = store.submit(_add_demo(store, "esol"), _request("esol", ["R", "hypergraph", "lowrank"]),
                     "me", "esol")
    b = store.submit(_add_demo(store, "bbbp"), _request("bbbp", ["R", "hypergraph", "lowrank"]),
                     "me", "bbbp")
    worst, last = 0.0, time.monotonic()
    t0 = time.time()
    while time.time() - t0 < 600:
        time.sleep(0.05)
        now = time.monotonic()
        worst, last = max(worst, now - last), now
        st = [store.get_run(r["id"])["status"] for r in (a, b)]
        if all(s in jobs.FINISHED for s in st):
            break
    assert st == ["completed", "completed"], [store.get_run(r["id"])["error"] for r in (a, b)]
    assert worst < 1.0, f"parent thread stalled for {worst:.1f}s"
    for r in (a, b):
        out = store.output_dir(store.get_run(r["id"]))
        name = r["request"]["dataset_name"]
        for tech in ("R", "hypergraph", "lowrank"):
            assert (out / "split_result" / f"datasail_split_{tech}_{name}.csv").is_file()


def test_cancel_kills_running_worker(store):
    r = store.submit(_add_demo(store, "bbbp"), _request("bbbp", ["C1e", "lowrank"]), "me", "bbbp")
    assert _wait(lambda: store.get_run(r["id"])["status"] == "running" and _worker_children(), 60)
    time.sleep(3)                                     # let it get into real work
    t = time.time()
    store.cancel(r["id"])
    assert _wait(lambda: store.get_run(r["id"])["status"] == "cancelled", 15)
    assert time.time() - t < 15
    assert _wait(lambda: not _worker_children(), 15), _worker_children()


def test_child_errors_keep_their_type(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("smiles,y\nCCO,1\nCCC,2\n")
    with pytest.raises(KeyError):
        worker.run_task({"kind": "map", "file_path": str(csv), "column": "nope",
                         "type": "molecule", "feature_sets": []}, lambda p, m: None)
    with pytest.raises(ValueError, match="only 2 unique"):
        worker.run_task({"kind": "map", "file_path": str(csv), "column": "smiles",
                         "type": "molecule", "feature_sets": []}, lambda p, m: None)


def test_progress_exception_kills_child(tmp_path):
    """on_progress raising (how cancellation works) stops and reaps the child."""
    calls = []

    def progress(pct, msg):
        calls.append(msg)
        raise jobs.RunCancelled()

    csv = tmp_path / "x.csv"
    csv.write_text("smiles,y\n" + "".join(f"{'C' * (i + 1)}O,{i}\n" for i in range(30)))
    with pytest.raises(jobs.RunCancelled):
        worker.run_task({"kind": "map", "file_path": str(csv), "column": "smiles",
                         "type": "molecule", "feature_sets": []}, progress)
    assert calls and not _worker_children()


def test_crashed_child_is_reported(tmp_path, monkeypatch):
    """A child killed from outside (e.g. the OOM killer) fails the job cleanly."""
    import signal
    started = threading.Event()

    def progress(pct, msg):
        if not started.is_set():
            started.set()
            for pid in _worker_children():
                os.kill(pid, signal.SIGKILL)

    csv = tmp_path / "x.csv"
    csv.write_text("smiles,y\n" + "".join(f"{'C' * (i + 1)}O,{i}\n" for i in range(30)))
    with pytest.raises(worker.WorkerError, match="signal 9"):
        worker.run_task({"kind": "map", "file_path": str(csv), "column": "smiles",
                         "type": "molecule", "feature_sets": []}, progress)
