"""Run store: queueing, concurrency cap, cancel, persistence, restart recovery."""
import threading
import time

import pytest

from PALM.webapp import jobs


def _wait(pred, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_CONCURRENT_RUNS", 2)
    gate = threading.Event()
    running = []

    def executor(run, progress):
        running.append(run["id"])
        progress(10, "[1/5] Loading data")
        while not gate.is_set():
            progress(None, "")            # lets cancellation interrupt
            time.sleep(0.01)
        progress(90, "almost")
        if run["request"].get("fail"):
            raise RuntimeError("boom")

    s = jobs.Store(tmp_path, executor)
    s.gate, s.running = gate, running
    ds_id, _ = s.new_dataset_dir()
    s.add_dataset(ds_id, tmp_path / ds_id / "x.csv", "x.csv", "me", {"columns": []})
    s.ds_id = ds_id
    s.start_workers()
    yield s
    gate.set()


def test_queue_respects_concurrency_and_completes(store):
    runs = [store.submit(store.ds_id, {"i": i}, "me", "x") for i in range(4)]
    assert _wait(lambda: len(store.running) == 2)
    time.sleep(0.2)
    assert len(store.running) == 2                       # cap holds
    statuses = {r["id"]: r["status"] for r in store.list_runs("me")}
    assert sorted(statuses.values()) == ["queued", "queued", "running", "running"]
    store.gate.set()
    assert _wait(lambda: all(store.get_run(r["id"])["status"] == "completed" for r in runs))
    assert store.get_run(runs[0]["id"])["phase"] == "Complete"


def test_cancel_queued_and_running(store):
    a = store.submit(store.ds_id, {}, "me", "x")
    b = store.submit(store.ds_id, {}, "me", "x")
    c = store.submit(store.ds_id, {}, "me", "x")
    assert _wait(lambda: store.get_run(b["id"])["status"] == "running")
    store.cancel(c["id"])
    assert store.get_run(c["id"])["status"] == "cancelled"
    store.cancel(a["id"])
    assert _wait(lambda: store.get_run(a["id"])["status"] == "cancelled")
    with pytest.raises(ValueError):
        store.delete(b["id"])                             # still running
    store.delete(a["id"])
    assert store.get_run(a["id"]) is None


def test_failure_message_and_client_scoping(store):
    r = store.submit(store.ds_id, {"fail": True}, "other", "x")
    store.gate.set()
    assert _wait(lambda: store.get_run(r["id"])["status"] == "failed")
    assert store.get_run(r["id"])["error"] == "boom"
    assert [x["id"] for x in store.list_runs("other")] == [r["id"]]
    assert store.list_runs("me") == []


def test_restart_requeues_queued_and_interrupts_running(store, tmp_path):
    a = store.submit(store.ds_id, {}, "me", "x")
    b = store.submit(store.ds_id, {}, "me", "x")
    c = store.submit(store.ds_id, {}, "me", "x")
    assert _wait(lambda: len(store.running) == 2)
    for r in (a, b):                                      # flush throttled writes
        store._persist(store.runs[r["id"]], force=True)
    fresh = jobs.Store(tmp_path, lambda run, progress: None)
    fresh.load()
    st = {r["id"]: r["status"] for r in fresh.list_runs("me")}
    assert st[a["id"]] == st[b["id"]] == "interrupted" and st[c["id"]] == "queued"
    fresh.start_workers()
    assert _wait(lambda: fresh.get_run(c["id"])["status"] == "completed")
