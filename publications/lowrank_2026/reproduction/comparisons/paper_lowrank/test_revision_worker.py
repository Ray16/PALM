"""Regression for a signal arriving while the worker blocks in Popen.wait."""
from pathlib import Path
import json
import subprocess
import sys
import time

import psutil


def test_sigterm_unwinds_wait_before_reaping_child_and_releasing_claim(tmp_path):
    script = Path(__file__).with_name("revision_worker_v8.py")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps(dict(source_sha256={}, cwd=str(tmp_path), jobs=[dict(
        id="shutdown_probe", study="scaling", kind="cpu", output=str(tmp_path / "output"), n=1,
        argv=[sys.executable, "-c", "import time; time.sleep(600)"])])))
    child = None
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(script), "--manifest", str(manifest),
                                    "--kind", "cpu", "--id", "test"], stdout=log, stderr=subprocess.STDOUT)
    try:
        owner_path = tmp_path / "claims/shutdown_probe/owner.json"
        deadline = time.monotonic() + 5
        blocked_in_wait = False
        while time.monotonic() < deadline:
            if owner_path.exists():
                owner = json.loads(owner_path.read_text())
                child = owner.get("child_pid")
                wchan = Path(f"/proc/{process.pid}/wchan").read_text().strip()
                if child and "wait" in wchan:
                    blocked_in_wait = True
                    break
            time.sleep(.02)
        assert blocked_in_wait, "did not observe the worker blocked waiting for its native child"
        process.terminate()
        assert process.wait(timeout=5) == 143
        assert not owner_path.parent.exists()
        assert not psutil.pid_exists(child)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child and psutil.pid_exists(child):
            native = psutil.Process(child)
            if native.status() != psutil.STATUS_ZOMBIE:
                native.kill()
