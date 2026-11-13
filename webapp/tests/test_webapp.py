"""Hypergraph/low-rank techniques, demo datasets and API wiring.

Run from data_splitting/ on CPU:  CUDA_VISIBLE_DEVICES= python -m pytest webapp/tests
"""
import json
from pathlib import Path

import numpy as np
import pytest

from PALM.benchmarks.datasail_pipeline.config import (
    EXTERNAL_TECHNIQUES, SplitContext, SplittingConfig, valid_techniques,
)
from PALM.webapp import methods

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_data"


def test_techniques_registered_for_1d_and_2d():
    for name in ("hypergraph", "lowrank"):
        assert name in EXTERNAL_TECHNIQUES
        assert name in valid_techniques(True) and name in valid_techniques(False)
    SplittingConfig(techniques=["R", "hypergraph", "lowrank"])        # validates


def test_builtin_names_cannot_be_overridden():
    from PALM.benchmarks.datasail_pipeline.config import register_technique
    with pytest.raises(ValueError):
        register_technique("C2", lambda ctx: {})


def _ctx_1d(n=120):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c * 4, 1, (n // 3, 8)) for c in range(3)])
    data = {f"m{i}": X[i] for i in range(len(X))}
    return SplitContext(True, data, None, None, {k: k for k in data}, None, None,
                        SplittingConfig(techniques=["lowrank"]))


@pytest.mark.parametrize("fn", [methods._split_lowrank, methods._split_hypergraph])
def test_1d_split_is_complete_and_balanced(fn):
    ctx = _ctx_1d()
    out = fn(ctx)
    assert set(out) == set(ctx.e1_data)
    frac_test = sum(v == "test" for v in out.values()) / len(out)
    assert abs(frac_test - 0.2) <= 0.05


def test_pair_assignment_majority_vote():
    pairs = [("a", "x"), ("a", "x"), ("a", "x"), ("b", "y")]
    ctx = SplitContext(False, {}, {}, pairs, {}, None, None,
                       SplittingConfig(techniques=["lowrank"]))
    out = methods._pair_assignment(ctx, {0: "train", 1: "test", 2: "train", 3: "test"})
    assert out == {("a", "x"): "train", ("b", "y"): "test"}


def test_demo_manifest_files_exist():
    demos = json.loads((DEMO_DIR / "demos.json").read_text())
    assert demos
    for d in demos:
        assert (DEMO_DIR / d["file"]).is_file(), d["file"]
        for t in d["techniques"]:
            assert t in valid_techniques(d.get("e2") is None), (d["id"], t)


def test_api_demo_runs_flow(tmp_path, monkeypatch):
    """Demo → two queued runs → results, split map, delete — through the API."""
    import time
    from fastapi.testclient import TestClient
    import PALM.webapp.app as app_mod
    from PALM.webapp import jobs
    store = jobs.Store(tmp_path, app_mod._execute_run)
    store.start_workers()
    monkeypatch.setattr(app_mod, "store", store)
    client = TestClient(app_mod.app, headers={"X-Client-Id": "test-browser"})

    assert client.get("/api/info").status_code == 200
    assert "esol" in [d["id"] for d in client.get("/api/demos").json()["demos"]]
    ds = client.post("/api/demo/esol").json()
    assert "smiles" in ds["columns"] and ds["demo"]["id"] == "esol"
    assert client.get(f"/api/datasets/{ds['dataset_id']}").json()["columns"] == ds["columns"]

    body = {"dataset_name": "esol demo!", "e1": {"name": "molecule", "type": "molecule",
            "column": "smiles", "feature_sets": ["rdkit_descriptors", "morgan_fingerprint"]},
            "splitting": {"techniques": ["R", "lowrank"], "ratio": "80/20"}}
    ids = [client.post(f"/api/run/{ds['dataset_id']}", json=body).json()["run_id"]
           for _ in range(2)]
    listed = client.get("/api/runs").json()["runs"]
    assert {r["id"] for r in listed} == set(ids)
    assert client.get("/api/runs", headers={"X-Client-Id": "someone-else"}).json()["runs"] == []

    t0 = time.time()
    while time.time() - t0 < 600:
        st = [client.get(f"/api/runs/{i}").json()["status"] for i in ids]
        if all(s in jobs.FINISHED for s in st):
            break
        time.sleep(0.5)
    assert st == ["completed", "completed"], st
    run = client.get(f"/api/runs/{ids[0]}").json()
    assert run["request"]["dataset_name"] == "esol_demo"
    cmp = client.get(f"/api/runs/{ids[0]}/metrics").json()["metrics"]["comparison_esol_demo"]
    assert set(cmp["status"]) == {"R", "lowrank"}
    files = client.get(f"/api/runs/{ids[0]}/files").json()
    assert any(t["name"] == "lowrank_esol_demo" for t in files["techniques"])
    sm = client.get(f"/api/runs/{ids[0]}/splitmap/e1").json()
    assert sm["status"] in ("missing", "completed")
    assert client.delete(f"/api/runs/{ids[1]}").status_code == 200
    assert client.get(f"/api/runs/{ids[1]}").status_code == 404
    assert client.get("/static/app.js").status_code == 200


def test_dataset_name_is_path_safe():
    from PALM.webapp.app import _safe_dataset_name
    assert _safe_dataset_name("../../etc/x", "abcdef123") == "etc_x"
    assert _safe_dataset_name("my data!", "abcdef123") == "my_data"
    assert _safe_dataset_name("", "abcdef123") == "abcdef12"


def test_technique_timings_parsed_from_log(monkeypatch, tmp_path):
    import PALM.webapp.app as app_mod
    from PALM.webapp import jobs
    store = jobs.Store(tmp_path, lambda run, progress: None)
    monkeypatch.setattr(app_mod, "store", store)
    run = {"id": "r1", "request": {"splitting": {"techniques": ["R", "C1e", "lowrank"]}},
           "log": ["Running lowrank...", "lowrank finished in 0.3s", "R finished in 0.1s",
                   "C1e finished in 12.0s", "All techniques finished in 12.1s",
                   "hypergraph finished in 3.0s"]}            # not requested: ignored
    assert app_mod._technique_timings(run) == {"lowrank": 0.3, "R": 0.1, "C1e": 12.0}


def test_run_request_rejects_bad_ratio_and_empty_methods(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import PALM.webapp.app as app_mod
    from PALM.webapp import jobs
    store = jobs.Store(tmp_path, lambda run, progress: None)
    monkeypatch.setattr(app_mod, "store", store)
    client = TestClient(app_mod.app)
    ds = client.post("/api/demo/esol").json()["dataset_id"]
    body = {"e1": {"name": "m", "type": "molecule", "column": "smiles"},
            "splitting": {"techniques": ["R"], "ratio": "60/40"}}
    r = client.post(f"/api/run/{ds}", json=body)
    assert r.status_code == 400 and "60/40" in r.json()["detail"]
    body["splitting"] = {"techniques": [], "ratio": "80/20"}
    assert client.post(f"/api/run/{ds}", json=body).status_code == 400
    assert store.list_runs(None) == []
