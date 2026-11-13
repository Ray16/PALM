"""Webapp security settings and request sanitisation.

Run from the directory containing PALM/:  python -m pytest PALM/webapp/tests
"""
import os

import pytest


def test_security_limits_configured():
    from PALM.webapp.app import _allowed_origins, MAX_UPLOAD_BYTES, RATE_LIMIT_MAX
    assert _allowed_origins is not None
    assert MAX_UPLOAD_BYTES > 0
    assert RATE_LIMIT_MAX > 0


def test_cross_origin_requests_closed_by_default():
    if "PALM_CORS_ORIGINS" in os.environ:
        pytest.skip("PALM_CORS_ORIGINS is set in this environment")
    from PALM.webapp.app import _allowed_origins
    assert _allowed_origins == []          # the app serves its own frontend (same origin)


def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import PALM.webapp.app as app_mod
    from PALM.webapp import jobs
    store = jobs.Store(tmp_path, lambda run, progress: None)      # queue only, no execution
    monkeypatch.setattr(app_mod, "store", store)
    client = TestClient(app_mod.app)
    return client, store, client.post("/api/demo/esol").json()["dataset_id"]


@pytest.mark.parametrize("hostile", ["../../../tmp/pwned", "/tmp/abs_override", "..", "a/b\\c", ""])
def test_names_from_requests_cannot_escape_the_output_dir(tmp_path, monkeypatch, hostile):
    client, store, ds = _client(tmp_path, monkeypatch)
    body = {"dataset_name": hostile,
            "e1": {"name": hostile, "type": "molecule", "column": "smiles"},
            "splitting": {"techniques": ["R"], "ratio": "80/20"}}
    r = client.post(f"/api/run/{ds}", json=body)
    assert r.status_code == 200, r.text
    req = store.get_run(r.json()["run_id"])["request"]
    for name in (req["dataset_name"], req["e1"]["name"]):
        assert name and "/" not in name and "\\" not in name and name not in (".", "..")
    # the names the worker will use are accepted by the pipeline's own validation
    from PALM.benchmarks.datasail_pipeline.config import check_path_component
    check_path_component(req["dataset_name"], "dataset_name")
    check_path_component(req["e1"]["name"], "entity name")


def test_identical_entity_names_are_made_distinct(tmp_path, monkeypatch):
    client, store, ds = _client(tmp_path, monkeypatch)
    body = {"e1": {"name": "x", "type": "molecule", "column": "smiles"},
            "e2": {"name": "x", "type": "molecule", "column": "smiles"},
            "splitting": {"techniques": ["R"], "ratio": "80/20"}}
    r = client.post(f"/api/run/{ds}", json=body)
    assert r.status_code == 200, r.text
    req = store.get_run(r.json()["run_id"])["request"]
    assert req["e1"]["name"] != req["e2"]["name"]
