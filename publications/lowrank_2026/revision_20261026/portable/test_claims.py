"""Ownership and open-reader behavior on the actual shared filesystem."""

import importlib.util
import json
from pathlib import Path

import pytest


def adapter():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("portable_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retirement_preserves_open_metadata_and_allows_new_owner(tmp_path):
    module = adapter()
    claim = tmp_path / "claims/job"
    claim.mkdir(parents=True)
    (claim / "owner.json").write_text(json.dumps(dict(token="first")))
    with (claim / "owner.json").open() as reader:
        retired = module.retire_claim(claim, "first", tmp_path / "retired")
        assert not claim.exists()
        assert json.loads(reader.read())["token"] == "first"
        claim.mkdir()
        (claim / "owner.json").write_text(json.dumps(dict(token="second")))
        assert json.loads((retired / "owner.json").read_text())["token"] == "first"
        assert json.loads((claim / "owner.json").read_text())["token"] == "second"


def test_changed_owner_is_never_released(tmp_path):
    module = adapter()
    claim = tmp_path / "job"
    claim.mkdir()
    (claim / "owner.json").write_text(json.dumps(dict(token="other")))
    with pytest.raises(RuntimeError, match="owner changed"):
        module.retire_claim(claim, "ours", tmp_path / "retired")
    assert claim.exists()
    assert not (tmp_path / "retired").exists()
