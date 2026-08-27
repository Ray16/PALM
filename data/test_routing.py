"""Tests for the featurization router (`PALM.data.routing`).

Run: `python -m pytest PALM/data/test_routing.py` (from the PALM parent), or
`python -m PALM.data.test_routing` for a plain-assert smoke run.
"""

from __future__ import annotations

from .routing import CONF_THRESHOLD, Routing, detect_entity_type, route


def test_detect_smiles():
    et, conf, _ = detect_entity_type(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC"])
    assert et == "molecule" and conf >= 0.7


def test_detect_formula():
    et, conf, _ = detect_entity_type(["Fe2O3", "LiMn2O4", "CaTiO3", "NaCl", "Al2O3"])
    assert et == "material" and conf >= 0.7


def test_detect_protein():
    seqs = ["MKTFFVAGLLLGSTQAAGVYLDGEECRWLKQ", "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGR"]
    et, _, _ = detect_entity_type(seqs)
    assert et == "biomolecule"


def test_detect_gene():
    et, _, _ = detect_entity_type(["ATGCGTACGTTAGCGATCGATCGTT", "ACGTACGTACGTACGTACGTACGT"])
    assert et == "gene"


def test_garbage_is_unknown_or_lowconf():
    et, conf, _ = detect_entity_type(["???", "42", "n/a", "   ", "%%%"])
    assert et == "unknown" or conf < CONF_THRESHOLD


def test_route_precedence_override_wins():
    ids = {0: "CCO", 1: "c1ccccc1"}
    r = route("demo", ids, entity_type="molecule",
              override={"feature_set": "maccs", "reason": "unit-test"}, log=False)
    assert isinstance(r, Routing) and r.source == "override" and r.feature_set == "maccs"


def test_route_default_when_no_heuristic():
    r = route("does_not_exist_xyz", entity_type="material",
              heuristics={}, log=False)          # empty heuristics -> type default
    assert r.source == "default" and r.feature_set == "magpie"


def test_route_strict_raises_on_low_confidence():
    try:
        route("g", {0: "???", 1: "42", 2: "n/a"}, strict=True, log=False)
    except ValueError:
        return
    raise AssertionError("strict route should raise on low-confidence detection")


def test_route_flags_low_confidence_nonstrict():
    r = route("g", {0: "???", 1: "42", 2: "n/a"}, heuristics={}, log=False)
    assert "VERIFY" in r.reason


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n== {len(fns)} routing tests passed")
