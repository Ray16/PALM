"""Independent retention, group, score and provenance audit of the completed grid."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--package", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.package))
from lowrank.grouped import must_link_groups
from lowrank.splitters.common.balanced_assignment import target_sizes

manifest = json.loads((a.run / "manifest.json").read_text())
base = Path(__file__).resolve().parents[3]
source_ref = "95e43be"
source_prefix = "publications/lowrank_2026/package/"
changed = []
used = {job["implementation"] for job in manifest["jobs"]}
for implementation in used:
    ref = source_ref if implementation == "development" else "9f7c34e"
    current_root = a.package if implementation == "development" else Path(manifest["implementations"][implementation])
    for rel, expected in manifest["source_files"][implementation].items():
        original = subprocess.check_output(["git", "show", ref + ':' + source_prefix + rel], cwd=base)
        if hashlib.sha256(original).hexdigest() != expected: raise RuntimeError("benchmark commit mismatch: " + rel)
        current = (current_root / rel).read_bytes()
        if hashlib.sha256(current).hexdigest() != expected:
            if implementation != "development" or rel != "lowrank/neighbours.py":
                raise RuntimeError("unreviewed source change: " + rel)
            before, after = ast.parse(original), ast.parse(current)
            for tree in (before, after):
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == "approximate_tanimoto_neighbours":
                        node.body = [ast.Pass()]
            if ast.dump(before) != ast.dump(after): raise RuntimeError("change outside ANN repair function")
            # The repaired ANN routine is not executed by any measured grid setting.
            for job in manifest["jobs"]:
                params = job["params"]
                if params.get("must_link_backend", "exact") != "exact" or params.get("local_backend", "tiled") != "tiled":
                    raise RuntimeError("grid could execute changed ANN routine")
            changed.append(dict(file=rel, function="approximate_tanimoto_neighbours",
                                grid_impact="none: all grid paths use exact neighbours",
                                validation="separate native ANN audit and regression tests"))

rows, by_dataset = [], {}
for job in manifest["jobs"]:
    row = json.loads((a.run / "rows" / (job["name"] + '.json')).read_text())
    if row["status"] != "ok": raise RuntimeError("non-success row: " + job["name"])
    path = a.run / "rows" / (job["name"] + '.npy')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["assignment_sha256"]: raise RuntimeError("assignment drift")
    labels = np.load(path)
    bundle = np.load(a.run / "cache" / (job["dataset"] + '.npz'))
    X, ids = bundle["X"], bundle["ids"]
    if len(np.unique(ids)) != len(ids) or labels.shape != (len(ids),): raise RuntimeError("row identity/retention error")
    if not np.array_equal(np.bincount(labels, minlength=3), target_sizes(len(ids), [8, 1, 1])):
        raise RuntimeError("fold count error")
    if hashlib.sha256(X.tobytes()).hexdigest() != row["feature_sha256"]: raise RuntimeError("feature drift")
    if row["source_files"] != manifest["source_files"][job["implementation"]]: raise RuntimeError("row source drift")
    if job["setting"] == "new_combined_strict":
        if job["dataset"] not in by_dataset:
            cap = int(.5 * target_sizes(len(X), [8, 1, 1]).min())
            by_dataset[job["dataset"]] = must_link_groups(X, "tanimoto", .4, cap, k=32)
        gid = by_dataset[job["dataset"]]
        low = np.full(int(gid.max()) + 1, 3); high = np.full(len(low), -1)
        np.minimum.at(low, gid, labels); np.maximum.at(high, gid, labels)
        if not np.array_equal(low, high): raise RuntimeError("strict group cut in assignment artifact")
    rows.append(row)

# Independently compute dense reference scores on the smallest full dataset.
small = "moleculenet_freesolv"
bundle = np.load(a.run / "cache" / (small + '.npz'))
X = bundle["X"].astype(np.float64)
dot = X @ X.T
union = X.sum(1)[:, None] + X.sum(1)[None, :] - dot
kernel = np.divide(dot, union, out=np.zeros_like(dot), where=union > 0)
np.fill_diagonal(kernel, 0)
maximum_score_error = 0.
for row in [r for r in rows if r["dataset"] == small]:
    lab = np.load(a.run / "rows" / (row["name"] + '.npy'))
    ratio = kernel[lab[:, None] != lab[None, :]].sum() / kernel.sum()
    near = kernel[lab == 2][:, lab == 0].max(1)
    ramp = np.maximum(0, (near - .4) / .6).mean()
    error = max(abs(ratio - row["scores"]["global_ratio"]), abs(ramp - row["scores"]["local_ramp_04"]))
    if error > 1e-6: raise RuntimeError("common scorer disagrees with independent NumPy dense reference")
    maximum_score_error = max(maximum_score_error, float(error))

sampled = [r for r in rows if r["setting"] == "new_auto_sampled"]
coverage = sum(r["diagnostics"]["leakage_sampled"]["ci95"][0] <= r["scores"]["global_ratio"] <=
               r["diagnostics"]["leakage_sampled"]["ci95"][1] for r in sampled)
record = dict(expected_cases=len(manifest["jobs"]), completed=len(rows),
    retained_rows_and_exact_counts_verified=len(rows), strict_group_artifacts_verified=len(by_dataset) * 3,
    source_refs={impl: source_ref if impl == "development" else "9f7c34e" for impl in used},
    final_source_delta=changed, dense_reference_cases=sum(r["dataset"] == small for r in rows),
    maximum_dense_score_error=maximum_score_error,
    sampled_diagnostic_intervals_cover_exact=dict(covered=coverage, total=len(sampled),
        interpretation="descriptive finite-grid coverage, not an interval guarantee"))
(a.run / "pipeline_review.json").write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
