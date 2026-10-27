"""Exercise the deliverable wheel without importing the source checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

p = argparse.ArgumentParser()
p.add_argument("--target", type=Path, required=True)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--wheel", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.target))
import numpy as np
import lowrank
from importlib.metadata import version

assert Path(lowrank.__file__).is_relative_to(a.target)
with zipfile.ZipFile(a.wheel) as z:
    files = [
        name
        for name in z.namelist()
        if name.startswith("lowrank/") and name.endswith(".py")
    ]
    for name in files:
        assert z.read(name) == (a.source / name).read_bytes(), name
    assert not any(
        name.split("/")[0] in ["datasets", "features", "splitters", "eval"]
        for name in z.namelist()
    )
X = (np.random.default_rng(12).random((120, 128)) < 0.25).astype(np.float32)
data = dict(enumerate(X))
spec = lowrank.SplitSpec([8, 1, 1], ["train", "val", "test"], seed=0)
results = [
    lowrank.split(
        "lowrank",
        data,
        spec,
        rank="auto",
        rank_min=8,
        rank_max=64,
        rank_check="cut_error",
        on_rank_failure="return",
        leakage_diagnostic="sampled",
    ),
    lowrank.split(
        "lowrank",
        data,
        spec,
        rank=16,
        groups={i: i // 3 for i in data},
        group_policy="strict",
        leakage_diagnostic="none",
    ),
    lowrank.split(
        "lowrank",
        data,
        spec,
        rank=16,
        fold_weights=[[0, 1, 2], [1, 0, 0.25], [2, 0.25, 0]],
        leakage_diagnostic="exact",
    ),
]
for result in results:
    assert result.diagnostics["split_counts"] == {"train": 96, "val": 12, "test": 12}
    json.dumps(result.to_json(), allow_nan=False)
assert results[1].diagnostics["all_groups_preserved"]
records = [{"a": str(i % 20), "b": str(i % 15)} for i in range(120)]
nd = lowrank.split(
    "lowrank_nd", (records, {"a": {}, "b": {}}), spec, score_exact_leakage=False
)
assert nd.diagnostics["factorization"]["storage"] == "CSR"
assert nd.diagnostics["split_counts"] == {"train": 96, "val": 12, "test": 12}
_, matched = lowrank.chronological_reference(
    list(range(120)), dict(enumerate(range(120))), [95, 107]
)
assert matched.splits == [96, 12, 12] and matched.epsilon == 0
try:
    lowrank.nystrom_features(np.full((4, 8), 1.5), metric="tanimoto")
except ValueError:
    pass
else:
    raise AssertionError("count-valued Tanimoto accepted")
record = dict(
    version=version("lowrank-split"),
    wheel_sha256=hashlib.sha256(a.wheel.read_bytes()).hexdigest(),
    imported_from=str(lowrank.__file__),
    source_files_matched=len(files),
    namespace_check=True,
    cut_adaptation=True,
    strict_groups=True,
    weighted_policy=True,
    sparse_nd=True,
    chronological_helper=True,
    invalid_input_rejected=True,
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
