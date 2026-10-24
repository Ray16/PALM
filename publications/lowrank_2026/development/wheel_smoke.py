"""Verify the built wheel in an isolated target, including exact source bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

p = argparse.ArgumentParser()
p.add_argument("--target", type=Path, required=True)
p.add_argument("--wheel", type=Path, required=True)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.target))
import lowrank
import numpy as np
from importlib.metadata import version
if not Path(lowrank.__file__).is_relative_to(a.target): raise RuntimeError("wheel import shadowed")
with zipfile.ZipFile(a.wheel) as z:
    files = [name for name in z.namelist() if name.startswith('lowrank/') and name.endswith('.py')]
    for name in files:
        if z.read(name) != (a.source / name).read_bytes(): raise RuntimeError("wheel source mismatch: " + name)
    if any(name.split('/')[0] in ('datasets', 'features', 'splitters', 'eval') for name in z.namelist()):
        raise RuntimeError("generic package namespace collision")
rng = np.random.default_rng(12)
X = (rng.random((120, 64)) < .25).astype(np.float32)
data = dict(enumerate(X))
spec = lowrank.SplitSpec([8, 1, 1], ['train', 'val', 'test'], seed=0)
results = [lowrank.split('lowrank', data, spec, rank='auto', rank_min=8, rank_max=64,
                        leakage_diagnostic='sampled'),
           lowrank.split('lowrank', data, spec, rank=16, groups={i:i//3 for i in data},
                        group_policy='strict', leakage_diagnostic='none'),
           lowrank.split('lowrank', data, spec, rank=16, local_weight=.25,
                        local_threshold=.2, leakage_diagnostic='none')]
for result in results:
    if result.diagnostics['split_counts'] != {'train':96, 'val':12, 'test':12}:
        raise RuntimeError("wheel exact-count failure")
    json.dumps(result.to_json(), allow_nan=False)
if not results[1].diagnostics['all_groups_preserved']: raise RuntimeError("wheel group-preservation failure")
record = dict(version=version('lowrank-split'), wheel_sha256=hashlib.sha256(a.wheel.read_bytes()).hexdigest(),
              imported_from=str(lowrank.__file__), source_files_matched=len(files),
              adaptive=True, strict_groups=True, local_objective=True, exact_counts=True, namespace_check=True)
a.output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
