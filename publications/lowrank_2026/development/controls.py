"""Complete the power-only/group-only controls specified in DEVELOPMENT.md.

The original 180-case grid is preserved. This additional grid is declared before
execution, with the same fixed p=3 and tau=.4 choices. No outcomes tune them.
"""
import argparse
from pathlib import Path
import benchmark

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--cache-source", type=Path, required=True)
p.add_argument("--release", type=Path, required=True)
p.add_argument("--development", type=Path, required=True)
p.add_argument("--cpus", nargs='+', default=["50,51", "52,53", "54,55"])
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
cache = a.output / "cache"
if not cache.exists(): cache.symlink_to(a.cache_source.resolve(), target_is_directory=True)
benchmark.SETTINGS = {
    "release_power3": ("release", {"kernel_power": 3}),
    "release_groups": ("release", {"must_link": .4}),
}
benchmark.atomic(a.output / "protocol_addition.json", dict(
    purpose="power-only and group-only controls specified in the development plan",
    original_grid_preserved=True, wrapper_sha256=benchmark.sha(__file__),
    settings=benchmark.SETTINGS, datasets=benchmark.DATASETS, seeds=[0, 1, 2]))
benchmark.launch(a)
