"""Pre-compute features + t-SNE maps for every demo so demos load instantly.

Featurization goes into PALM's feature cache (shared with pipeline runs) and
the maps into the web app's map cache, both keyed by content, so jobs created
from the demos hit them directly. Protein demos use ESM2-650M: run this on a
reserved GPU, from data_splitting/:

    gpu_reserve run <idx> --label palm-demo-prewarm -- \
        conda run -n palm python PALM/webapp/demo_data/prewarm.py
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))         # directory containing PALM/

from PALM.webapp.embedding import compute_map  # noqa: E402


def main(only=None):
    demos = json.loads((HERE / "demos.json").read_text())
    for d in demos:
        if only and d["id"] not in only:
            continue
        for ax in ("e1", "e2"):
            ent = d.get(ax)
            if not ent:
                continue
            t0 = time.time()
            m = compute_map(HERE / d["file"], ent["column"], ent["type"], ent["feature_sets"],
                            progress=lambda msg: print("   ", msg, flush=True))
            print(f"{d['id']:22s} {ax}  {m['n_shown']:5d} pts  {m['embedding']}  "
                  f"({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
