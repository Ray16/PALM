"""Download a composition-level snapshot of Materials Project to a local CSV.

Pulls up to ``N`` summary documents (no structures — just formula + scalar
properties) so the ``materials_project`` dataset is reproducible offline instead
of hitting the live API on every load.

    python PALM/data/download_materials_project.py            # 100k rows
    python PALM/data/download_materials_project.py --n 5000   # smaller

Needs ``MP_API_KEY`` in the environment (see ~/.bashrc). Output:
``PALM/data/materials_project/summary.csv`` with columns:
    material_id, formula_pretty, formation_energy_per_atom, energy_above_hull,
    band_gap, density, nsites, is_stable, spacegroup
"""

import argparse
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "materials_project")
OUT = os.path.join(OUT_DIR, "summary.csv")

FIELDS = [
    "material_id", "formula_pretty", "formation_energy_per_atom",
    "energy_above_hull", "band_gap", "density", "nsites", "is_stable",
    "symmetry",
]


def download(n: int) -> str:
    key = os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        raise SystemExit("MP_API_KEY not set (export it or add to ~/.bashrc)")
    from mp_api.client import MPRester

    per = 1000  # MP caps chunk_size at 1000
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(
            fields=FIELDS,
            num_chunks=max(1, math.ceil(n / per)),
            chunk_size=per,
        )
    docs = docs[:n]

    rows = []
    for d in docs:
        sym = getattr(d, "symmetry", None)
        rows.append({
            "material_id": str(d.material_id),
            "formula_pretty": getattr(d, "formula_pretty", None),
            "formation_energy_per_atom": getattr(d, "formation_energy_per_atom", None),
            "energy_above_hull": getattr(d, "energy_above_hull", None),
            "band_gap": getattr(d, "band_gap", None),
            "density": getattr(d, "density", None),
            "nsites": getattr(d, "nsites", None),
            "is_stable": getattr(d, "is_stable", None),
            "spacegroup": getattr(sym, "symbol", None) if sym is not None else None,
        })
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[MP] wrote {len(df)} rows -> {OUT}")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    args = ap.parse_args()
    download(args.n)
