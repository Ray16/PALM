"""One-command fetch for every git-ignored PALM dataset.

Small/medium datasets (MoleculeNet, Rfam, NASA, LP-PDBBind, PPI, PDBBind-core,
USPTO-MCR) are committed to the repo, so a fresh clone already has them. This
script fetches the *large / regenerable* ones that are git-ignored, so that after
    pip install -e ".[benchmark]"
    python -m PALM.data.download_all
every entry in PALM.data.sources.REGISTRY is loadable and the master benchmark
(PALM.benchmarks.master.run_benchmark) can run end-to-end.

Each step is idempotent (skips if its output already exists). Use --only / --skip
to target subsets, --cache-dir to keep HuggingFace/tmp scratch off a full home
disk, and --mp-n / --limit to size the samplable sources.

    python -m PALM.data.download_all --cache-dir /scratch/palm_cache
    python -m PALM.data.download_all --only qmof oc22
    python -m PALM.data.download_all --skip lincs_l1000    # skip the 5 GB one

Requires the `datasets` optional deps (see pyproject: PyTDC, mp-api, ase, lmdb,
datasets, h5py). Materials Project also needs MP_API_KEY. `omol25` (28 GB) is NOT
fetched here — see data/README.md.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# NOTE: qmof is NOT fetched here. Its 21 MB derived table (data/qmof/qmof.csv) is
# committed to the repo, because the figshare source is a 392 MB zip behind a flaky
# 202-async endpoint. A fresh clone already has it.


# name -> (relative output marker, callable). Order = cheap/small first.
def _steps(mp_n, limit):
    return {
        "tdc":              ("tdc/Solubility_AqSolDB.csv",
                             lambda: __import__("PALM.data.prepare_tdc",
                                                fromlist=["prepare"]).prepare()),
        "genomic":          ("genomic/records.csv",
                             lambda: __import__("PALM.data.prepare_genomic",
                                                fromlist=["prepare"]).prepare()),
        "openpolymer26":    ("openpolymer26/records.csv",
                             lambda: __import__("PALM.data.prepare_openpolymer26",
                                                fromlist=["prepare"]).prepare(limit)),
        "materials_project": ("materials_project/summary.csv",
                              lambda: __import__("PALM.data.download_materials_project",
                                                 fromlist=["download"]).download(mp_n)),
        "oc22":             ("oc22/records.csv",
                             lambda: __import__("PALM.data.prepare_oc22",
                                                fromlist=["prepare"]).prepare()),
        "lincs_l1000":      ("lincs_l1000/records.csv",
                             lambda: __import__("PALM.data.prepare_lincs_l1000",
                                                fromlist=["prepare"]).prepare(limit)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", help="only these datasets")
    ap.add_argument("--skip", nargs="+", default=[], help="skip these datasets")
    ap.add_argument("--cache-dir", help="dir for HF_HOME + TMPDIR (full-disk machines)")
    ap.add_argument("--mp-n", type=int, default=10_000,
                    help="Materials Project rows (100000 for the full snapshot)")
    ap.add_argument("--limit", type=int, default=10_000,
                    help="sample cap for openpolymer26 / lincs_l1000")
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args(argv)

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(args.cache_dir, "hub"))
        os.environ.setdefault("TMPDIR", args.cache_dir)
        print(f"[cache] HF_HOME + TMPDIR -> {args.cache_dir}")

    steps = _steps(args.mp_n, args.limit)
    names = args.only or list(steps)
    done, skipped, failed = [], [], []
    for name in names:
        if name in args.skip or name not in steps:
            continue
        marker, fn = steps[name]
        out = os.path.join(HERE, marker)
        if os.path.exists(out) and not args.force:
            print(f"[{name}] already present ({marker}) — skipping")
            skipped.append(name)
            continue
        try:
            print(f"[{name}] preparing ...")
            fn()
            done.append(name)
        except Exception as exc:                                    # noqa: BLE001
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            failed.append(name)

    print(f"\n== done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
