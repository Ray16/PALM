"""Opt-in preparation of OMol25 — the one large, *gated* PALM dataset.

Unlike the other ``prepare_*`` scripts, OMol25 is **not** part of the default
``download_all`` run: the raw data is a ~28 GB gated download from the
``facebook/OMol25`` HuggingFace repo, and building the cache the loader reads
(``_cache/features.npy`` [9.55M × 115] + ``meta.parquet``) featurizes every
structure. Run it explicitly:

    python -m PALM.data.download_all --only omol25        # via the orchestrator
    python -m PALM.data.prepare_omol25 --workers 64       # or directly

Preconditions (checked up-front; if unmet the script prints exact steps and
stops rather than failing obscurely):
  1. Accept the OMol25 license at https://huggingface.co/facebook/OMol25
  2. ``pip install huggingface_hub`` and authenticate — ``huggingface-cli login``
     (or ``export HF_TOKEN=...``).
  3. ``ase`` + ``pandas`` (already in the ``[benchmark]`` extra) for featurizing.

Idempotent at every stage: skips the whole run if the merged cache exists, skips
the download if the tarballs / extracted shards are already present, and the
featurizer itself skips shards it has already done.
"""

from __future__ import annotations

import glob
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "omol25")
CACHE_DIR = os.path.join(DATA_DIR, "_cache")
SPLITS = ["train_4M", "val", "test"]

# Gated HuggingFace repo hosting the *.aselmdb data (per fairchem's OMol25 docs).
# It is registered as a *model* repo (URL has no /datasets/ segment); both the id
# and the repo type are overridable in case the org re-hosts it.
HF_REPO = os.environ.get("OMOL25_HF_REPO", "facebook/OMol25")
HF_REPO_TYPE = os.environ.get("OMOL25_HF_REPO_TYPE", "model")
LICENSE_URL = "https://huggingface.co/facebook/OMol25"


class OMol25Unavailable(RuntimeError):
    """A precondition for fetching/featurizing OMol25 is not met."""


def _has_merged_cache() -> bool:
    return (os.path.exists(os.path.join(CACHE_DIR, "features.npy"))
            and os.path.exists(os.path.join(CACHE_DIR, "meta.parquet")))


def _shards_present() -> bool:
    return all(glob.glob(os.path.join(DATA_DIR, s, "*.aselmdb")) for s in SPLITS)


def _tarballs_present() -> bool:
    return all(os.path.exists(os.path.join(DATA_DIR, f"{s}.tar.gz")) for s in SPLITS)


def _fail(reason: str, steps: str) -> None:
    """Print actionable instructions, then raise so callers see a clean failure."""
    print("\n" + "=" * 72, file=sys.stderr)
    print(f"[omol25] cannot proceed: {reason}", file=sys.stderr)
    print(steps.rstrip(), file=sys.stderr)
    print("=" * 72 + "\n", file=sys.stderr)
    raise OMol25Unavailable(reason)


def _download() -> None:
    """Fetch the three split tarballs from the gated HF dataset into DATA_DIR."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _fail("huggingface_hub is not installed",
              "  pip install huggingface_hub\n"
              "  huggingface-cli login        # or: export HF_TOKEN=...")
        return

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    os.makedirs(DATA_DIR, exist_ok=True)
    # Fetch whichever form the repo hosts: per-split tarballs and/or raw shards.
    patterns = [p for s in SPLITS
                for p in (f"{s}.tar.gz", f"**/{s}.tar.gz", f"{s}/*.aselmdb", f"**/{s}/*.aselmdb")]
    print(f"[omol25] downloading {SPLITS} data (~28 GB, gated) from "
          f"{HF_REPO} [{HF_REPO_TYPE}] ...", flush=True)
    try:
        snapshot_download(repo_id=HF_REPO, repo_type=HF_REPO_TYPE,
                          allow_patterns=patterns, local_dir=DATA_DIR, token=token)
    except Exception as exc:  # noqa: BLE001 — surface as actionable guidance
        _fail(f"HuggingFace download failed ({type(exc).__name__}: "
              f"{str(exc)[:200]})",
              f"  1. Accept the license (once) at {LICENSE_URL}\n"
              "  2. Authenticate: huggingface-cli login  (or export HF_TOKEN=...)\n"
              "  3. Re-run: python -m PALM.data.prepare_omol25\n"
              "  If the org has moved the data, set OMOL25_HF_REPO / OMOL25_HF_REPO_TYPE.")
        return

    # snapshot_download may nest files under the repo's own subdirs; hoist the
    # tarballs (and any extracted shards) up so extraction/featurization find them.
    for s in SPLITS:
        dest = os.path.join(DATA_DIR, f"{s}.tar.gz")
        if not os.path.exists(dest):
            found = glob.glob(os.path.join(DATA_DIR, "**", f"{s}.tar.gz"), recursive=True)
            if found:
                os.replace(found[0], dest)
        if not glob.glob(os.path.join(DATA_DIR, s, "*.aselmdb")):
            nested = glob.glob(os.path.join(DATA_DIR, "**", s, "*.aselmdb"), recursive=True)
            if nested:
                os.makedirs(os.path.join(DATA_DIR, s), exist_ok=True)
                for f in nested:
                    os.replace(f, os.path.join(DATA_DIR, s, os.path.basename(f)))


def _extract() -> None:
    for s in SPLITS:
        if glob.glob(os.path.join(DATA_DIR, s, "*.aselmdb")):
            continue                                   # already extracted
        tar = os.path.join(DATA_DIR, f"{s}.tar.gz")
        if not os.path.exists(tar):
            continue
        print(f"[omol25] extracting {s}.tar.gz ...", flush=True)
        with tarfile.open(tar) as t:
            try:
                t.extractall(DATA_DIR, filter="data")   # py>=3.12 safe extraction
            except TypeError:
                t.extractall(DATA_DIR)                   # older Python fallback


def prepare(workers: int = 64) -> None:
    """Ensure the OMol25 descriptor cache exists, downloading + featurizing if needed."""
    if _has_merged_cache():
        print(f"[omol25] descriptor cache already present at {CACHE_DIR} — nothing to do")
        return

    if not _shards_present():
        if not _tarballs_present():
            _download()
        _extract()

    if not _shards_present():
        _fail("no *.aselmdb shards found after download/extract",
              f"  Expected {DATA_DIR}/{{train_4M,val,test}}/*.aselmdb\n"
              f"  Accept the license at {LICENSE_URL}, authenticate, and re-run.")

    # ase-only featurizer (no fairchem/GPU); portable, repo-relative paths.
    try:
        from PALM.benchmarks.omol25.omol25_featurize_merged import featurize
    except ImportError as exc:
        _fail(f"featurizer import failed ({exc})",
              "  pip install -e \".[benchmark]\"   # provides ase + pandas")
        return
    print(f"[omol25] featurizing shards -> {CACHE_DIR}/features.npy (this is the slow step)",
          flush=True)
    featurize(workers)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=64,
                    help="parallel featurization workers (default 64)")
    args = ap.parse_args(argv)
    try:
        prepare(args.workers)
    except OMol25Unavailable:
        return 1                     # instructions already printed by _fail
    return 0


if __name__ == "__main__":
    sys.exit(main())
