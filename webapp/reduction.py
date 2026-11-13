"""Reuse identical display reductions across maps, methods and splitting runs.

Only display coordinates are cached. Splits and their quality metrics are
always computed from the original features. Heavy libraries load in workers.
"""
import contextlib
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import os
import tempfile
import zipfile
from pathlib import Path

_native_reduce = None


@functools.lru_cache(maxsize=1)
def signature_stamp():
    """Invalidate display caches when the recipe, features or libraries change."""
    root = Path(importlib.util.find_spec('PALM').origin).parent
    h = hashlib.sha256(b'webapp-display-reduction-v2')
    for path in [Path(__file__), root/'benchmarks/datasail_pipeline/visualization.py',
                 root/'benchmarks/datasail_pipeline/cache.py', root/'benchmarks/datasail_pipeline/config.py', *sorted((root/'features').glob('*.py'))]:
        h.update(path.name.encode()); h.update(path.read_bytes())
    for package in ['numpy', 'scikit-learn', 'scipy', 'rdkit', 'torch', 'transformers', 'matminer', 'pymatgen']:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = 'absent'
        h.update((package + '=' + version).encode())
    return h.hexdigest()


def _native():
    global _native_reduce
    if _native_reduce is None:
        from PALM.benchmarks.datasail_pipeline.visualization import _reduce
        _native_reduce = _reduce
    return _native_reduce


@contextlib.contextmanager
def _lock(path):
    # Advisory locks are released by the OS if a cancellable worker is killed.
    # Atomic writes still protect correctness on platforms without flock.
    try:
        import fcntl
    except ImportError:
        yield
        return
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def cached_reduce(X, method, seed=42):
    import numpy as np
    matrix = np.ascontiguousarray(X)
    h = hashlib.sha256(signature_stamp().encode())
    h.update(f'{matrix.dtype}:{matrix.shape}:{method}:{seed}'.encode())
    h.update(matrix.tobytes())
    key = h.hexdigest()
    root = Path(os.environ.get('PALM_CACHE_DIR', '~/.palm_cache')).expanduser()/'webapp_reductions'
    root.mkdir(parents=True, exist_ok=True)
    path = root/(key+'.npz')

    def read():
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as saved:
                coords = saved['coords']
            if coords.shape == (len(matrix), 2) and np.isfinite(coords).all():
                return coords
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass
        return None

    hit = read()
    if hit is not None:
        return hit
    with _lock(root/(key+'.lock')):
        hit = read()
        if hit is not None:
            return hit
        coords = _native()(X, method, seed=seed)
        if coords is None:
            return None
        coords = np.asarray(coords)
        if coords.shape != (len(matrix), 2) or not np.isfinite(coords).all():
            raise ValueError('Display reduction produced invalid coordinates')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, suffix='.npz', delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, coords=coords)
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return coords


def install_pipeline_cache():
    """Patch only this isolated worker's display hooks, preserving the recipe."""
    from PALM.benchmarks.datasail_pipeline import pipeline, visualization
    _native()  # capture the actual reducer before installing the cache wrapper
    pipeline._reduce = cached_reduce
    visualization._reduce = cached_reduce
