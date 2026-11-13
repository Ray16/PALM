# Deployment

The app serves its own frontend and a Python compute API. Run **one** API process:
the on-disk queue has a single owner. Run/map workers share a configurable compute
budget; adding HTTP workers would create duplicate queue consumers.

## Private access

Keep the server on loopback (the default) and reach it through an SSH tunnel or an
authenticated HTTPS reverse proxy:

```bash
ssh -N -L 8080:127.0.0.1:8080 <user>@<host>    # then open http://localhost:8080
```

Set `PALM_AUTH_USER` and `PALM_AUTH_PASSWORD` to require HTTP Basic login. Keep the
credentials in server configuration (e.g. a mode-600 env file), never in this repo.

## Portable container

Build the low-rank wheel, then create a clean build context (run from the directory
containing `PALM/`):

```bash
pip wheel --no-deps PALM/low_rank -w dist/
python PALM/webapp/deployment/export.py --output /path/to/new-build-context \
  --lowrank-wheel dist/lowrank_split-*.whl
cd /path/to/new-build-context
# Set PALM_AUTH_PASSWORD in the shell or a private .env file first.
docker compose up --build -d
```

The build context contains only the app sources, PALM's Python modules, the six
public demos and the low-rank wheel: no uploads, jobs, credentials, caches,
`publications/` or research datasets. Compose publishes only `127.0.0.1:7860`; the
persistent `/data` volume holds uploads, caches and model downloads.

The container profile is CPU-only: one compute task at a time with two native
threads, 25 MB uploads, a 100 MB ZIP-extraction limit, bounded per-browser run
queues, and runs that expire after 24 hours of inactivity (see `Dockerfile` for the
environment variables). A shutdown interrupts the active run cleanly; queued runs
survive a restart. Protein/transformer embeddings keep the selected representation
and can be slow on CPU; no GPU is used automatically. Display-coordinate caching
changes neither assignments nor quality metrics.

The image uses Python 3.11 to get binary wheels for scikit-learn 1.2.2 (DataSAIL's
API). DataSAIL 1.4 is pinned; the worker passes custom matrices through the SDK's
callable form, avoiding DataSAIL 1.4's SciPy-1.17 validation error for
array-containing tuples, and solver threads follow the worker budget instead of
DataSAIL's fourteen-thread default. Optional crystal featurizers need their extra
packages; lightweight molecular/formula features and the supplied sequence features
are included.

## Hugging Face Space

The exported context plus `space.README.md` (copied as the context's `README.md`)
form a Docker Space. Choose **private** visibility in the Space settings. Docker
Spaces require an eligible paid plan; the bundle never requests paid hardware.
Ephemeral Space storage does not survive a rebuild, so attach persistent storage
before relying on it for results.

## Verification

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest -q PALM/webapp/tests
```
