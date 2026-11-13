FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    WEBAPP_WORKER_THREADS=2 WEBAPP_SPLITTER_THREADS=2 WEBAPP_MAX_CONCURRENT_RUNS=1 \
    WEBAPP_MAX_CONCURRENT_MAPS=1 WEBAPP_MAX_CONCURRENT_JOBS=1 \
    WEBAPP_MAX_PENDING_RUNS=8 WEBAPP_MAX_CLIENT_RUNS=2 WEBAPP_MAX_PENDING_MAPS=4 \
    WEBAPP_RUN_TIMEOUT_SECONDS=3600 WEBAPP_MAP_TIMEOUT_SECONDS=1800 \
    PALM_MAX_UPLOAD_MB=25 PALM_MAX_EXTRACT_MB=100 PALM_MAX_ARCHIVE_FILES=5000 \
    DATASAIL_JOBS_DIR=/data/jobs PALM_CACHE_DIR=/data/cache \
    HF_HOME=/data/models PALM_CORS_ORIGINS="" PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 libglib2.0-0 libxrender1 libxext6 build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-cpu.txt /app/requirements-cpu.txt
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /app/requirements-cpu.txt
COPY vendor/ /app/vendor/
RUN python -m pip install --no-cache-dir --no-deps /app/vendor/*.whl
COPY PALM/ /app/PALM/
RUN useradd --create-home --uid 1000 app && mkdir -p /data \
    && chown -R app:app /data /app
USER app
RUN python -c "import sys,PALM.webapp.app; assert 'torch' not in sys.modules; from lowrank.splitters import split; import mtkahypar,rdkit"
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz',timeout=3)"
CMD ["python", "-m", "PALM.webapp.deployment.serve"]
