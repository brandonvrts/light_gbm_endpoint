#!/usr/bin/env bash
# Start the light_gbm_endpoint API (CPU-only, FastAPI/uvicorn) on the DGX Spark.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
exec python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${GBME_PORT:-8910}" \
  --workers "${GBME_WORKERS:-1}" \
  --log-level info
