#!/usr/bin/env bash
# Start the Streamlit stand-up app (calls the API).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
exec streamlit run frontend/app.py \
  --server.address 0.0.0.0 \
  --server.port "${GBME_UI_PORT:-8911}" \
  --server.headless true \
  --browser.gatherUsageStats false
