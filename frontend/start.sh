#!/bin/sh
set -e

# Cloud Run inject PORT; fallback 8501 untuk lokal
PORT="${PORT:-8501}"

# Jalankan Streamlit headless dan bind ke 0.0.0.0
exec python -m streamlit run /app/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port="$PORT" \
  --browser.gatherUsageStats=false \
  --server.headless=true
