#!/bin/sh
set -euo pipefail

echo "== churn-ui boot =="
echo "PWD: $(pwd)"
echo "Listing /app:"
ls -lah /app || true
echo "PORT env: ${PORT:-<unset>}"

# Cek file & modul yang sering salah
python - <<'PY'
import os, sys
print("Python:", sys.version)
print("PORT env:", os.getenv("PORT"))
assert os.path.exists("/app/streamlit_app.py"), "ERROR: /app/streamlit_app.py not found"
try:
    import streamlit as st
    print("Streamlit version:", st.__version__)
except Exception as e:
    print("ERROR: streamlit import failed ->", repr(e))
    raise
PY

# Jalankan Streamlit headless, bind ke 0.0.0.0 dan PORT dari env
exec python -m streamlit run /app/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
