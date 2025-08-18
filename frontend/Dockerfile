FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# default; Cloud Run akan override dengan env PORT sendiri
ENV PORT=8501
EXPOSE 8501

# gunakan shell form agar ${PORT} diexpand dan bind ke 0.0.0.0
CMD bash -lc 'exec python -m streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT}'
