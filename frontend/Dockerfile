FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# optional default; Cloud Run akan override PORT
ENV PORT=8501
EXPOSE 8501

# gunakan shell form agar ${PORT} diexpand, dan bind ke 0.0.0.0
CMD ["bash", "-lc", "python -m streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT}"]
