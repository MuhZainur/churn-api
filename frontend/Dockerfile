FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# salin app & entrypoint
COPY . .
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# default port (Cloud Run akan override env PORT)
ENV PORT=8501
EXPOSE 8501

# gunakan script sh (tanpa perlu bash) agar PORT dipakai dengan benar
CMD ["/app/start.sh"]
