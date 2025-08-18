# Dockerfile
FROM python:3.10-slim
WORKDIR /app
#
# runtime untuk scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# copy dep dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app dan model
COPY app.py /app/
COPY model_bundle.joblib /app/

ENV BUNDLE_PATH=/app/model_bundle.joblib

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host","0.0.0.0", "--port","8000"]
