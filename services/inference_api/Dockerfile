FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app/services/inference_api

# Repo layout inside image:
# /app/ml/...
# /app/services/inference_api/...
COPY ml /app/ml
COPY services/inference_api/requirements.txt /app/services/inference_api/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/services/inference_api/requirements.txt

COPY services/inference_api/app /app/services/inference_api/app

ENV PYTHONPATH=/app/ml/src \
    REPO_ROOT=/app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
