# RunPod Serverless Dockerfile (GPU-capable) for music generation
# Mirrors the "serverless handler" pattern from your chess project:
# - cold start loads models/tokenizer once
# - per-request calls handler(event)
#
# Note: This uses a CUDA runtime base image and installs PyTorch via pip.
# If you prefer a RunPod base image with torch preinstalled, you can swap the FROM line.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python 3.11 + build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        build-essential \
        && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Make python/python3 point to 3.11
RUN rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    python --version

WORKDIR /app

# Copy only what we need for serverless inference
COPY ml /app/ml
COPY services/runpod_serverless/requirements.txt /app/services/runpod_serverless/requirements.txt
COPY services/runpod_serverless/handler.py /app/handler.py

# Install deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/services/runpod_serverless/requirements.txt

ENV PYTHONPATH=/app/ml/src \
    REPO_ROOT=/app

CMD ["python", "-u", "handler.py"]

