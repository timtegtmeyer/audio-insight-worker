FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps (stable layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 wget curl git && \
    rm -rf /var/lib/apt/lists/*

# Base Python deps (stable layer — torch, transformers)
COPY builder/requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# App Python deps (changes sometimes)
COPY builder/requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Model download (large, stable layer)
COPY builder/download_model.sh /builder/download_model.sh
RUN chmod +x /builder/download_model.sh
RUN --mount=type=secret,id=HF_TOKEN /builder/download_model.sh

# Source code (small, changes often — LAST)
COPY src/ /app/src/

ENV MODEL_PATH="/models/Qwen2-Audio-7B-Instruct"

CMD ["python3", "-u", "/app/src/handler.py"]
