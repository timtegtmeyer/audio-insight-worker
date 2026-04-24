FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# torch 2.8 + CUDA 12.8 is the minimum combination that supports Blackwell
# (sm_100 / sm_120) — RTX 5090 workers would otherwise hang in "initializing"
# because 2.5.1-cu124 has no kernels for the GPU it landed on.

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
