#!/bin/bash
set -e

MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct"
MODEL_DIR="/models/Qwen2-Audio-7B-Instruct"

HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo '')"

if [ -n "$HF_TOKEN" ]; then
    # Support both old (huggingface-cli) and new (hf) CLI versions
    if command -v hf &>/dev/null; then
        hf auth login --token "$HF_TOKEN"
    else
        huggingface-cli login --token "$HF_TOKEN"
    fi
fi

echo "Downloading $MODEL_ID to $MODEL_DIR ..."
mkdir -p "$MODEL_DIR"

# Support both old and new CLI
if command -v hf &>/dev/null; then
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_ID" \
        --local-dir "$MODEL_DIR"
else
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
fi

echo "Download complete."
