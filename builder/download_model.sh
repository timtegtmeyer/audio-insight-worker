#!/bin/bash
# Resilient per-file model downloader.
#
# `hf download <model>` in one shot historically stalls mid-way through a
# 15-file Qwen2-Audio snapshot with no retry and no visible progress on
# the affected file — the whole Docker layer then hangs indefinitely.
#
# Instead we:
#   1. list the files in the repo via the HF Hub API
#   2. download each file individually with a hard per-file timeout
#   3. retry each up to 3× before giving up
#   4. print progress line-by-line so `docker buildx build --progress=plain`
#      shows exactly which file is active + how long it took
#
# hf_transfer is enabled for throughput; each file's network timeout is
# capped so a stuck HF TCP connection can't hang the whole build.
set -euo pipefail

MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct"
MODEL_DIR="/models/Qwen2-Audio-7B-Instruct"
PER_FILE_TIMEOUT_SEC=600
MAX_ATTEMPTS=3

HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo '')"
if [ -n "$HF_TOKEN" ]; then
    if command -v hf >/dev/null 2>&1; then
        hf auth login --token "$HF_TOKEN"
    else
        huggingface-cli login --token "$HF_TOKEN"
    fi
fi

echo "Listing files in $MODEL_ID ..."
FILES=$(python3 -c "
from huggingface_hub import HfApi
api = HfApi()
for f in sorted(api.list_repo_files('$MODEL_ID')):
    print(f)
")
COUNT=$(echo "$FILES" | wc -l)
echo "Found $COUNT files to download."

mkdir -p "$MODEL_DIR"

INDEX=0
for FILE in $FILES; do
    INDEX=$((INDEX + 1))
    echo "[$INDEX/$COUNT] $FILE"

    ATTEMPT=0
    while :; do
        ATTEMPT=$((ATTEMPT + 1))
        START=$(date +%s)
        set +e
        if command -v hf >/dev/null 2>&1; then
            HF_HUB_ENABLE_HF_TRANSFER=1 \
                timeout --signal=KILL --kill-after=10 "${PER_FILE_TIMEOUT_SEC}s" \
                hf download "$MODEL_ID" "$FILE" --local-dir "$MODEL_DIR"
        else
            HF_HUB_ENABLE_HF_TRANSFER=1 \
                timeout --signal=KILL --kill-after=10 "${PER_FILE_TIMEOUT_SEC}s" \
                huggingface-cli download "$MODEL_ID" "$FILE" \
                    --local-dir "$MODEL_DIR" \
                    --local-dir-use-symlinks False
        fi
        RC=$?
        set -e
        END=$(date +%s)
        ELAPSED=$((END - START))

        if [ $RC -eq 0 ]; then
            echo "[$INDEX/$COUNT] $FILE ok (${ELAPSED}s)"
            break
        fi

        echo "[$INDEX/$COUNT] $FILE failed (exit=$RC, ${ELAPSED}s, attempt $ATTEMPT/$MAX_ATTEMPTS)"
        if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
            echo "[$INDEX/$COUNT] $FILE giving up after $MAX_ATTEMPTS attempts"
            exit 1
        fi
        sleep 3
    done
done

echo "Download complete."
