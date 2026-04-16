# Audio LLM — podQuery RunPod Worker

> Selective audio intelligence using Qwen2-Audio-7B-Instruct on RunPod Serverless.

## Purpose

Analyzes high-importance audio segments (identified by the Audio Analysis worker) using an audio-native LLM. Extracts tone, intent, delivery style, and observations that are invisible to text-only analysis. Only runs on segments that meet importance thresholds — not on every utterance.

## Docker Image

| Registry | Image |
|----------|-------|
| GHCR | `ghcr.io/timtegtmeyer/audio-insight-worker` |

## RunPod Endpoint Setup

### 1. Create Serverless Endpoint

- Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
- Click **New Endpoint**

### 2. Configuration

| Setting | Value |
|---------|-------|
| **Container Image** | `ghcr.io/timtegtmeyer/audio-insight-worker@sha256:<digest>` |
| **Container Disk** | 40 GB |
| **GPU Type** | A40 (48 GB) or RTX 4090 (24 GB) |
| **Min Workers** | 0 |
| **Max Workers** | 1 |
| **Idle Timeout** | 5s |
| **Flash Boot** | Enabled |

### 3. Environment Variables

No environment variables required. Model is baked into the image.

### 4. Test the Endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/<endpoint-id>/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"debug": true}}'
```

## Building

### Local Build (recommended)

```bash
make build    # builds, pushes to GHCR, captures digest, tags
make digest   # shows digest for RunPod endpoint config
```

## Input Schema

```json
{
  "audio_url": "https://cdn.example.com/episode.mp3",
  "segments": [
    {
      "start": 12.5,
      "end": 28.3,
      "text": "That's completely wrong and I'll tell you why...",
      "speaker": "John Doe",
      "importance_score": 0.92
    }
  ]
}
```

## Output Schema

```json
{
  "insights": [
    {
      "tone": "aggressive",
      "intent": "challenge",
      "delivery": "passionate",
      "notes": "Speaker raises voice and speeds up when contradicting the previous point"
    }
  ]
}
```

`insights` array is positionally aligned with the input `segments` array. `null` entries indicate segments that were too short to analyze (<500ms).

## Model

- **Qwen2-Audio-7B-Instruct** — audio-native multimodal LLM
- ~14 GB model weight at float16
- Baked into image at build time
