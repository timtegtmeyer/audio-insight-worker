"""
RunPod serverless handler for selective audio LLM analysis using Qwen2-Audio-7B-Instruct.

Analyzes high-importance audio segments to extract tone, intent, delivery style,
and observations invisible to text-only analysis.

Expected input:
    {
        "audio_url": "https://cdn.example.com/episode.mp3",
        "segments": [
            {"start": 12.5, "end": 28.3, "text": "...", "speaker": "John", "importance_score": 0.92}
        ]
    }

    Debug mode:
    { "debug": true }
"""
import json
import logging
import os
import sys
import tempfile
from typing import Any

import librosa
import requests
import torch
import runpod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("audio-insight-worker")

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/Qwen2-Audio-7B-Instruct")

# ---------------------------------------------------------------------------
# Model — loaded once at container start
# ---------------------------------------------------------------------------

_processor = None
_model = None
_load_error = None


def _load_model() -> None:
    global _processor, _model, _load_error

    if _model is not None:
        return

    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    log.info("Loading Qwen2-Audio model from %s", MODEL_PATH)
    try:
        _processor = AutoProcessor.from_pretrained(MODEL_PATH)
        _model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        _model.eval()
        log.info("Model loaded successfully")
    except Exception as exc:
        _load_error = str(exc)
        log.exception("Failed to load model")
        raise


try:
    _load_model()
except Exception as exc:
    log.error("Could not pre-load model: %s", exc)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_segment(
    audio_array: Any,
    sr: int,
    text: str,
    speaker: str,
) -> dict[str, str]:
    """Run audio LLM on a single segment."""
    assert _processor is not None and _model is not None

    conversation = [
        {
            "role": "system",
            "content": "You analyze podcast audio segments. Respond with ONLY valid JSON.",
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "PLACEHOLDER"},
                {
                    "type": "text",
                    "text": (
                        f"Speaker: {speaker}\n"
                        f"Transcript: {text}\n\n"
                        "Analyze the speaker's delivery in this audio segment. "
                        "Respond with JSON: "
                        '{"tone": "<warm|cold|neutral|aggressive|enthusiastic|hesitant|authoritative>", '
                        '"intent": "<inform|persuade|question|deflect|agree|challenge|joke>", '
                        '"delivery": "<confident|uncertain|passionate|monotone|animated|rushed|measured>", '
                        '"notes": "<one sentence observation about what stands out>"}'
                    ),
                },
            ],
        },
    ]

    text_input = _processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = [librosa.resample(audio_array, orig_sr=sr, target_sr=16000)]
    inputs = _processor(text=text_input, audios=audios, return_tensors="pt", padding=True)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)

    response = _processor.batch_decode(
        output_ids[:, inputs["input_ids"].size(1):],
        skip_special_tokens=True,
    )[0]

    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return {
            "tone": "unknown",
            "intent": "unknown",
            "delivery": "unknown",
            "notes": response[:200],
        }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    input_data: dict[str, Any] = job.get("input", {})

    if input_data.get("debug"):
        return {
            "status": "ok" if _model is not None else "error",
            "worker": "audio-insight-worker",
            "model_path": MODEL_PATH,
            "load_error": _load_error,
        }

    audio_url: str = input_data["audio_url"]
    segments: list[dict[str, Any]] = input_data["segments"]

    log.info("Analyzing %d segments from %s", len(segments), audio_url[:80])

    # Download audio
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        resp = requests.get(audio_url, timeout=300)
        resp.raise_for_status()
        tmp.write(resp.content)
        tmp.flush()
    finally:
        tmp.close()

    y_full, sr = librosa.load(tmp.name, sr=16000, mono=True)

    results: list[dict[str, str] | None] = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        y_seg = y_full[start_sample:end_sample]

        if len(y_seg) < sr * 0.5:  # skip < 500ms
            results.append(None)
            continue

        try:
            insight = analyze_segment(y_seg, sr, seg.get("text", ""), seg.get("speaker", ""))
            results.append(insight)
            log.info("Segment %.1f-%.1f: tone=%s intent=%s", seg["start"], seg["end"], insight.get("tone"), insight.get("intent"))
        except Exception as exc:
            log.warning("Failed to analyze segment %.1f-%.1f: %s", seg["start"], seg["end"], exc)
            results.append(None)

    os.unlink(tmp.name)

    analyzed = sum(1 for r in results if r is not None)
    log.info("Done: %d/%d segments analyzed", analyzed, len(segments))

    return {"insights": results}


runpod.serverless.start({"handler": handler})
