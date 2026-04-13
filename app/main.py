"""
Parakeet TDT 0.6b v3 — OpenAI-Compatible ASR API

Endpoints:
  POST /v1/audio/transcriptions   — OpenAI Whisper-compatible transcription
  POST /v1/audio/translations     — (stub) maps to transcription
  GET  /health                    — Health check
  GET  /v1/models                 — List available models
  GET  /                          — Service info

Compatible with any client that speaks the OpenAI audio API:
  - Open WebUI
  - Home Assistant Whisper integration
  - Any OpenAI SDK client
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.transcriber import transcriber, MODEL_NAME, DEVICE

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("parakeet-api")


# ── Lifespan: load model at startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*60)
    logger.info("Parakeet TDT 0.6b v3 — Starting up")
    logger.info(f"  Model:  {MODEL_NAME}")
    logger.info(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"  GPU:    {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"  VRAM:   {mem:.1f} GB")
    logger.info("="*60)

    # Load model (downloads on first run, ~1.2 GB)
    transcriber.load_model()
    logger.info("Model ready — accepting requests")

    yield

    logger.info("Shutting down...")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Parakeet ASR API",
    description="OpenAI-compatible Speech-to-Text powered by NVIDIA Parakeet TDT 0.6b v3",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local/homelab use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── GET / — Service info ────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "parakeet-asr",
        "model": MODEL_NAME,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "endpoints": {
            "transcribe": "/v1/audio/transcriptions",
            "models": "/v1/models",
            "health": "/health",
        },
        "supported_languages": [
            "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk",
            "cs", "sk", "sl", "hr", "bg", "ro", "hu", "el", "da", "sv",
            "fi", "et", "lt", "lv", "mt",
        ],
    }


# ── GET /health ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    gpu_mem = None
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_mem = {"allocated_gb": round(allocated, 2), "total_gb": round(total, 1)}

    return {
        "status": "ready" if transcriber._loaded else "loading",
        "model": MODEL_NAME,
        "device": DEVICE,
        "gpu_memory": gpu_mem,
    }


# ── GET /v1/models — OpenAI-compatible model list ───────────────────────────
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "parakeet-tdt-0.6b-v3",
                "object": "model",
                "created": 1723593600,  # Aug 2025
                "owned_by": "nvidia",
                "permission": [],
            },
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1723593600,
                "owned_by": "nvidia",
                "permission": [],
                "_note": "Alias — routes to parakeet-tdt-0.6b-v3 for OpenAI client compat",
            },
        ],
    }


# ── POST /v1/audio/transcriptions — OpenAI Whisper-compatible endpoint ──────
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="parakeet-tdt-0.6b-v3"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    temperature: Optional[float] = Form(default=0.0),
    timestamp_granularities: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
):
    """
    OpenAI-compatible audio transcription endpoint.

    Accepts the same parameters as OpenAI's /v1/audio/transcriptions:
      - file: Audio file (wav, mp3, flac, ogg, webm, m4a, mp4, etc.)
      - model: Model name (ignored — always uses parakeet-tdt-0.6b-v3)
      - language: ISO language code or 'auto' for detection
      - response_format: 'json', 'text', 'verbose_json', 'srt', 'vtt'
      - temperature: (ignored — Parakeet is non-generative)
    """
    if not transcriber._loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read audio bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {e}")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Check file size (limit to 200 MB)
    max_size = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {max_size // (1024*1024)} MB"
        )

    # Include timestamps?
    want_timestamps = (
        response_format in ("verbose_json", "srt", "vtt")
        or timestamp_granularities is not None
    )

    # Transcribe
    logger.info(
        f"Transcription request: {file.filename} "
        f"({len(audio_bytes) / 1024:.0f} KB, lang={language}, fmt={response_format})"
    )

    start_time = time.time()

    try:
        result = transcriber.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename,
            language=language,
            timestamps=want_timestamps,
        )
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    elapsed = time.time() - start_time
    rtfx = result.get("duration", 0) / elapsed if elapsed > 0 else 0

    logger.info(
        f"Transcription complete: {result.get('duration', 0):.1f}s audio "
        f"in {elapsed:.1f}s ({rtfx:.0f}x realtime)"
    )
    # ── Write response in the Log ────────────────────────────────────────
    logger.info(f"Recognized Text: {result['text']}")

    # ── Format response ──────────────────────────────────────────────────
    if response_format == "text":
        return JSONResponse(
            content=result["text"],
            media_type="text/plain",
        )

    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language or "auto",
            "duration": result.get("duration", 0),
            "text": result["text"],
            "segments": result.get("segments", []),
        }

    if response_format == "srt":
        srt = _to_srt(result)
        return JSONResponse(content=srt, media_type="text/plain")

    if response_format == "vtt":
        vtt = _to_vtt(result)
        return JSONResponse(content=vtt, media_type="text/plain")

    # Default: json (OpenAI format)
    return {"text": result["text"]}


# ── POST /v1/audio/translations — stub (maps to transcription) ──────────────
@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="parakeet-tdt-0.6b-v3"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    temperature: Optional[float] = Form(default=0.0),
    prompt: Optional[str] = Form(default=None),
):
    """
    Stub for translation endpoint.
    Parakeet TDT v3 does transcription only (no translation).
    Falls back to transcription.
    """
    logger.warning(
        "Translation endpoint called — Parakeet v3 does not translate. "
        "Falling back to transcription."
    )
    return await transcribe(
        file=file,
        model=model,
        language=language,
        response_format=response_format,
        temperature=temperature,
    )


# ── Subtitle formatters ─────────────────────────────────────────────────────

def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _to_srt(result: dict) -> str:
    """Convert transcription result to SRT format."""
    segments = result.get("segments", [])
    if not segments:
        # No timestamps — return single block
        return f"1\n00:00:00,000 --> 99:59:59,999\n{result['text']}\n"

    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp_srt(seg.get("start", 0))
        end = _format_timestamp_srt(seg.get("end", 0))
        lines.append(f"{i}\n{start} --> {end}\n{seg.get('text', '')}\n")
    return "\n".join(lines)


def _to_vtt(result: dict) -> str:
    """Convert transcription result to WebVTT format."""
    segments = result.get("segments", [])
    header = "WEBVTT\n\n"
    if not segments:
        return header + f"00:00:00.000 --> 99:59:59.999\n{result['text']}\n"

    lines = [header]
    for seg in segments:
        start = _format_timestamp_vtt(seg.get("start", 0))
        end = _format_timestamp_vtt(seg.get("end", 0))
        lines.append(f"{start} --> {end}\n{seg.get('text', '')}\n")
    return "\n".join(lines)
