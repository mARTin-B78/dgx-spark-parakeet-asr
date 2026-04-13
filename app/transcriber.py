"""
Parakeet TDT 0.6b v3 — NeMo Model Wrapper for DGX Spark

Handles:
  - Model loading with GPU placement
  - Audio preprocessing (format conversion, resampling to 16kHz mono)
  - Long audio chunking (24-min segments for full-attention, or local-attention for 3h+)
  - Batch inference with memory management
"""

import io
import os
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
MAX_SEGMENT_SECONDS = int(os.getenv("MAX_SEGMENT_SECONDS", "1200"))  # 20 min default
TARGET_SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ParakeetTranscriber:
    """Singleton wrapper around the NeMo Parakeet TDT model."""

    def __init__(self):
        self.model = None
        self._loaded = False

    def load_model(self):
        """Load the Parakeet model onto GPU. Called once at startup."""
        if self._loaded:
            return

        logger.info(f"Loading model {MODEL_NAME} on {DEVICE}...")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        self.model.eval()

        if DEVICE == "cuda":
            self.model = self.model.cuda()

        self._loaded = True
        logger.info(f"Model loaded successfully on {DEVICE}")

    def _convert_to_wav_16k_mono(self, audio_bytes: bytes, original_filename: str) -> str:
        """
        Convert any audio format to WAV 16kHz mono using ffmpeg.
        Returns path to temporary WAV file.
        """
        suffix = Path(original_filename).suffix.lower() if original_filename else ".wav"

        # Write input to temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        # Output WAV path
        tmp_out_path = tmp_in_path.replace(suffix, ".converted.wav")

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_in_path,
                "-ac", "1",              # mono
                "-ar", "16000",          # 16kHz
                "-sample_fmt", "s16",    # 16-bit PCM
                "-f", "wav",
                tmp_out_path
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg conversion failed: {stderr[:500]}")

            return tmp_out_path

        finally:
            # Clean up input file (keep output for caller to clean)
            try:
                os.unlink(tmp_in_path)
            except OSError:
                pass

    def _chunk_audio_file(self, wav_path: str) -> list[str]:
        """
        Split a WAV file into segments of MAX_SEGMENT_SECONDS.
        Returns list of paths to chunk files.
        """
        data, sr = sf.read(wav_path)
        total_seconds = len(data) / sr

        if total_seconds <= MAX_SEGMENT_SECONDS:
            return [wav_path]

        logger.info(
            f"Audio is {total_seconds:.0f}s, chunking into "
            f"{MAX_SEGMENT_SECONDS}s segments..."
        )

        chunk_samples = MAX_SEGMENT_SECONDS * sr
        chunks = []
        offset = 0

        while offset < len(data):
            end = min(offset + chunk_samples, len(data))
            chunk_data = data[offset:end]

            chunk_path = wav_path.replace(".wav", f".chunk{len(chunks):04d}.wav")
            sf.write(chunk_path, chunk_data, sr)
            chunks.append(chunk_path)
            offset = end

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
        timestamps: bool = False,
    ) -> dict:
        """
        Transcribe audio bytes → text.

        Args:
            audio_bytes: Raw audio file bytes (any format ffmpeg can read)
            filename:    Original filename (used for format detection)
            language:    Language code (e.g. 'en', 'de', 'fr') or None for auto-detect
            timestamps:  Whether to include word-level timestamps

        Returns:
            dict with 'text', 'language' (detected), 'duration', 'segments' (if timestamps)
        """
        if not self._loaded:
            self.load_model()

        wav_path = None
        chunk_paths = []

        try:
            # Step 1: Convert to WAV 16kHz mono
            wav_path = self._convert_to_wav_16k_mono(audio_bytes, filename)

            # Get audio duration
            data, sr = sf.read(wav_path)
            duration = len(data) / sr
            logger.info(f"Audio duration: {duration:.1f}s ({filename})")

            # Step 2: Chunk if needed
            chunk_paths = self._chunk_audio_file(wav_path)

            # Step 3: Transcribe each chunk
            all_texts = []
            all_segments = []
            time_offset = 0.0

            for i, chunk_path in enumerate(chunk_paths):
                logger.info(f"Transcribing chunk {i + 1}/{len(chunk_paths)}...")

                # Configure transcription
                transcribe_kwargs = {
                    "batch_size": 1,
                }

                if timestamps:
                    transcribe_kwargs["return_hypotheses"] = True

                # Run inference
                with torch.no_grad():
                    output = self.model.transcribe(
                        [chunk_path],
                        **transcribe_kwargs,
                    )

                # Extract text
                if timestamps and hasattr(output, '__iter__'):
                    # When return_hypotheses=True, output is list of Hypothesis objects
                    for hyp in output:
                        if hasattr(hyp, 'text'):
                            all_texts.append(hyp.text)
                        elif isinstance(hyp, str):
                            all_texts.append(hyp)
                        # Extract timestamps if available
                        if hasattr(hyp, 'timestep') and hyp.timestep:
                            for ts in hyp.timestep:
                                seg = {
                                    "start": round(ts.get("start", 0) + time_offset, 3),
                                    "end": round(ts.get("end", 0) + time_offset, 3),
                                    "text": ts.get("word", ""),
                                }
                                all_segments.append(seg)
                else:
                    # Simple string output
                    if isinstance(output, list):
                        for item in output:
                            if isinstance(item, str):
                                all_texts.append(item)
                            elif hasattr(item, 'text'):
                                all_texts.append(item.text)
                            elif isinstance(item, list) and len(item) > 0:
                                # Nested list: [[text1], [text2]]
                                all_texts.append(str(item[0]) if not isinstance(item[0], str) else item[0])
                    elif isinstance(output, str):
                        all_texts.append(output)

                # Update time offset for next chunk
                chunk_data, chunk_sr = sf.read(chunk_path)
                time_offset += len(chunk_data) / chunk_sr

                # Free GPU memory between chunks
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            # Combine results
            full_text = " ".join(all_texts).strip()

            result = {
                "text": full_text,
                "duration": round(duration, 3),
                "model": MODEL_NAME,
                "device": DEVICE,
            }

            if timestamps and all_segments:
                result["segments"] = all_segments

            return result

        finally:
            # Clean up temp files
            for path in chunk_paths:
                if path != wav_path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass


# ── Singleton instance ───────────────────────────────────────────────────────
transcriber = ParakeetTranscriber()
