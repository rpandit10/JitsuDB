"""faster-whisper transcription wrapper with BJJ vocabulary priming."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from jitsudb.models import WordTimestamp

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


def load_model(model_size: str = "medium", compute_type: str = "float16") -> "WhisperModel":
    """Load the faster-whisper model. Call once at pipeline start; reuse across videos.

    Attempts CUDA first. If unavailable, falls back to CPU with int8 quantisation.
    """
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        logger.info(f"Loaded Whisper '{model_size}' on CUDA ({compute_type})")
    except Exception as exc:
        if compute_type != "int8":
            logger.warning(f"CUDA {compute_type} unavailable ({exc}), retrying with int8 on CUDA")
            try:
                model = WhisperModel(model_size, device="cuda", compute_type="int8")
                logger.info(f"Loaded Whisper '{model_size}' on CUDA (int8 fallback)")
            except Exception as exc2:
                logger.warning(f"CUDA unavailable ({exc2}), falling back to CPU (int8)")
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                logger.info(f"Loaded Whisper '{model_size}' on CPU (int8)")
        else:
            logger.warning(f"CUDA unavailable ({exc}), falling back to CPU (int8)")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info(f"Loaded Whisper '{model_size}' on CPU (int8)")

    return model


def transcribe(
    wav_path: Path,
    model: "WhisperModel",
    glossary: str = "",
    language: str = "en",
    progress_callback=None,
) -> list[WordTimestamp]:
    """Transcribe a WAV file and return word-level timestamps.

    Passes the glossary as ``initial_prompt`` to bias the decoder toward BJJ
    terminology. VAD filtering is enabled to skip silent segments.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    logger.info(f"Transcribing {wav_path.name} ...")

    def _run(m):
        segs, run_info = m.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=True,
            initial_prompt=glossary or None,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        result: list[WordTimestamp] = []
        for segment in segs:
            if segment.words:
                for w in segment.words:
                    result.append(WordTimestamp(word=w.word, start=w.start, end=w.end))
            if progress_callback is not None and run_info.duration > 0:
                progress_callback(min(segment.end / run_info.duration, 1.0))
        return result, run_info

    try:
        words, info = _run(model)
    except RuntimeError as exc:
        if "cuda" in str(exc).lower() or "cublas" in str(exc).lower() or "cudnn" in str(exc).lower():
            logger.warning(f"CUDA error during transcription ({exc}), retrying on CPU")
            from faster_whisper import WhisperModel as _WM
            cpu_model = _WM(model.model_size_or_path, device="cpu", compute_type="int8")
            words, info = _run(cpu_model)
        else:
            raise

    logger.info(
        f"Transcribed {len(words)} words "
        f"(detected language: {info.language}, probability: {info.language_probability:.2f})"
    )
    return words


def load_glossary(glossary_path: Path) -> str:
    """Read the glossary file and return its contents as a single string."""
    if not glossary_path.exists():
        logger.warning(f"Glossary not found at {glossary_path}; transcribing without vocabulary priming")
        return ""
    return glossary_path.read_text(encoding="utf-8").strip()
