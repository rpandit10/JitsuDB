"""Audio extraction — converts video files to 16kHz mono WAV for Whisper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from loguru import logger


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from a video file as a 16kHz mono WAV.

    Uses ffmpeg: ``ffmpeg -y -i <video> -vn -acodec pcm_s16le -ar 16000 -ac 1 <wav>``
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]

    logger.debug(f"Extracting audio: {video_path.name} → {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting audio from {video_path.name}:\n{result.stderr}"
        )

    logger.debug(f"Audio extracted to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def verify_audio_stream(video_path: Path) -> bool:
    """Check whether the video file contains at least one audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "codec_type=audio" in result.stdout


def get_video_duration(video_path: Path) -> float:
    """Return the total duration of a video file in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {result.stderr}")

    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {result.stdout}") from exc
