"""ffmpeg thumbnail extraction and on-demand clip generation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger

from jitsudb.models import Chunk


def generate_thumbnail(
    chunk: Chunk,
    video_path: Path,
    output_dir: Path,
    width: int = 320,
    quality: int = 5,
    offset_seconds: float = 5.0,
) -> Path:
    """Extract a single frame from a video at the chunk's start_time + offset.

    Uses: ``ffmpeg -ss <ts> -i <video> -vframes 1 -vf scale=<w>:-1 -q:v <q> <out.jpg>``

    The ``-ss`` flag is placed before ``-i`` for fast seeking (keyframe-level),
    which is acceptable for thumbnail generation.
    """
    timestamp = max(0.0, chunk.start_time + offset_seconds)
    out_path = output_dir / chunk.video_id / f"{chunk.id}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", f"scale={width}:-1",
        "-q:v", str(quality),
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg thumbnail failed for chunk {chunk.id} at {timestamp:.1f}s:\n{result.stderr}"
        )

    return out_path


def generate_thumbnails_batch(
    chunks: list[Chunk],
    video_path: Path,
    output_dir: Path,
    width: int = 320,
    quality: int = 5,
    offset_seconds: float = 5.0,
) -> list[Path]:
    """Generate thumbnails for a list of chunks from the same video.

    Skips chunks whose thumbnail file already exists (resume-safe).
    """
    paths: list[Path] = []
    for chunk in chunks:
        expected_path = output_dir / chunk.video_id / f"{chunk.id}.jpg"
        if expected_path.exists():
            logger.debug(f"Thumbnail already exists, skipping: {expected_path.name}")
            paths.append(expected_path)
            continue

        try:
            path = generate_thumbnail(chunk, video_path, output_dir, width, quality, offset_seconds)
            paths.append(path)
        except RuntimeError as exc:
            logger.error(f"Thumbnail generation failed for chunk {chunk.id}: {exc}")
            paths.append(expected_path)  # placeholder — file won't exist

    return paths


def get_or_generate_clip(
    chunk: Chunk,
    video_path: Path,
    clips_dir: Path,
    duration_seconds: int = 15,
    padding_seconds: int = 3,
    width: int = 640,
    max_cache_count: int = 100,
) -> Path:
    """Return a cached clip or generate it on demand, evicting the oldest if needed.

    Uses: ``ffmpeg -ss <start-pad> -i <video> -t <dur> -vf scale=<w>:-1
              -c:v libx264 -preset ultrafast -c:a aac <out.mp4>``
    """
    clips_dir.mkdir(parents=True, exist_ok=True)
    out_path = clips_dir / f"{chunk.id}.mp4"

    if out_path.exists():
        return out_path

    _evict_oldest_clips(clips_dir, max_cache_count)

    start = max(0.0, chunk.start_time - padding_seconds)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration_seconds),
        "-vf", f"scale={width}:-1",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        str(out_path),
    ]

    logger.debug(f"Generating clip for chunk {chunk.id} at {start:.1f}s")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg clip generation failed for chunk {chunk.id}:\n{result.stderr}"
        )

    return out_path


def _evict_oldest_clips(clips_dir: Path, max_cache_count: int) -> None:
    """Delete the oldest clip files if the cache exceeds max_cache_count."""
    clips = sorted(clips_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    while len(clips) >= max_cache_count:
        oldest = clips.pop(0)
        logger.debug(f"Evicting clip from cache: {oldest.name}")
        oldest.unlink(missing_ok=True)
