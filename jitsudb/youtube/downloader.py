"""yt-dlp wrapper for downloading YouTube instructionals."""

from __future__ import annotations

from pathlib import Path


def extract_metadata(url: str) -> dict:
    """Fetch video metadata without downloading."""
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract metadata: {exc}") from exc

    return {
        "title": info.get("title", "Unknown"),
        "uploader": info.get("uploader") or info.get("channel", ""),
        "duration_seconds": float(info.get("duration") or 0),
        "thumbnail_url": info.get("thumbnail", ""),
        "video_id": info.get("id", ""),
    }


def download_video(
    url: str,
    output_dir: Path,
    format_str: str,
    progress_hook: "callable | None" = None,
) -> Path:
    """Download a YouTube video using yt-dlp."""
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts: dict = {
        "format": format_str,
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    if progress_hook is not None:
        ydl_opts["progress_hooks"] = [progress_hook]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise RuntimeError(f"yt-dlp download failed: {exc}") from exc

    video_id = info["id"]
    # merge_output_format='mp4' → file is always <video_id>.mp4
    candidate = output_dir / f"{video_id}.mp4"
    if candidate.exists():
        return candidate

    # Fallback: find whichever file yt-dlp actually wrote
    for path in output_dir.glob(f"{video_id}.*"):
        return path

    raise RuntimeError(f"Downloaded file not found in {output_dir} for id={video_id!r}")
