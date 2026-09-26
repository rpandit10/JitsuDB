"""GET /api/thumbnail and /api/preview — media file serving."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from jitsudb.api.dependencies import get_db
from jitsudb.config import get_config
from jitsudb.models import Chunk
from jitsudb.pipeline.thumbnails import get_or_generate_clip

router = APIRouter(tags=["media"])


def _load_chunk(chunk_id: str, db: sqlite3.Connection) -> sqlite3.Row:
    row = db.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return row


@router.get("/thumbnail/{chunk_id}")
async def serve_thumbnail(
    chunk_id: str,
    db: sqlite3.Connection = Depends(get_db),
) -> FileResponse:
    """Serve the thumbnail JPEG for a chunk."""
    row = _load_chunk(chunk_id, db)
    if not row["thumbnail_path"]:
        raise HTTPException(status_code=404, detail="Thumbnail not generated yet")

    thumb_path = Path(row["thumbnail_path"])
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file not found on disk")

    return FileResponse(str(thumb_path), media_type="image/jpeg")


@router.get("/preview/{chunk_id}")
async def serve_preview(
    chunk_id: str,
    db: sqlite3.Connection = Depends(get_db),
) -> FileResponse:
    """Generate (or return cached) a clip preview for a chunk."""
    chunk_row = _load_chunk(chunk_id, db)

    video_row = db.execute(
        "SELECT * FROM videos WHERE id = ?", (chunk_row["video_id"],)
    ).fetchone()
    if video_row is None:
        raise HTTPException(status_code=404, detail="Source video not found")

    video_path = Path(video_row["file_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    cfg = get_config()
    clips_dir = cfg.paths.data_dir / "clips"

    chunk = Chunk(
        id=chunk_row["id"],
        video_id=chunk_row["video_id"],
        start_time=chunk_row["start_time"],
        end_time=chunk_row["end_time"],
        transcript_text=chunk_row["transcript_text"],
    )

    try:
        clip_path = get_or_generate_clip(
            chunk=chunk,
            video_path=video_path,
            clips_dir=clips_dir,
            duration_seconds=cfg.clips.duration_seconds,
            padding_seconds=cfg.clips.padding_seconds,
            width=cfg.clips.width,
            max_cache_count=cfg.clips.max_cache_count,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return FileResponse(str(clip_path), media_type="video/mp4")
