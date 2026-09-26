"""GET /api/videos — list and manage indexed videos."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from jitsudb.api.dependencies import get_db
from jitsudb.models import Video

router = APIRouter(tags=["videos"])


def _row_to_video(row: sqlite3.Row) -> Video:
    return Video(**dict(row))


@router.get("/videos", response_model=list[Video])
async def list_videos(db: sqlite3.Connection = Depends(get_db)) -> list[Video]:
    """Return all videos in the library ordered by added_at descending."""
    rows = db.execute("SELECT * FROM videos ORDER BY added_at DESC").fetchall()
    return [_row_to_video(r) for r in rows]


@router.get("/videos/{video_id}", response_model=Video)
async def get_video(video_id: str, db: sqlite3.Connection = Depends(get_db)) -> Video:
    """Return a single video by ID."""
    row = db.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return _row_to_video(row)
