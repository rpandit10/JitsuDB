"""GET/PUT /api/queue — processing queue CRUD."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from jitsudb.api.dependencies import get_db
from jitsudb.models import QueueItem

router = APIRouter(tags=["queue"])


class EnqueueRequest(BaseModel):
    """Request body for adding videos to the processing queue."""

    video_ids: list[str]
    priority: int = 0


class ReorderRequest(BaseModel):
    """Request body for updating a single video's queue priority."""

    priority: int


def _fetch_queue(db: sqlite3.Connection) -> list[QueueItem]:
    rows = db.execute(
        "SELECT * FROM processing_queue ORDER BY priority ASC"
    ).fetchall()
    return [QueueItem(**dict(r)) for r in rows]


@router.get("/queue", response_model=list[QueueItem])
async def list_queue(db: sqlite3.Connection = Depends(get_db)) -> list[QueueItem]:
    """Return all queue items ordered by priority ascending."""
    return _fetch_queue(db)


@router.post("/queue", response_model=list[QueueItem])
async def enqueue_videos(
    body: EnqueueRequest,
    db: sqlite3.Connection = Depends(get_db),
) -> list[QueueItem]:
    """Add one or more videos to the processing queue.

    Videos that are already queued are skipped (idempotent).
    """
    for video_id in body.video_ids:
        row = db.execute("SELECT id FROM videos WHERE id = ?", (video_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

        db.execute(
            """
            INSERT OR IGNORE INTO processing_queue (video_id, priority, status)
            VALUES (?, ?, 'pending')
            """,
            (video_id, body.priority),
        )
    db.commit()
    return _fetch_queue(db)


@router.put("/queue/{video_id}/priority", response_model=QueueItem)
async def update_priority(
    video_id: str,
    body: ReorderRequest,
    db: sqlite3.Connection = Depends(get_db),
) -> QueueItem:
    """Update the priority of a queued video (used for drag-to-reorder)."""
    row = db.execute(
        "SELECT * FROM processing_queue WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Video not in queue")

    db.execute(
        "UPDATE processing_queue SET priority = ? WHERE video_id = ?",
        (body.priority, video_id),
    )
    db.commit()

    updated = db.execute(
        "SELECT * FROM processing_queue WHERE video_id = ?", (video_id,)
    ).fetchone()
    return QueueItem(**dict(updated))


@router.delete("/queue")
async def clear_queue(db: sqlite3.Connection = Depends(get_db)) -> Response:
    """Remove all items from the processing queue."""
    db.execute("DELETE FROM processing_queue")
    db.commit()
    return Response(status_code=204)


@router.delete("/queue/{video_id}")
async def remove_from_queue(
    video_id: str,
    db: sqlite3.Connection = Depends(get_db),
) -> Response:
    """Remove a video from the processing queue."""
    row = db.execute(
        "SELECT video_id FROM processing_queue WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Video not in queue")

    db.execute("DELETE FROM processing_queue WHERE video_id = ?", (video_id,))
    db.commit()
    return Response(status_code=204)
