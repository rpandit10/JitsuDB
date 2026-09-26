"""POST /api/feedback/impression, POST /api/feedback/click, GET /api/feedback/stats."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from jitsudb.api.dependencies import get_db

router = APIRouter(tags=["feedback"])


class ImpressionPayload(BaseModel):
    session_id: str
    query_embedding: list[float]
    chunk_ids: list[str]


class ClickPayload(BaseModel):
    session_id: str
    chunk_id: str


@router.post("/feedback/impression", response_class=Response, status_code=204)
async def record_impression(
    payload: ImpressionPayload,
    db: sqlite3.Connection = Depends(get_db),
) -> Response:
    """Log one retrieval_log row per chunk returned in a search result page."""
    if not payload.chunk_ids:
        return

    emb_bytes = np.array(payload.query_embedding, dtype=np.float32).tobytes()
    now = datetime.now(timezone.utc).isoformat()

    placeholders = ",".join("?" * len(payload.chunk_ids))
    existing = {
        row["id"]
        for row in db.execute(
            f"SELECT id FROM chunks WHERE id IN ({placeholders})", payload.chunk_ids
        ).fetchall()
    }

    db.executemany(
        """
        INSERT INTO retrieval_log
            (session_id, query_embedding, chunk_id, rank, clicked, logged_at)
        VALUES (?, ?, ?, ?, 0, ?)
        """,
        [
            (payload.session_id, emb_bytes, cid, rank, now)
            for rank, cid in enumerate(payload.chunk_ids)
            if cid in existing
        ],
    )
    db.commit()
    return Response(status_code=204)


@router.post("/feedback/click", response_class=Response, status_code=204)
async def record_click(
    payload: ClickPayload,
    db: sqlite3.Connection = Depends(get_db),
) -> Response:
    """Mark the most recent impression of chunk_id for this session as clicked."""
    row = db.execute("SELECT id FROM chunks WHERE id = ?", (payload.chunk_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    from jitsudb.search.click_reranker import log_click
    log_click(payload.session_id, payload.chunk_id, db)
    return Response(status_code=204)


@router.get("/feedback/stats")
async def feedback_stats(db: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    """Return aggregate click-through statistics."""
    total_impressions = db.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0]
    total_clicks      = db.execute("SELECT COUNT(*) FROM retrieval_log WHERE clicked = 1").fetchone()[0]
    unique_sessions   = db.execute("SELECT COUNT(DISTINCT session_id) FROM retrieval_log").fetchone()[0]
    top_chunks = db.execute(
        """
        SELECT chunk_id, COUNT(*) AS clicks
        FROM retrieval_log WHERE clicked = 1
        GROUP BY chunk_id ORDER BY clicks DESC LIMIT 10
        """
    ).fetchall()
    ctr = round(total_clicks / total_impressions, 4) if total_impressions else 0.0
    return {
        "total_impressions": total_impressions,
        "total_clicks":      total_clicks,
        "unique_sessions":   unique_sessions,
        "ctr":               ctr,
        "top_chunks": [{"chunk_id": r["chunk_id"], "clicks": r["clicks"]} for r in top_chunks],
    }
