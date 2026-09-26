"""GET /api/labels/review, POST /api/labels/{chunk_id}, GET /api/labels/drift."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from jitsudb.api.dependencies import get_db

router = APIRouter(tags=["labels"])


@router.get("/labels/review")
async def get_review_queue(
    limit: int = 20,
    db: sqlite3.Connection = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return the highest-entropy unreviewed chunks, enriched with video context."""
    rows = db.execute(
        """
        SELECT uc.chunk_id, uc.entropy, uc.top_label, uc.second_label,
               c.transcript_text, c.start_time, c.end_time, c.context, c.gi_nogi,
               v.id AS video_id, v.title AS video_title, v.instructor
        FROM uncertain_chunks uc
        JOIN chunks c ON c.id = uc.chunk_id
        JOIN videos v ON v.id = c.video_id
        WHERE uc.reviewed = 0
        ORDER BY uc.entropy DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    results = []
    for row in rows:
        labels = db.execute(
            "SELECT label_type, label_value FROM taxonomy_labels WHERE chunk_id = ?",
            (row["chunk_id"],),
        ).fetchall()
        positions  = [r["label_value"] for r in labels if r["label_type"] == "position"]
        intents    = [r["label_value"] for r in labels if r["label_type"] == "intent"]
        techniques = [r["label_value"] for r in labels if r["label_type"] == "technique"]

        results.append({
            "chunk_id":      row["chunk_id"],
            "entropy":       round(row["entropy"], 4),
            "transcript":    row["transcript_text"],
            "start_time":    row["start_time"],
            "end_time":      row["end_time"],
            "context":       row["context"],
            "gi_nogi":       row["gi_nogi"],
            "video_id":      row["video_id"],
            "video_title":   row["video_title"],
            "instructor":    row["instructor"],
            "positions":     positions,
            "intents":       intents,
            "techniques":    techniques,
        })
    return results


@router.get("/labels/stats")
async def label_stats(db: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    """Return summary stats for the labeling queue."""
    total    = db.execute("SELECT COUNT(*) FROM uncertain_chunks").fetchone()[0]
    pending  = db.execute("SELECT COUNT(*) FROM uncertain_chunks WHERE reviewed = 0").fetchone()[0]
    reviewed = total - pending
    drift    = db.execute("SELECT COUNT(*) FROM drift_log").fetchone()[0]
    avg_entropy = db.execute(
        "SELECT AVG(entropy) FROM uncertain_chunks WHERE reviewed = 0"
    ).fetchone()[0]
    return {
        "total":       total,
        "pending":     pending,
        "reviewed":    reviewed,
        "drift_events": drift,
        "avg_entropy": round(avg_entropy or 0.0, 4),
    }


class LabelCorrection(BaseModel):
    positions:  list[str] = []
    intents:    list[str] = []
    techniques: list[str] = []
    context:    str | None = None
    gi_nogi:    str | None = None
    approved:   bool = False


@router.post("/labels/{chunk_id}")
async def save_label(
    chunk_id: str,
    correction: LabelCorrection,
    db: sqlite3.Connection = Depends(get_db),
) -> dict[str, str]:
    """Save corrected labels for a chunk and mark it as reviewed."""
    row = db.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    if not correction.approved:
        db.execute("DELETE FROM taxonomy_labels WHERE chunk_id = ?", (chunk_id,))
        for label in correction.positions:
            db.execute(
                "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?, 'position', ?)",
                (chunk_id, label),
            )
        for label in correction.intents:
            db.execute(
                "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?, 'intent', ?)",
                (chunk_id, label),
            )
        for label in correction.techniques:
            db.execute(
                "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?, 'technique', ?)",
                (chunk_id, label),
            )
        # Update context / gi_nogi on the chunk row if provided
        if correction.context is not None:
            db.execute("UPDATE chunks SET context = ? WHERE id = ?", (correction.context, chunk_id))
        if correction.gi_nogi is not None:
            db.execute("UPDATE chunks SET gi_nogi = ? WHERE id = ?", (correction.gi_nogi, chunk_id))

    db.execute(
        "UPDATE uncertain_chunks SET reviewed = 1 WHERE chunk_id = ?", (chunk_id,)
    )
    db.commit()
    return {"status": "ok"}


@router.get("/labels/drift")
async def get_drift_log(
    limit: int = 50,
    db: sqlite3.Connection = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return recent drift events ordered by distance descending."""
    rows = db.execute(
        """
        SELECT id, taxonomy_class, distance, detected_at
        FROM drift_log
        ORDER BY detected_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


@router.post("/labels/backfill")
async def backfill_entropy(db: sqlite3.Connection = Depends(get_db)) -> dict[str, int]:
    """Compute entropy for all existing chunks that lack an uncertain_chunks row."""
    from jitsudb.pipeline.entropy import backfill_entropy as _backfill
    written = _backfill(db)
    return {"written": written}


@router.post("/labels/cluster")
async def trigger_clustering(db: sqlite3.Connection = Depends(get_db)) -> dict[str, int]:
    """Run HDBSCAN over all chunk embeddings and populate chunk_label_proposals."""
    from jitsudb.api.dependencies import get_search_engine
    from jitsudb.pipeline.cluster import run_cluster_proposals

    engine = get_search_engine()
    written = run_cluster_proposals(engine.collection, db)
    return {"written": written}


@router.get("/labels/propose/{chunk_id}")
async def get_label_proposals(
    chunk_id: str,
    db: sqlite3.Connection = Depends(get_db),
) -> dict:
    """Return HDBSCAN-derived label proposals for a chunk, or 404 if unavailable."""
    from jitsudb.pipeline.cluster import get_proposals

    row = db.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    proposals = get_proposals(chunk_id, db)
    if proposals is None:
        raise HTTPException(status_code=404, detail="No proposals — run /labels/cluster first")

    return proposals
