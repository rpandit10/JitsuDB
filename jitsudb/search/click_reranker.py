"""Online click-through reranker.

Stores per-result impressions and click events in retrieval_log, then uses
accumulated click history to apply a small cosine-weighted score boost at
search time — no offline training required.

Usage flow:
  1. After engine.search(), call log_impression() to record all shown results.
  2. When the user plays a clip, call log_click() to mark that chunk clicked.
  3. On the next search, engine.search() calls boost() to re-order results.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jitsudb.models import SearchResult

_BOOST_WEIGHT = 0.15
_MAX_HISTORY  = 500


def log_impression(
    session_id: str,
    query_embedding: list[float],
    results: "list[SearchResult]",
    conn: sqlite3.Connection,
    latency_ms: float = 0.0,
    query_text: str = "",
) -> None:
    """Insert one retrieval_log row per result shown to the user."""
    emb_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO retrieval_log
            (session_id, query_text, query_embedding, chunk_id, rank, clicked, latency_ms, logged_at)
        VALUES (?, ?, ?, ?, ?, 0, ?, ?)
        """,
        [
            # Only store latency on the rank-0 row so percentile queries are simple.
            (session_id, query_text, emb_bytes, r.chunk.id, rank, latency_ms if rank == 0 else 0.0, now)
            for rank, r in enumerate(results)
        ],
    )
    conn.commit()


def log_click(
    session_id: str,
    chunk_id: str,
    conn: sqlite3.Connection,
) -> None:
    """Mark the most recent impression of chunk_id for this session as clicked."""
    conn.execute(
        """
        UPDATE retrieval_log
        SET clicked = 1
        WHERE id = (
            SELECT id FROM retrieval_log
            WHERE session_id = ? AND chunk_id = ?
            ORDER BY logged_at DESC
            LIMIT 1
        )
        """,
        (session_id, chunk_id),
    )
    conn.commit()


def boost(
    query_embedding: list[float],
    results: "list[SearchResult]",
    conn: sqlite3.Connection,
) -> "list[SearchResult]":
    """Re-score results using click-weighted cosine similarity to past queries.

    For each candidate chunk, accumulates similarity-weighted click counts
    across recent history, then adds a normalised boost capped at
    _BOOST_WEIGHT to the existing relevance_score. Ties in boosted score
    maintain the original ranking order.
    """
    rows = conn.execute(
        """
        SELECT query_embedding, chunk_id
        FROM retrieval_log
        WHERE clicked = 1
        ORDER BY logged_at DESC
        LIMIT ?
        """,
        (_MAX_HISTORY,),
    ).fetchall()

    if not rows:
        return results

    q = np.array(query_embedding, dtype=np.float32)
    chunk_scores: dict[str, float] = {}

    for row in rows:
        past_q = np.frombuffer(row["query_embedding"], dtype=np.float32)
        if past_q.shape != q.shape:
            continue
        sim = float(np.dot(q, past_q))
        if sim <= 0:
            continue
        cid = row["chunk_id"]
        chunk_scores[cid] = chunk_scores.get(cid, 0.0) + sim

    if not chunk_scores:
        return results

    max_score = max(chunk_scores.values())
    for r in results:
        raw = chunk_scores.get(r.chunk.id, 0.0)
        r.relevance_score += _BOOST_WEIGHT * (raw / max_score)

    results.sort(key=lambda r: r.relevance_score, reverse=True)
    return results
