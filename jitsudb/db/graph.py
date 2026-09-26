"""Temporal chunk graph — edge construction and neighbour traversal."""

from __future__ import annotations

import sqlite3
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    import chromadb

    from jitsudb.models import Chunk


def build_edges_for_video(
    video_id: str,
    chunks: list["Chunk"],
    collection: "chromadb.Collection",
    conn: sqlite3.Connection | None = None,
) -> int:
    """Write directed temporal edges for every adjacent chunk pair in a video.

    Fetches embeddings from the ChromaDB collection, computes cosine similarity
    between each consecutive pair (sorted by start_time), then upserts rows into
    ``chunk_edges``.  Any previous edges for this video are deleted first so the
    operation is safe to re-run after a re-index.  Returns the number of edges
    written, or 0 if fewer than two chunks have embeddings.
    """
    if len(chunks) < 2:
        return 0

    if conn is None:
        from jitsudb.db import get_connection
        conn = get_connection()

    ordered = sorted(chunks, key=lambda c: c.start_time)
    chunk_ids = [c.id for c in ordered]

    # Fetch embeddings in one round-trip; ChromaDB may return them in any order.
    result = collection.get(ids=chunk_ids, include=["embeddings"])
    emb_map: dict[str, np.ndarray] = {}
    embeddings = result.get("embeddings")
    if embeddings is not None and len(embeddings) > 0:
        for cid, emb in zip(result["ids"], embeddings):
            emb_map[cid] = np.array(emb, dtype=np.float32)

    if len(emb_map) < 2:
        logger.warning(
            f"graph: only {len(emb_map)} embeddings found for video {video_id}; "
            "skipping edge construction"
        )
        return 0

    conn.execute("DELETE FROM chunk_edges WHERE video_id = ?", (video_id,))

    edges_written = 0
    for i in range(len(ordered) - 1):
        src = ordered[i]
        dst = ordered[i + 1]

        src_emb = emb_map.get(src.id)
        dst_emb = emb_map.get(dst.id)
        if src_emb is None or dst_emb is None:
            continue

        gap = max(0.0, dst.start_time - src.end_time)
        score = _cosine_similarity(src_emb, dst_emb)

        conn.execute(
            """
            INSERT OR REPLACE INTO chunk_edges
                (src_chunk_id, dst_chunk_id, video_id, temporal_gap_sec, continuity_score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (src.id, dst.id, video_id, gap, score),
        )
        edges_written += 1

    conn.commit()
    logger.debug(f"graph: wrote {edges_written} edges for video {video_id}")
    return edges_written


def get_neighbours(
    chunk_id: str,
    direction: str = "both",
    max_hops: int = 2,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """BFS traversal through chunk_edges from a starting chunk.

    ``direction`` is ``"next"`` (forward edges), ``"prev"`` (backward edges) or
    ``"both"``.  Yields dicts of ``chunk_id``, ``hop`` and ``direction``; the
    seed chunk itself is excluded.
    """
    if direction not in ("next", "prev", "both"):
        raise ValueError(f"direction must be 'next', 'prev', or 'both'; got {direction!r}")

    if conn is None:
        from jitsudb.db import get_connection
        conn = get_connection()

    visited: set[str] = {chunk_id}
    queue: deque[tuple[str, int]] = deque([(chunk_id, 0)])
    results: list[dict] = []

    while queue:
        current, hop = queue.popleft()
        if hop >= max_hops:
            continue

        if direction in ("next", "both"):
            rows = conn.execute(
                "SELECT dst_chunk_id FROM chunk_edges WHERE src_chunk_id = ?",
                (current,),
            ).fetchall()
            for row in rows:
                nid = row[0]
                if nid not in visited:
                    visited.add(nid)
                    results.append({"chunk_id": nid, "hop": hop + 1, "direction": "next"})
                    queue.append((nid, hop + 1))

        if direction in ("prev", "both"):
            rows = conn.execute(
                "SELECT src_chunk_id FROM chunk_edges WHERE dst_chunk_id = ?",
                (current,),
            ).fetchall()
            for row in rows:
                nid = row[0]
                if nid not in visited:
                    visited.add(nid)
                    results.append({"chunk_id": nid, "hop": hop + 1, "direction": "prev"})
                    queue.append((nid, hop + 1))

    return results


def get_contiguous_window_cte(
    anchor_ids: list[str],
    beam_width: int = 2,
    conn: sqlite3.Connection | None = None,
) -> list[str]:
    """Expand anchor chunks using a single recursive CTE query.

    Walks ``chunk_edges`` in both directions up to ``beam_width`` hops from
    every anchor simultaneously, then returns all reached chunk IDs sorted by
    ``start_time``.  This replaces the per-anchor Python BFS loop with one
    SQL round-trip.
    """
    if not anchor_ids:
        return []

    if conn is None:
        from jitsudb.db import get_connection
        conn = get_connection()

    ph = ",".join("?" * len(anchor_ids))
    sql = f"""
        WITH RECURSIVE walk(chunk_id, hop) AS (
            SELECT id, 0 FROM chunks WHERE id IN ({ph})
            UNION
            SELECT ce.dst_chunk_id, walk.hop + 1
            FROM chunk_edges ce JOIN walk ON ce.src_chunk_id = walk.chunk_id
            WHERE walk.hop < ?
            UNION
            SELECT ce.src_chunk_id, walk.hop + 1
            FROM chunk_edges ce JOIN walk ON ce.dst_chunk_id = walk.chunk_id
            WHERE walk.hop < ?
        )
        SELECT DISTINCT c.id FROM walk
        JOIN chunks c ON c.id = walk.chunk_id
        ORDER BY c.start_time ASC
    """
    params = list(anchor_ids) + [beam_width, beam_width]
    rows = conn.execute(sql, params).fetchall()
    return [row[0] for row in rows]


def get_contiguous_window(
    anchor_ids: list[str],
    beam_width: int = 2,
    conn: sqlite3.Connection | None = None,
) -> list[str]:
    """Expand anchor chunks by ``beam_width`` hops in both directions.

    Returns deduplicated chunk IDs sorted by ``start_time`` (temporal order).
    If ``beam_width`` is 0, returns the anchors themselves sorted by start_time.
    """
    if not anchor_ids:
        return []

    if conn is None:
        from jitsudb.db import get_connection
        conn = get_connection()

    if beam_width == 0:
        ph = ",".join("?" * len(anchor_ids))
        rows = conn.execute(
            f"SELECT id FROM chunks WHERE id IN ({ph}) ORDER BY start_time ASC",
            list(anchor_ids),
        ).fetchall()
        return [row[0] for row in rows]

    return get_contiguous_window_cte(anchor_ids, beam_width=beam_width, conn=conn)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float32 arrays. Returns 0.0 for zero vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
