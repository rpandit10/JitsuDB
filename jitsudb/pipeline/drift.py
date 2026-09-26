"""Taxonomy drift detection — flags videos whose embeddings diverge from category centroids.

Drift = new chunks whose embeddings are far from the mean of existing chunks
in the same primary_category, suggesting the video covers content the taxonomy
hasn't seen before.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import chromadb

    from jitsudb.models import Chunk

# Cosine distance above this threshold triggers a drift_log entry.
DRIFT_THRESHOLD = 0.45
# Minimum existing chunks per category to compute a reliable centroid.
MIN_BASELINE_CHUNKS = 10


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def _category_centroid(
    category: str,
    exclude_video_id: str,
    collection: "chromadb.Collection",
    conn: sqlite3.Connection,
) -> np.ndarray | None:
    """Fetch embeddings for existing chunks in `category` (excluding the current video)
    and return their mean, or None if fewer than MIN_BASELINE_CHUNKS exist.
    """
    rows = conn.execute(
        """
        SELECT c.id FROM chunks c
        JOIN videos v ON v.id = c.video_id
        WHERE v.primary_category = ? AND c.video_id != ?
        LIMIT 500
        """,
        (category, exclude_video_id),
    ).fetchall()

    if len(rows) < MIN_BASELINE_CHUNKS:
        return None

    ids = [r["id"] for r in rows]
    result = collection.get(ids=ids, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None

    matrix = np.array(embeddings, dtype=np.float32)
    return matrix.mean(axis=0)


def detect_drift(
    video_id: str,
    category: str,
    chunks: list["Chunk"],
    collection: "chromadb.Collection",
    conn: sqlite3.Connection,
) -> float | None:
    """Compute mean cosine distance of `chunks` from the category centroid.

    Returns the mean distance, or None if baseline is too small to judge.
    Writes to drift_log if the distance exceeds DRIFT_THRESHOLD.
    """
    if not chunks:
        return None

    centroid = _category_centroid(category, video_id, collection, conn)
    if centroid is None:
        return None

    chunk_ids = [c.id for c in chunks]
    result = collection.get(ids=chunk_ids, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None

    distances = [_cosine_distance(centroid, np.array(e, dtype=np.float32)) for e in embeddings]
    mean_dist = float(np.mean(distances))

    now = datetime.now(timezone.utc).isoformat()
    if mean_dist > DRIFT_THRESHOLD:
        conn.execute(
            """
            INSERT INTO drift_log (taxonomy_class, distance, baseline_at, detected_at)
            VALUES (?, ?, ?, ?)
            """,
            (category, mean_dist, now, now),
        )
        conn.commit()

    return mean_dist
