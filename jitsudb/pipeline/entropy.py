"""Classification entropy scoring for uncertain chunk detection.

Entropy measures how ambiguous a chunk's taxonomy assignment is:
- 0.0 = highly confident (exactly one label per category)
- 1.0 = maximally uncertain (no matches or too many matches)
"""

from __future__ import annotations

import math
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jitsudb.models import Chunk

# Taxonomy sizes — used for log2 normalisation
_N_POSITIONS = 14
_N_INTENTS = 7
_N_TECHNIQUES = 39


def _category_entropy(n_matches: int, n_total: int) -> float:
    """Shannon entropy for a single taxonomy category, normalised to [0, 1].

    - 0 matches → 1.0  (no information, cannot classify)
    - 1 match   → 0.0  (certain)
    - k matches → log2(k) / log2(n_total)  (grows toward 1 as ambiguity increases)
    """
    if n_matches == 0:
        return 1.0
    if n_matches == 1:
        return 0.0
    return math.log2(n_matches) / math.log2(n_total)


def compute_chunk_entropy(chunk: "Chunk") -> float:
    """Return a [0, 1] entropy score for a classified chunk.

    Combines position, intent, technique, and context uncertainty.
    """
    pos_e  = _category_entropy(len(chunk.position),   _N_POSITIONS)
    int_e  = _category_entropy(len(chunk.intent),     _N_INTENTS)
    tech_e = _category_entropy(len(chunk.techniques), _N_TECHNIQUES)
    # context "neutral" or None is ambiguous; clear top/bottom/standing is not
    ctx_e  = 0.0 if chunk.context in ("top", "bottom", "standing") else 0.5
    return (pos_e + int_e + tech_e + ctx_e) / 4.0


def upsert_uncertain_chunk(chunk: "Chunk", entropy: float, conn: sqlite3.Connection) -> None:
    """Write or update an uncertain_chunks row for a chunk above the entropy threshold."""
    top_label    = chunk.position[0]    if chunk.position    else (chunk.intent[0] if chunk.intent else "none")
    second_label = chunk.position[1]    if len(chunk.position) > 1 else None

    conn.execute(
        """
        INSERT OR REPLACE INTO uncertain_chunks
            (chunk_id, entropy, top_label, second_label, reviewed)
        VALUES (?, ?, ?, ?, 0)
        """,
        (chunk.id, entropy, top_label, second_label),
    )


def score_and_store_chunks(
    chunks: list["Chunk"],
    conn: sqlite3.Connection,
    entropy_threshold: float = 0.35,
) -> int:
    """Compute entropy for each chunk; store high-entropy ones in uncertain_chunks.

    Returns the number of uncertain chunks written.
    """
    written = 0
    for chunk in chunks:
        entropy = compute_chunk_entropy(chunk)
        if entropy >= entropy_threshold:
            upsert_uncertain_chunk(chunk, entropy, conn)
            written += 1
        else:
            # Remove from uncertain if it was previously flagged (e.g. after re-classify)
            conn.execute("DELETE FROM uncertain_chunks WHERE chunk_id = ?", (chunk.id,))
    return written


def backfill_entropy(conn: sqlite3.Connection, entropy_threshold: float = 0.35) -> int:
    """Backfill entropy scores for all existing chunks that have no uncertain_chunks row.

    Returns number of rows written.
    """
    from jitsudb.models import Chunk

    chunk_rows = conn.execute(
        """
        SELECT c.id, c.video_id, c.start_time, c.end_time, c.transcript_text,
               c.context, c.gi_nogi
        FROM chunks c
        WHERE c.id NOT IN (SELECT chunk_id FROM uncertain_chunks)
        """
    ).fetchall()

    # Load taxonomy labels for all chunks in one pass
    all_labels = conn.execute(
        "SELECT chunk_id, label_type, label_value FROM taxonomy_labels"
    ).fetchall()
    label_map: dict[str, dict[str, list[str]]] = {}
    for row in all_labels:
        entry = label_map.setdefault(row["chunk_id"], {"position": [], "intent": [], "technique": []})
        entry[row["label_type"]].append(row["label_value"])

    written = 0
    for row in chunk_rows:
        labels = label_map.get(row["id"], {"position": [], "intent": [], "technique": []})
        chunk = Chunk(
            id=row["id"],
            video_id=row["video_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            transcript_text=row["transcript_text"],
            context=row["context"],
            gi_nogi=row["gi_nogi"],
            position=labels["position"],
            intent=labels["intent"],
            techniques=labels["technique"],
        )
        entropy = compute_chunk_entropy(chunk)
        if entropy >= entropy_threshold:
            upsert_uncertain_chunk(chunk, entropy, conn)
            written += 1

    conn.commit()
    return written
