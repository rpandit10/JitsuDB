"""Re-run the taxonomy classifier on all indexed chunks.

Usage:
    python scripts/reclassify.py [--dry-run]

Run this after updating jitsudb/taxonomy/dictionary.py and bumping
TAXONOMY_VERSION in constants.py. It re-classifies every chunk in the
database and updates the taxonomy_labels table and schema_version record.
Does NOT re-transcribe or re-embed — only taxonomy labels are updated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from jitsudb.constants import TAXONOMY_VERSION
from jitsudb.db import get_connection, init_db
from jitsudb.models import Chunk, Video
from jitsudb.pipeline.classifier import classify_chunk


def reclassify(dry_run: bool = False) -> None:
    """Re-classify all chunks using the current taxonomy dictionary."""
    if not dry_run:
        init_db()

    conn = get_connection()

    chunks_rows = conn.execute("SELECT * FROM chunks").fetchall()
    logger.info(f"Reclassifying {len(chunks_rows)} chunks (taxonomy version {TAXONOMY_VERSION})")

    for row in chunks_rows:
        video_row = conn.execute("SELECT * FROM videos WHERE id = ?", (row["video_id"],)).fetchone()
        if not video_row:
            logger.warning(f"Orphaned chunk {row['id']} — no parent video found, skipping")
            continue

        video = Video(
            id=video_row["id"],
            title=video_row["title"],
            instructor=video_row["instructor"],
            file_path=Path(video_row["file_path"]),
            duration_seconds=video_row["duration_seconds"],
            gi_nogi=video_row["gi_nogi"],
            primary_category=video_row["primary_category"],
            status=video_row["status"],
            added_at=video_row["added_at"],
        )

        # Reconstruct a minimal Chunk for classify_chunk input
        from jitsudb.models import RawChunk
        raw = RawChunk(
            start_time=row["start_time"],
            end_time=row["end_time"],
            words=[],
            transcript_text=row["transcript_text"],
        )

        classified: Chunk = classify_chunk(raw, video)

        if dry_run:
            logger.info(
                f"[dry-run] chunk {row['id'][:8]}… → "
                f"positions={classified.position} intents={classified.intent} "
                f"context={classified.context}"
            )
            continue

        # Clear old labels and re-insert
        conn.execute("DELETE FROM taxonomy_labels WHERE chunk_id = ?", (row["id"],))
        for label in classified.position:
            conn.execute(
                "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?, ?, ?)",
                (row["id"], "position", label),
            )
        for label in classified.intent:
            conn.execute(
                "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?, ?, ?)",
                (row["id"], "intent", label),
            )
        for label in classified.techniques:
            conn.execute(
                "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?, ?, ?)",
                (row["id"], "technique", label),
            )
        if classified.context:
            conn.execute(
                "UPDATE chunks SET context = ? WHERE id = ?",
                (classified.context, row["id"]),
            )

    if not dry_run:
        conn.execute(
            "INSERT OR REPLACE INTO schema_version(key, value) VALUES ('taxonomy_version', ?)",
            (TAXONOMY_VERSION,),
        )
        conn.commit()
        logger.info(f"Reclassification complete. taxonomy_version updated to {TAXONOMY_VERSION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-run BJJ taxonomy classifier on all chunks")
    parser.add_argument("--dry-run", action="store_true", help="Log changes without modifying the DB")
    args = parser.parse_args()
    reclassify(dry_run=args.dry_run)
