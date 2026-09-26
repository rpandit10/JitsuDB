"""Backfill CLIP visual embeddings for all chunks that already have thumbnails.

Run once after adding visual_embed stage to populate jitsudb_visual collection
for videos that completed the pipeline before this stage was added.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jitsudb.config import get_config
from jitsudb.db import get_connection, init_db
from jitsudb.models import Chunk
from jitsudb.pipeline.visual_embedder import embed_chunks_visual, get_visual_collection, load_clip_model


def main() -> None:
    init_db()
    cfg = get_config()
    conn = get_connection()

    rows = conn.execute(
        "SELECT * FROM chunks WHERE thumbnail_path IS NOT NULL"
    ).fetchall()
    print(f"Found {len(rows)} chunks with thumbnails")

    chunks = [
        Chunk(
            id=row["id"],
            video_id=row["video_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            transcript_text=row["transcript_text"],
            thumbnail_path=Path(row["thumbnail_path"]),
            gi_nogi=row["gi_nogi"],
            context=row["context"],
        )
        for row in rows
        if row["thumbnail_path"] and Path(row["thumbnail_path"]).exists()
    ]
    print(f"{len(chunks)} have existing thumbnail files")

    chroma_dir = cfg.paths.data_dir / "chromadb"
    clip_model = load_clip_model(cfg.visual_embedding.model_name, cfg.visual_embedding.device)
    visual_collection = get_visual_collection(chroma_dir)

    already = visual_collection.count()
    print(f"Visual collection already has {already} docs")

    n = embed_chunks_visual(chunks, clip_model, visual_collection)
    print(f"Done — embedded {n} thumbnails. Total in collection: {visual_collection.count()}")


if __name__ == "__main__":
    main()
