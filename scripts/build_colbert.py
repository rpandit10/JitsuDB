"""Build the ColBERT per-token embedding index from all indexed chunks.

ColBERT late-interaction rescoring requires a separate binary index of
per-token embeddings.  This script reads every chunk from SQLite and
writes their token-level embeddings into data/colbert/ so the search
engine can run MaxSim rescoring at query time.

Run this once after indexing, then again incrementally as new videos are
processed.  New chunks are appended without rebuilding existing entries.

Usage:
    python scripts/build_colbert.py
    python scripts/build_colbert.py --batch-size 16 --device cuda
    python scripts/build_colbert.py --rebuild   # wipe and rebuild from scratch

After running:
    Set colbert_enabled: true in config.yaml and restart the server.
"""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ColBERT token embedding index")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Chunks encoded per forward pass (default: 16)")
    parser.add_argument("--device", default=None,
                        help="cpu or cuda (default: reads from config.yaml)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete and rebuild the index from scratch")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only index the first N chunks (for testing)")
    args = parser.parse_args()

    from jitsudb.config import get_config
    from jitsudb.db import get_connection
    from jitsudb.models import Chunk
    from jitsudb.pipeline.embedder import load_embedding_model
    from jitsudb.retrieval.colbert_index import ColBERTIndex

    cfg = get_config()
    conn = get_connection()
    device = args.device or cfg.embedding.device
    colbert_dir = cfg.paths.data_dir / "colbert"

    if args.rebuild and colbert_dir.exists():
        print(f"Wiping existing index at {colbert_dir} …")
        shutil.rmtree(colbert_dir)

    print(f"ColBERT index dir : {colbert_dir}")
    print(f"Device            : {device}")
    print(f"Batch size        : {args.batch_size}")
    print()

    model_src = cfg.embedding.encoder_path or cfg.embedding.model_name
    print(f"Loading model: {model_src}")
    model = load_embedding_model(
        model_name=cfg.embedding.model_name,
        device=device,
        model_path=cfg.embedding.encoder_path,
        trust_remote_code=cfg.embedding.trust_remote_code,
    )

    # Open index (may already contain entries; we only process missing ones)
    index = ColBERTIndex(colbert_dir)
    already_indexed = set(index._meta.keys())
    print(f"Existing index   : {len(already_indexed)} chunks")

    rows = conn.execute(
        "SELECT id, video_id, start_time, end_time, transcript_text, context, gi_nogi "
        "FROM chunks"
    ).fetchall()

    if args.limit:
        rows = rows[:args.limit]

    # Skip already-indexed chunks (incremental mode)
    new_rows = [r for r in rows if r["id"] not in already_indexed]
    print(f"Chunks to index  : {len(new_rows)} new / {len(rows)} total")

    if not new_rows:
        print("Nothing to do — index is up to date.")
        return

    chunks = [
        Chunk(
            id=r["id"],
            video_id=r["video_id"],
            start_time=r["start_time"],
            end_time=r["end_time"],
            transcript_text=r["transcript_text"],
            context=r["context"],
            gi_nogi=r["gi_nogi"],
        )
        for r in new_rows
    ]

    # Prepend document prefix (nomic-embed-text needs this for asymmetric retrieval)
    doc_prefix = cfg.embedding.document_prefix
    if doc_prefix:
        for c in chunks:
            c.transcript_text = doc_prefix + c.transcript_text

    total = 0
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        written = index.build(batch, model)
        total += written
        print(f"  [{start + written}/{len(chunks)}] indexed {written} chunks", end="\r")

    print(f"\nDone — {total} chunks written to {colbert_dir}")
    print(f"Total index size : {len(index)} chunks")
    print()
    print("Next: enable ColBERT rescoring in config.yaml:")
    print("  colbert_enabled: true")
    print("  colbert_weight: 0.3   # adjust blend weight (0.1–0.5 typical)")


if __name__ == "__main__":
    main()
