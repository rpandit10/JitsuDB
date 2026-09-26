"""Re-embed all indexed chunks into a ChromaDB collection.

Usage:
    # After switching to nomic-embed-text-v1.5 (reads collection_name from config.yaml):
    python scripts/reembed.py

    # After fine-tuning (custom model + collection):
    python scripts/reembed.py --model models/bjj-nomic-ft --collection text_ft

    # Other options:
    python scripts/reembed.py --batch-size 64 --device cuda

This script reads every chunk from SQLite, embeds them with the configured model,
and upserts the result into the target ChromaDB collection.  The old collection is
left untouched so both model versions coexist for A/B comparison.

Run this after changing ``embedding.model_name`` in ``config.yaml``.
"""

from __future__ import annotations

# Must run before any jitsudb import so the setdefault("HF_HUB_OFFLINE","1") in
# jitsudb/__init__.py does not block the initial model download.
import os
os.environ["HF_HUB_OFFLINE"] = "0"

import argparse
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import SentenceTransformer


def reembed(
    conn: sqlite3.Connection,
    model: "SentenceTransformer",
    collection: "chromadb.Collection",
    batch_size: int = 32,
    document_prefix: str = "",
) -> int:
    """Re-embed all chunks from ``conn`` and upsert into ``collection``.

    Fetches taxonomy labels so ChromaDB metadata stays consistent with the
    original collection schema produced by ``embed_and_index``.
    """
    from jitsudb.models import Chunk
    from jitsudb.pipeline.embedder import build_chroma_metadata, embed_and_index

    rows = conn.execute(
        "SELECT id, video_id, start_time, end_time, transcript_text, context, gi_nogi "
        "FROM chunks"
    ).fetchall()

    if not rows:
        return 0

    chunks = [
        Chunk(
            id=row["id"],
            video_id=row["video_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            transcript_text=row["transcript_text"],
            context=row["context"],
            gi_nogi=row["gi_nogi"],
        )
        for row in rows
    ]

    chunk_ids = [c.id for c in chunks]
    ph = ",".join("?" * len(chunk_ids))
    label_rows = conn.execute(
        f"SELECT chunk_id, label_type, label_value FROM taxonomy_labels "
        f"WHERE chunk_id IN ({ph})",
        chunk_ids,
    ).fetchall()

    labels_by_chunk: dict[str, dict[str, list[str]]] = {}
    for row in label_rows:
        cid = row["chunk_id"]
        if cid not in labels_by_chunk:
            labels_by_chunk[cid] = {"position": [], "intent": [], "technique": []}
        labels_by_chunk[cid][row["label_type"]].append(row["label_value"])

    for chunk in chunks:
        lbl = labels_by_chunk.get(chunk.id, {})
        chunk.position = lbl.get("position", [])
        chunk.intent = lbl.get("intent", [])
        chunk.techniques = lbl.get("technique", [])

    embed_and_index(chunks, model, collection, batch_size=batch_size, document_prefix=document_prefix)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed all chunks into a ChromaDB collection")
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model name or local path (default: reads from config.yaml)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Target ChromaDB collection name (default: reads collection_name from config.yaml)",
    )
    parser.add_argument(
        "--document-prefix",
        default=None,
        help="Prefix prepended to each document before encoding (default: reads from config.yaml)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cpu or cuda (default: reads from config.yaml)")
    args = parser.parse_args()

    from jitsudb.config import get_config
    from jitsudb.db import get_connection
    from jitsudb.pipeline.embedder import get_chroma_collection, load_embedding_model

    cfg = get_config()
    conn = get_connection()
    chroma_dir = cfg.paths.data_dir / "chromadb"

    # Resolve local paths to absolute so SentenceTransformer doesn't mistake
    # "models/nomic-embed-text-v1.5" for a HuggingFace repo ID.
    if args.model:
        candidate = Path(args.model)
        model_name = str(candidate.resolve()) if candidate.exists() else args.model
    else:
        model_name = cfg.embedding.model_name
    device = args.device or cfg.embedding.device
    collection_name = args.collection or cfg.embedding.collection_name
    document_prefix = args.document_prefix if args.document_prefix is not None else cfg.embedding.document_prefix

    print(f"Model      : {model_name}")
    print(f"Collection : {collection_name}")
    print(f"Doc prefix : {document_prefix!r}")
    print(f"Device     : {device}")
    print()

    print(f"Loading model …")
    model = load_embedding_model(
        model_name=model_name,
        device=device,
        trust_remote_code=cfg.embedding.trust_remote_code,
    )

    print(f"Opening ChromaDB collection '{collection_name}' …")
    collection = get_chroma_collection(chroma_dir, collection_name)

    print("Re-embedding chunks …")
    n = reembed(conn, model, collection, batch_size=args.batch_size, document_prefix=document_prefix)
    print(f"Done — {n} chunks upserted into '{collection_name}'")
    print(f"Collection now contains {collection.count()} documents")


if __name__ == "__main__":
    main()
