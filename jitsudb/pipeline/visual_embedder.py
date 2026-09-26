"""CLIP visual embedding — encodes chunk thumbnails into a separate ChromaDB collection.

Enables cross-modal retrieval: a text query embeds via CLIP's text encoder and
retrieves chunks whose thumbnails are visually similar, even when the transcript
doesn't mention the technique by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import SentenceTransformer

    from jitsudb.models import Chunk

VISUAL_COLLECTION_NAME = "jitsudb_visual"


def load_clip_model(model_name: str = "clip-ViT-B-32", device: str = "cpu") -> "SentenceTransformer":
    """Load the CLIP sentence-transformer. Call once and reuse."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading CLIP model '{model_name}' on {device}")
    model = SentenceTransformer(model_name, device=device)
    logger.info("CLIP model loaded")
    return model


def get_visual_collection(persist_dir: Path) -> "chromadb.Collection":
    """Open or create the ChromaDB collection for visual embeddings."""
    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=VISUAL_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug(
        f"Visual ChromaDB collection '{VISUAL_COLLECTION_NAME}' ready ({collection.count()} docs)"
    )
    return collection


def embed_chunks_visual(
    chunks: list["Chunk"],
    clip_model: "SentenceTransformer",
    collection: "chromadb.Collection",
    batch_size: int = 16,
) -> int:
    """Encode chunk thumbnails with CLIP and upsert into the visual collection.

    Only chunks with an existing thumbnail_path are processed. Thumbnails are
    loaded as RGB PIL images so JPEG artifacts don't affect the colour channels.
    """
    from PIL import Image

    to_embed = [c for c in chunks if c.thumbnail_path and c.thumbnail_path.exists()]
    if not to_embed:
        logger.warning("No thumbnails found — skipping visual embedding")
        return 0

    logger.info(f"Embedding {len(to_embed)} thumbnails with CLIP (batch_size={batch_size})")

    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i : i + batch_size]
        images = [Image.open(c.thumbnail_path).convert("RGB") for c in batch]
        try:
            embeddings = clip_model.encode(
                images,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except RuntimeError as exc:
            if "no kernel image" in str(exc) or "CUDA" in str(exc):
                logger.warning(f"CUDA encode failed ({exc}); falling back to CPU for CLIP model")
                clip_model.to("cpu")
                embeddings = clip_model.encode(
                    images,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            else:
                raise
        collection.upsert(
            ids=[c.id for c in batch],
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "video_id": c.video_id,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                }
                for c in batch
            ],
        )

    logger.info(f"Upserted {len(to_embed)} visual embeddings into '{VISUAL_COLLECTION_NAME}'")
    return len(to_embed)
