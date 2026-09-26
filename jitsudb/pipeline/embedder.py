"""Sentence-transformer embedding and ChromaDB upsert."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from jitsudb.constants import CHROMADB_COLLECTION_NAME
from jitsudb.models import Chunk

if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import SentenceTransformer


def load_embedding_model(
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    device: str = "cpu",
    model_path: str | None = None,
    trust_remote_code: bool = False,
) -> "SentenceTransformer":
    """Load the sentence-transformer model. Call once and reuse."""
    from sentence_transformers import SentenceTransformer

    source = model_path if model_path else model_name
    logger.info(f"Loading embedding model from '{source}' on {device}")
    model = SentenceTransformer(source, device=device, trust_remote_code=trust_remote_code)
    logger.info("Embedding model loaded")
    return model


def get_chroma_collection(
    persist_dir: Path,
    collection_name: str = CHROMADB_COLLECTION_NAME,
) -> "chromadb.Collection":
    """Open or create the base ChromaDB collection for chunk embeddings."""
    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug(f"ChromaDB collection '{collection_name}' ready ({collection.count()} docs)")
    return collection


def get_ft_chroma_collection(persist_dir: Path, collection_name: str = "text_ft") -> "chromadb.Collection":
    """Open or create the fine-tuned-encoder ChromaDB collection.

    Keeps the fine-tuned embeddings separate from the base collection so both
    can coexist for A/B comparison without re-ingesting the base collection.
    """
    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug(f"ChromaDB collection '{collection_name}' ready ({collection.count()} docs)")
    return collection


def embed_and_index(
    chunks: list[Chunk],
    model: "SentenceTransformer",
    collection: "chromadb.Collection",
    batch_size: int = 32,
    document_prefix: str = "",
) -> None:
    """Embed a batch of chunks and upsert them into ChromaDB.

    Each document is stored with its taxonomy labels flattened into metadata so
    ChromaDB's ``where`` filter can apply structured filters at query time.
    """
    if not chunks:
        return

    texts = [f"{document_prefix}{c.transcript_text}" for c in chunks]

    logger.info(f"Embedding {len(chunks)} chunks (batch_size={batch_size})")
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(chunks) > batch_size,
            convert_to_numpy=True,
        )
    except RuntimeError as exc:
        if "no kernel image" in str(exc) or "CUDA" in str(exc):
            logger.warning(f"CUDA encode failed ({exc}); falling back to CPU for this model")
            model.to("cpu")
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(chunks) > batch_size,
                convert_to_numpy=True,
            )
        else:
            raise

    # ChromaDB rejects upserts larger than 5461 rows; split into safe batches.
    _CHROMA_MAX = 5000
    metadatas = [build_chroma_metadata(c) for c in chunks]
    ids = [c.id for c in chunks]
    emb_list = embeddings.tolist()
    for start in range(0, len(chunks), _CHROMA_MAX):
        end = start + _CHROMA_MAX
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=emb_list[start:end],
        )
    logger.info(f"Upserted {len(chunks)} chunks into ChromaDB")


def build_chroma_metadata(chunk: Chunk) -> dict:
    """Flatten a Chunk's taxonomy fields into a ChromaDB-compatible metadata dict.

    ChromaDB metadata values must be str | int | float | bool. Lists are
    serialised as comma-joined strings so they can be used with the
    ``$contains`` operator.
    """
    return {
        "video_id": chunk.video_id,
        "start_time": chunk.start_time,
        "end_time": chunk.end_time,
        "context": chunk.context or "",
        "gi_nogi": chunk.gi_nogi,
        "positions": ",".join(chunk.position),
        "intents": ",".join(chunk.intent),
        "techniques": ",".join(chunk.techniques),
    }
