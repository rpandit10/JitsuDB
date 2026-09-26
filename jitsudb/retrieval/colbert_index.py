"""ColBERT-style late-interaction rescoring backed by a numpy memory-mapped store.

Only the text collection is indexed here; CLIP visual embeddings stay pooled.

Usage::

    index = ColBERTIndex(Path("data/colbert"))
    index.build(chunks, sentence_transformer_model)
    scores = index.rescore("heel hook setup", candidate_chunk_ids, model)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from jitsudb.models import Chunk

_EMBED_FILE = "colbert_embeddings.bin"
_META_FILE = "colbert_meta.json"


class ColBERTIndex:
    """Per-token embedding store with MaxSim scoring.

    Token embeddings are written to a single flat binary file (float32).
    A JSON sidecar maps each chunk ID to its byte offset, token count, and
    embedding dimension so any chunk can be sliced out via ``np.memmap``
    without loading the entire file into RAM.
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._embed_path = self.index_dir / _EMBED_FILE
        self._meta_path = self.index_dir / _META_FILE
        self._meta: dict[str, dict] = {}
        if self._meta_path.exists():
            self._meta = json.loads(self._meta_path.read_text())

    def build(self, chunks: list["Chunk"], model: "SentenceTransformer") -> int:
        """Index per-token embeddings for a list of chunks.

        Appends new chunks to the existing binary file so the index can be
        grown incrementally.  Re-indexing a chunk that already exists
        overwrites its meta entry but does NOT compact the file.
        """
        if not chunks:
            return 0

        texts = [c.transcript_text for c in chunks]
        token_embeddings: list[np.ndarray] = model.encode(
            texts,
            output_value="token_embeddings",
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        start_offset = self._embed_path.stat().st_size if self._embed_path.exists() else 0

        with self._embed_path.open("ab") as f:
            offset = start_offset
            for chunk, embs in zip(chunks, token_embeddings):
                arr = np.asarray(embs, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                n_tokens, dim = arr.shape
                f.write(arr.tobytes())
                self._meta[chunk.id] = {
                    "offset": offset,
                    "n_tokens": n_tokens,
                    "dim": dim,
                }
                offset += arr.nbytes

        self._meta_path.write_text(json.dumps(self._meta))
        logger.debug(f"ColBERT: indexed {len(chunks)} chunks → {self._embed_path}")
        return len(chunks)

    def maxsim_score(self, query_tokens: np.ndarray, chunk_id: str) -> float:
        """Compute MaxSim between query token vectors and a stored chunk.

        MaxSim = Σᵢ maxⱼ(qᵢ · dⱼ)  — sum over query tokens of the
        maximum dot product with any document token.  ``query_tokens`` is a
        2-D float32 array of shape (n_query_tokens, dim).  Unindexed chunks
        score 0.0.
        """
        if chunk_id not in self._meta or not self._embed_path.exists():
            return 0.0

        info = self._meta[chunk_id]
        n_tokens: int = info["n_tokens"]
        dim: int = info["dim"]
        offset: int = info["offset"]

        mm = np.memmap(self._embed_path, dtype=np.float32, mode="r")
        start = offset // 4  # bytes → float32 index
        doc_tokens = mm[start : start + n_tokens * dim].reshape(n_tokens, dim).copy()

        # (n_query × n_doc) — take max over doc axis, then sum over query axis
        sim = query_tokens @ doc_tokens.T
        return float(sim.max(axis=1).sum())

    def rescore(
        self,
        query: str,
        chunk_ids: list[str],
        model: "SentenceTransformer",
    ) -> dict[str, float]:
        """Return MaxSim scores for candidate chunk IDs, 0.0 when unindexed."""
        if not chunk_ids or not self._embed_path.exists():
            return {cid: 0.0 for cid in chunk_ids}

        query_embs: list[np.ndarray] = model.encode(
            [query],
            output_value="token_embeddings",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_tokens = np.asarray(query_embs[0], dtype=np.float32)
        if query_tokens.ndim == 1:
            query_tokens = query_tokens.reshape(1, -1)

        return {cid: self.maxsim_score(query_tokens, cid) for cid in chunk_ids}

    def __len__(self) -> int:
        """Return the number of chunks currently in the index."""
        return len(self._meta)
