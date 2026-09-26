"""Cross-encoder re-ranker for two-stage neural retrieval.

Stage 1 (bi-encoder): ChromaDB cosine similarity retrieves top-50 candidates fast.
Stage 2 (cross-encoder): Reads query + chunk jointly through full attention for
accurate relevance scoring. Slower per pair but only runs on the small candidate set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from jitsudb.models import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Re-ranks a candidate list using a cross-encoder relevance model.

    The cross-encoder concatenates (query, passage) as a single input and
    scores joint relevance, capturing negation and context that bi-encoder
    cosine similarity misses.
    """

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        from sentence_transformers import CrossEncoder
        try:
            self.model: CrossEncoder = CrossEncoder(model_name, device=device, max_length=512)
            # Warm up with a trivial call to surface CUDA kernel errors early
            if device != "cpu":
                self.model.predict([("test", "test")], show_progress_bar=False)
            logger.info(f"Cross-encoder loaded: '{model_name}' on {device}")
        except Exception as exc:
            if device != "cpu":
                logger.warning(f"Cross-encoder failed on {device} ({exc}), falling back to CPU")
                self.model = CrossEncoder(model_name, device="cpu", max_length=512)
                logger.info(f"Cross-encoder loaded: '{model_name}' on cpu (fallback)")
            else:
                raise
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Score each (query, chunk) pair and return the top_k highest-scoring results.

        Scores from ms-marco models are raw logits. We apply min-max normalisation
        so the displayed relevance_score stays in [0, 1] across the result set.
        """
        if not results:
            return results

        pairs = [(query, r.chunk.transcript_text) for r in results]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        for result, score in zip(results, raw_scores):
            result.relevance_score = float(score)

        # Normalise to [0, 1] so the UI percentage badge makes sense
        if len(raw_scores) > 1:
            lo, hi = float(min(raw_scores)), float(max(raw_scores))
            if hi > lo:
                for result in results:
                    result.relevance_score = (result.relevance_score - lo) / (hi - lo)
            else:
                for result in results:
                    result.relevance_score = 1.0

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        logger.debug(f"Re-ranked {len(results)} candidates → top {top_k}")
        return results[:top_k]
