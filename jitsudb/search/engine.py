"""Hybrid search engine — semantic retrieval merged with structured SQLite filtering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from jitsudb.models import Chunk, SearchResult, Video

if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import SentenceTransformer

    from jitsudb.retrieval.colbert_index import ColBERTIndex
    from jitsudb.search.reranker import CrossEncoderReranker


class SearchEngine:
    """Executes hybrid searches over the indexed chunk corpus.

    Three-stage retrieval:
      Stage 1a — bi-encoder (text): ChromaDB cosine similarity on transcript embeddings.
      Stage 1b — bi-encoder (visual): CLIP cross-modal similarity on thumbnail embeddings.
      Stage 2  — cross-encoder: joint (query, chunk) scoring over the merged candidate set.
    """

    def __init__(
        self,
        embedding_model: "SentenceTransformer",
        collection: "chromadb.Collection",
        top_k: int = 30,
        semantic_weight: float = 0.7,
        reranker: "CrossEncoderReranker | None" = None,
        clip_model: "SentenceTransformer | None" = None,
        visual_collection: "chromadb.Collection | None" = None,
        visual_weight: float = 0.3,
        beam_width: int = 2,
        max_expand: int = 4,
        reranker_top_k: int = 20,
        candidate_k: int = 30,
        hyde_enabled: bool = False,
        hyde_n_hypotheticals: int = 3,
        click_reranker_enabled: bool = True,
        query_prefix: str = "",
        colbert_index: "ColBERTIndex | None" = None,
        colbert_weight: float = 0.3,
    ) -> None:
        self.model = embedding_model
        self.collection = collection
        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self.reranker = reranker
        self.clip_model = clip_model
        self.visual_collection = visual_collection
        self.visual_weight = visual_weight
        self.beam_width = beam_width
        self.max_expand = max_expand
        self.reranker_top_k = reranker_top_k
        self.candidate_k = candidate_k
        self.hyde_enabled = hyde_enabled
        self.hyde_n_hypotheticals = hyde_n_hypotheticals
        self.click_reranker_enabled = click_reranker_enabled
        self.query_prefix = query_prefix
        self.colbert_index = colbert_index
        self.colbert_weight = colbert_weight
        self._last_query_embedding: list[float] | None = None
        # Populated after every search(); used by the metrics endpoint and search route.
        self._last_timing: dict[str, float] | None = None
        self._last_total_ms: float | None = None
        # LRU-style cache for HyDE centroid embeddings (keyed by "n:query")
        self._hyde_cache: dict[str, list[float]] = {}
        self._hyde_cache_max = 128

    def search(
        self,
        query: str,
        filters: dict[str, list[str]] | None = None,
    ) -> list[SearchResult]:
        """Run a hybrid search and return ranked results.

        Steps:
        1. Preprocess the query (abbreviation expansion, auto-filter detection).
        2. Embed the expanded query.
        3. Retrieve semantic top-K=50 candidates from ChromaDB.
        4. If filters are active, cross-reference with SQLite taxonomy_labels.
        5. Deduplicate and rank via ranking.py.

        ``filters`` holds manually applied UI filters, keyed by "positions",
        "intents", "techniques", "context", "gi_nogi" or "instructor", each
        mapping to a list of label strings.
        """
        from jitsudb.db import get_connection
        from jitsudb.search.query_preprocessor import detect_temporal_connectives, preprocess_query
        from jitsudb.search.ranking import build_snippet, rank_results
        from jitsudb.search.timing import SearchTiming

        if filters is None:
            filters = {}

        t = SearchTiming()

        t.start("preprocess")
        expanded_query, detected_filters = preprocess_query(query)
        is_multihop = detect_temporal_connectives(expanded_query)
        logger.debug(f"Query: {expanded_query!r}, auto-detected: {detected_filters}")
        t.stop("preprocess")

        t.start("embed")
        embedding = self._embed_query(expanded_query)
        self._last_query_embedding = embedding
        t.stop("embed")

        if self.hyde_enabled:
            t.start("hyde")
            hyde_embedding = self._expand_query_hyde(
                expanded_query, n=self.hyde_n_hypotheticals
            )
            if hyde_embedding is not None:
                words = len(expanded_query.split())
                alpha = max(0.2, 1.0 - words / 12.0)
                embedding = [
                    alpha * h + (1.0 - alpha) * e
                    for h, e in zip(hyde_embedding, embedding)
                ]
                logger.debug(f"HyDE: {self.hyde_n_hypotheticals} hypotheticals, α={alpha:.2f}")
            t.stop("hyde")

        t.start("ann_text")
        text_candidates = self._fetch_semantic_candidates(embedding, candidate_k=self.candidate_k)
        t.stop("ann_text")

        # score_map: chunk_id → {text_score, visual_score}
        score_map: dict[str, dict[str, float]] = {
            c["id"]: {"text_score": max(0.0, 1.0 - c["distance"]), "visual_score": 0.0}
            for c in text_candidates
        }

        # Stage 1b: visual retrieval (CLIP cross-modal)
        if self.clip_model is not None and self.visual_collection is not None:
            t.start("ann_visual")
            clip_embedding = self._embed_query_clip(expanded_query)
            visual_candidates = self._fetch_visual_candidates(clip_embedding, candidate_k=self.candidate_k)
            for vc in visual_candidates:
                vis_score = max(0.0, 1.0 - vc["distance"])
                if vc["id"] in score_map:
                    score_map[vc["id"]]["visual_score"] = vis_score
                else:
                    score_map[vc["id"]] = {"text_score": 0.0, "visual_score": vis_score}
            logger.debug(
                f"Visual retrieval: {len(visual_candidates)} candidates, "
                f"{len(score_map)} total after merge"
            )
            t.stop("ann_visual")

        if not score_map:
            self._last_timing = t.stages
            self._last_total_ms = t.total_ms
            return []

        candidate_ids = list(score_map.keys())

        # Merge auto-detected filters with manually applied UI filters
        merged_filters: dict[str, list[str]] = {}
        for k in ("positions", "intents", "techniques"):
            combined = list(set(detected_filters.get(k, []) + filters.get(k, [])))
            if combined:
                merged_filters[k] = combined
        for k in ("context", "gi_nogi", "instructor"):
            val = filters.get(k)
            if val:
                merged_filters[k] = val if isinstance(val, list) else [val]

        if merged_filters:
            t.start("filter")
            candidate_ids = self._apply_structured_filters(candidate_ids, merged_filters)
            t.stop("filter")

        if not candidate_ids:
            self._last_timing = t.stages
            self._last_total_ms = t.total_ms
            return []

        t.start("db_fetch")
        conn = get_connection()

        placeholders = ",".join("?" * len(candidate_ids))

        chunk_rows = conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",
            candidate_ids,
        ).fetchall()

        if not chunk_rows:
            return []

        video_ids = list({r["video_id"] for r in chunk_rows})
        vid_placeholders = ",".join("?" * len(video_ids))
        video_rows = conn.execute(
            f"SELECT * FROM videos WHERE id IN ({vid_placeholders})",
            video_ids,
        ).fetchall()
        video_map = {r["id"]: r for r in video_rows}

        label_rows = conn.execute(
            f"SELECT chunk_id, label_type, label_value FROM taxonomy_labels WHERE chunk_id IN ({placeholders})",
            candidate_ids,
        ).fetchall()
        labels_by_chunk: dict[str, dict[str, list[str]]] = {}
        for row in label_rows:
            cid = row["chunk_id"]
            if cid not in labels_by_chunk:
                labels_by_chunk[cid] = {"position": [], "intent": [], "technique": []}
            labels_by_chunk[cid][row["label_type"]].append(row["label_value"])

        results: list[SearchResult] = []
        for chunk_row in chunk_rows:
            chunk_id = chunk_row["id"]
            scores = score_map.get(chunk_id)
            if scores is None:
                continue

            text_score = scores["text_score"]
            visual_score = scores["visual_score"]
            # Combined score is used for pre-reranker sorting; cross-encoder will override
            combined_score = (
                (1.0 - self.visual_weight) * text_score
                + self.visual_weight * visual_score
            )

            video_row = video_map.get(chunk_row["video_id"])
            if video_row is None:
                continue

            labels = labels_by_chunk.get(chunk_id, {"position": [], "intent": [], "technique": []})

            chunk = Chunk(
                id=chunk_row["id"],
                video_id=chunk_row["video_id"],
                start_time=chunk_row["start_time"],
                end_time=chunk_row["end_time"],
                transcript_text=chunk_row["transcript_text"],
                thumbnail_path=Path(chunk_row["thumbnail_path"]) if chunk_row["thumbnail_path"] else None,
                position=labels.get("position", []),
                context=chunk_row["context"],
                intent=labels.get("intent", []),
                techniques=labels.get("technique", []),
                gi_nogi=chunk_row["gi_nogi"],
            )

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
                processed_at=video_row["processed_at"],
            )

            snippet = build_snippet(chunk.transcript_text, expanded_query)

            if text_score > 0 and visual_score > 0:
                retrieved_by = "both"
            elif visual_score > 0:
                retrieved_by = "visual"
            else:
                retrieved_by = "text"

            results.append(SearchResult(
                chunk=chunk,
                video=video,
                relevance_score=combined_score,
                snippet=snippet,
                detected_filters=detected_filters,
                visual_score=visual_score,
                retrieved_by=retrieved_by,
            ))
        t.stop("db_fetch")

        deduped = results
        if self.reranker is not None:
            from jitsudb.search.ranking import deduplicate
            deduped = deduplicate(results)
            deduped.sort(key=lambda r: r.relevance_score, reverse=True)
            reranker_input = deduped[: self.reranker_top_k]
            t.start("rerank")
            all_ranked = self.reranker.rerank(expanded_query, reranker_input, top_k=len(reranker_input))
            t.stop("rerank")
        else:
            all_ranked = rank_results(results, top_k=len(results))

        # Guarantee at least VISUAL_MIN visual-only results surface in the final list.
        # Without this, the cross-encoder (text-only) always pushes visual candidates out.
        visual_min = 3 if (self.clip_model is not None and self.visual_collection is not None) else 0
        primary = [r for r in all_ranked if r.retrieved_by != "visual"]
        visual_only = [r for r in all_ranked if r.retrieved_by == "visual"]

        # Fill the final list: primary up to (top_k - guaranteed slots), then visual
        guaranteed = min(visual_min, len(visual_only))
        main_slots = self.top_k - guaranteed
        final = primary[:main_slots] + visual_only[:guaranteed]
        # Re-sort so visual guarantees slot in by score rather than always appending at end
        final.sort(key=lambda r: r.relevance_score, reverse=True)

        # Stage 3: Beam-walk context expansion
        # Expand each top-ranked anchor by beam_width temporal hops in both directions,
        # then append any novel neighbours (up to max_expand) as context results.
        if self.beam_width > 0 and final:
            t.start("beam")
            from jitsudb.db.graph import get_contiguous_window
            anchor_ids = [r.chunk.id for r in final]
            expanded_ids = get_contiguous_window(anchor_ids, beam_width=self.beam_width, conn=conn)
            known_ids = set(anchor_ids)
            new_ids = [cid for cid in expanded_ids if cid not in known_ids][: self.max_expand]
            if new_ids:
                new_ph = ",".join("?" * len(new_ids))
                new_chunk_rows = conn.execute(
                    f"SELECT * FROM chunks WHERE id IN ({new_ph})", new_ids
                ).fetchall()
                new_label_rows = conn.execute(
                    f"SELECT chunk_id, label_type, label_value FROM taxonomy_labels"
                    f" WHERE chunk_id IN ({new_ph})",
                    new_ids,
                ).fetchall()
                new_labels: dict[str, dict[str, list[str]]] = {}
                for row in new_label_rows:
                    cid = row["chunk_id"]
                    if cid not in new_labels:
                        new_labels[cid] = {"position": [], "intent": [], "technique": []}
                    new_labels[cid][row["label_type"]].append(row["label_value"])
                for chunk_row in new_chunk_rows:
                    video_row = video_map.get(chunk_row["video_id"])
                    if video_row is None:
                        continue
                    lbl = new_labels.get(
                        chunk_row["id"], {"position": [], "intent": [], "technique": []}
                    )
                    chunk = Chunk(
                        id=chunk_row["id"],
                        video_id=chunk_row["video_id"],
                        start_time=chunk_row["start_time"],
                        end_time=chunk_row["end_time"],
                        transcript_text=chunk_row["transcript_text"],
                        thumbnail_path=Path(chunk_row["thumbnail_path"]) if chunk_row["thumbnail_path"] else None,
                        position=lbl.get("position", []),
                        context=chunk_row["context"],
                        intent=lbl.get("intent", []),
                        techniques=lbl.get("technique", []),
                        gi_nogi=chunk_row["gi_nogi"],
                    )
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
                        processed_at=video_row["processed_at"],
                    )
                    final.append(SearchResult(
                        chunk=chunk,
                        video=video,
                        relevance_score=0.0,
                        snippet=build_snippet(chunk.transcript_text, expanded_query),
                        detected_filters=detected_filters,
                        retrieved_by="text",
                    ))
                logger.debug(f"Beam expansion: added {len(new_chunk_rows)} context chunks")
            t.stop("beam")

        # Multi-hop temporal ordering
        # When the query implies a temporal chain ("before", "after", "leads to", …),
        # group results by video and sort within each group by start_time so the
        # user sees a narrative flow rather than a pure relevance ranking.
        if is_multihop and final:
            from collections import defaultdict
            video_groups: dict[str, list[SearchResult]] = defaultdict(list)
            for r in final:
                video_groups[r.video.id].append(r)
            sorted_groups = sorted(
                video_groups.values(),
                key=lambda g: max(r.relevance_score for r in g),
                reverse=True,
            )
            final = []
            for group in sorted_groups:
                final.extend(sorted(group, key=lambda r: r.chunk.start_time))

        # ColBERT late-interaction rescoring
        if self.colbert_index is not None and final and len(self.colbert_index) > 0:
            t.start("colbert")
            query_for_colbert = (self.query_prefix + expanded_query) if self.query_prefix else expanded_query
            candidate_ids_for_colbert = [r.chunk.id for r in final if r.chunk.id in self.colbert_index._meta]
            if candidate_ids_for_colbert:
                import numpy as np
                raw_embs = self.model.encode(
                    [query_for_colbert],
                    output_value="token_embeddings",
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                query_tokens = np.asarray(raw_embs[0], dtype=np.float32)
                if query_tokens.ndim == 1:
                    query_tokens = query_tokens.reshape(1, -1)

                maxsim: dict[str, float] = {
                    cid: self.colbert_index.maxsim_score(query_tokens, cid)
                    for cid in candidate_ids_for_colbert
                }
                # Normalise MaxSim scores to [0, 1] within this result set
                vals = list(maxsim.values())
                lo, hi = min(vals), max(vals)
                span = hi - lo if hi > lo else 1.0
                for r in final:
                    raw = maxsim.get(r.chunk.id)
                    if raw is not None:
                        norm = (raw - lo) / span
                        r.relevance_score = (
                            (1.0 - self.colbert_weight) * r.relevance_score
                            + self.colbert_weight * norm
                        )
                final.sort(key=lambda r: r.relevance_score, reverse=True)
                logger.debug(f"ColBERT: rescored {len(candidate_ids_for_colbert)}/{len(final)} chunks")
            t.stop("colbert")

        # Click-through boost
        if self.click_reranker_enabled and final and not is_multihop:
            t.start("click_boost")
            from jitsudb.search.click_reranker import boost as click_boost
            final = click_boost(embedding, final, conn)
            t.stop("click_boost")

        self._last_timing = t.stages
        self._last_total_ms = t.total_ms
        return final

    def embed(self, text: str) -> list[float]:
        """Public wrapper around _embed_query for external callers."""
        return self._embed_query(text)

    def _expand_query_hyde(self, query: str, n: int = 1) -> list[float] | None:
        """Generate N hypothetical passages via Ollama and return their centroid embedding.

        Short queries (≤5 words) use n=1 regardless of the configured value because
        extra hypotheticals don't improve retrieval for unambiguous short queries and
        they are the main source of latency. Longer or ambiguous queries still benefit
        from multiple passes so the configured n is respected.

        Results are cached in `_hyde_cache` (up to 128 entries, LRU eviction) so
        repeated identical searches pay no Ollama cost after the first call.

        Returns None when Ollama is unavailable or all generations fail.
        """
        from jitsudb.config import get_config
        from jitsudb.rag.synthesizer import RagSynthesizer

        # Adaptive N: short queries don't benefit from multiple hypotheticals
        word_count = len(query.split())
        effective_n = 1 if word_count <= 5 else (min(n, 2) if word_count <= 10 else n)

        cache_key = f"{effective_n}:{query}"
        if cache_key in self._hyde_cache:
            logger.debug(f"HyDE cache hit ({effective_n} hyp): {query!r:.50}")
            return self._hyde_cache[cache_key]

        cfg = get_config()
        synth = RagSynthesizer(ollama_url=cfg.rag.ollama_url, model=cfg.rag.model)
        if not synth.is_available():
            logger.debug("HyDE: Ollama unavailable, skipping expansion")
            return None

        embeddings: list[list[float]] = []
        for i in range(effective_n):
            try:
                passage = "".join(synth._prompt_hyde(query))
                if not passage.strip():
                    continue
                logger.debug(f"HyDE [{i+1}/{effective_n}] ({len(passage)} chars): {passage[:60]!r}…")
                embeddings.append(self._embed_query(passage))
            except Exception as exc:
                logger.warning(f"HyDE generation [{i+1}/{effective_n}] failed: {exc}")

        if not embeddings:
            return None

        dim = len(embeddings[0])
        centroid = [
            sum(e[j] for e in embeddings) / len(embeddings) for j in range(dim)
        ]

        # Store in cache (evict oldest when full)
        if len(self._hyde_cache) >= self._hyde_cache_max:
            oldest = next(iter(self._hyde_cache))
            del self._hyde_cache[oldest]
        self._hyde_cache[cache_key] = centroid

        return centroid

    def _embed_query_clip(self, text: str) -> list[float]:
        """Encode a query with CLIP's text encoder into the visual embedding space."""
        embedding = self.clip_model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def _fetch_visual_candidates(
        self, embedding: list[float], candidate_k: int = 50
    ) -> list[dict]:
        """Query the visual ChromaDB collection for the top candidate_k thumbnail matches."""
        count = self.visual_collection.count()  # type: ignore[union-attr]
        if count == 0:
            return []
        n = min(candidate_k, count)
        results = self.visual_collection.query(  # type: ignore[union-attr]
            query_embeddings=[embedding],
            n_results=n,
            include=["metadatas", "distances"],
        )
        candidates: list[dict] = []
        if not results["ids"] or not results["ids"][0]:
            return candidates
        for i, chunk_id in enumerate(results["ids"][0]):
            candidates.append({
                "id": chunk_id,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return candidates

    def _embed_query(self, text: str) -> list[float]:
        """Embed a query string using the configured model and query prefix."""
        embedding = self.model.encode(f"{self.query_prefix}{text}", normalize_embeddings=True)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def _fetch_semantic_candidates(
        self, embedding: list[float], candidate_k: int = 50
    ) -> list[dict]:
        """Query ChromaDB for the top candidate_k most similar chunks."""
        count = self.collection.count()
        if count == 0:
            return []
        n = min(candidate_k, count)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["metadatas", "distances", "documents"],
        )
        candidates: list[dict] = []
        if not results["ids"] or not results["ids"][0]:
            return candidates
        for i, chunk_id in enumerate(results["ids"][0]):
            candidates.append({
                "id": chunk_id,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "document": results["documents"][0][i],
            })
        return candidates

    def _apply_structured_filters(
        self, candidate_ids: list[str], filters: dict[str, list[str]]
    ) -> list[str]:
        """Filter candidate chunk IDs against active taxonomy labels in SQLite."""
        if not candidate_ids:
            return []

        from jitsudb.db import get_connection
        conn = get_connection()

        passing_ids = set(candidate_ids)

        for filter_key, values in filters.items():
            if not values or not passing_ids:
                continue

            current_ids = list(passing_ids)
            current_placeholders = ",".join("?" * len(current_ids))

            if filter_key in ("positions", "intents", "techniques"):
                label_type = {
                    "positions": "position",
                    "intents": "intent",
                    "techniques": "technique",
                }[filter_key]
                val_placeholders = ",".join("?" * len(values))
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT chunk_id FROM taxonomy_labels
                    WHERE chunk_id IN ({current_placeholders})
                    AND label_type = ?
                    AND label_value IN ({val_placeholders})
                    """,
                    (*current_ids, label_type, *values),
                ).fetchall()
                passing_ids &= {r["chunk_id"] for r in rows}

            elif filter_key == "context":
                ctx_placeholders = ",".join("?" * len(values))
                rows = conn.execute(
                    f"""
                    SELECT id FROM chunks
                    WHERE id IN ({current_placeholders})
                    AND context IN ({ctx_placeholders})
                    """,
                    (*current_ids, *values),
                ).fetchall()
                passing_ids &= {r["id"] for r in rows}

            elif filter_key == "gi_nogi":
                non_both = [v for v in values if v != "both"]
                if non_both:
                    gi_placeholders = ",".join("?" * len(non_both))
                    rows = conn.execute(
                        f"""
                        SELECT id FROM chunks
                        WHERE id IN ({current_placeholders})
                        AND (gi_nogi IN ({gi_placeholders}) OR gi_nogi = 'both')
                        """,
                        (*current_ids, *non_both),
                    ).fetchall()
                    passing_ids &= {r["id"] for r in rows}

            elif filter_key == "instructor":
                matched: set[str] = set()
                for instr in values:
                    rows = conn.execute(
                        f"""
                        SELECT c.id FROM chunks c
                        JOIN videos v ON c.video_id = v.id
                        WHERE c.id IN ({current_placeholders})
                        AND LOWER(v.instructor) LIKE LOWER(?)
                        """,
                        (*current_ids, f"%{instr}%"),
                    ).fetchall()
                    matched |= {r["id"] for r in rows}
                passing_ids &= matched

        # Preserve original candidate ordering
        return [cid for cid in candidate_ids if cid in passing_ids]
