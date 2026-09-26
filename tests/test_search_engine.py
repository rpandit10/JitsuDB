"""Unit tests for jitsudb.search.engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jitsudb.search.engine import SearchEngine


@pytest.fixture
def mock_engine():
    """SearchEngine with mocked embedding model and ChromaDB collection."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 384

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [[]],
        "distances": [[]],
        "metadatas": [[]],
        "documents": [[]],
    }

    return SearchEngine(
        embedding_model=mock_model,
        collection=mock_collection,
        top_k=10,
    )


class TestSearchEngine:
    def test_returns_list(self, mock_engine):
        results = mock_engine.search("side control escape")
        assert isinstance(results, list)

    def test_empty_corpus_returns_empty(self, mock_engine):
        results = mock_engine.search("armbar from mount")
        assert results == []

    def test_search_calls_embed(self, mock_engine):
        mock_engine.search("kimura grip")
        mock_engine.model.encode.assert_called_once()

    def test_search_calls_chroma_query(self, mock_engine):
        mock_engine.search("rear naked choke")
        mock_engine.collection.query.assert_called_once()


class TestRerankerCap:
    def test_reranker_receives_at_most_reranker_top_k(self, tmp_db):
        """Reranker is called with ≤ reranker_top_k candidates regardless of pool size."""
        import sqlite3

        import numpy as np
        from jitsudb.models import Chunk
        from jitsudb.search.engine import SearchEngine

        video_id = "vid-reranker-cap"
        tmp_db.execute(
            "INSERT INTO videos (id, title, instructor, file_path, duration_seconds, "
            "gi_nogi, primary_category, status, added_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (video_id, "Cap Test", "Tester", "/fake/v.mp4", 450.0,
             "nogi", "Passing", "indexed", "2024-01-01T00:00:00"),
        )
        chunk_ids = [f"cap-chunk-{i:03d}" for i in range(10)]
        for i, cid in enumerate(chunk_ids):
            tmp_db.execute(
                "INSERT INTO chunks (id, video_id, start_time, end_time, "
                "transcript_text, context, gi_nogi) VALUES (?,?,?,?,?,?,?)",
                (cid, video_id, float(i * 45), float(i * 45 + 44),
                 f"Chunk {i} text.", "neutral", "nogi"),
            )
        tmp_db.commit()

        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [chunk_ids],
            "distances": [[0.1 + i * 0.02 for i in range(10)]],
            "metadatas": [[{}] * 10],
            "documents": [[f"doc {i}" for i in range(10)]],
        }

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []

        engine = SearchEngine(
            embedding_model=mock_model,
            collection=mock_collection,
            top_k=10,
            reranker=mock_reranker,
            reranker_top_k=5,
            beam_width=0,
        )
        engine.search("basic drill")

        assert mock_reranker.rerank.call_count == 1
        candidates_passed = mock_reranker.rerank.call_args.args[1]
        assert len(candidates_passed) <= 5

    def test_reranker_candidates_presorted_by_score(self, tmp_db):
        """Candidates passed to reranker are sorted by relevance_score descending."""
        from jitsudb.search.engine import SearchEngine

        video_id = "vid-reranker-sort"
        tmp_db.execute(
            "INSERT INTO videos (id, title, instructor, file_path, duration_seconds, "
            "gi_nogi, primary_category, status, added_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (video_id, "Sort Test", "Tester", "/fake/v.mp4", 450.0,
             "nogi", "Passing", "indexed", "2024-01-01T00:00:00"),
        )
        chunk_ids = [f"sort-chunk-{i:03d}" for i in range(6)]
        # Give each chunk a distinct distance so scores differ clearly
        distances = [0.05, 0.4, 0.1, 0.6, 0.2, 0.5]
        for i, cid in enumerate(chunk_ids):
            tmp_db.execute(
                "INSERT INTO chunks (id, video_id, start_time, end_time, "
                "transcript_text, context, gi_nogi) VALUES (?,?,?,?,?,?,?)",
                (cid, video_id, float(i * 45), float(i * 45 + 44),
                 f"Chunk {i} text.", "neutral", "nogi"),
            )
        tmp_db.commit()

        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [chunk_ids],
            "distances": [distances],
            "metadatas": [[{}] * 6],
            "documents": [[f"doc {i}" for i in range(6)]],
        }

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []

        engine = SearchEngine(
            embedding_model=mock_model,
            collection=mock_collection,
            top_k=10,
            reranker=mock_reranker,
            reranker_top_k=10,  # large cap — all 6 pass through
            beam_width=0,
        )
        engine.search("basic drill")

        candidates = mock_reranker.rerank.call_args.args[1]
        scores = [r.relevance_score for r in candidates]
        assert scores == sorted(scores, reverse=True)


class TestQueryPrefix:
    def test_query_prefix_prepended_to_encode_call(self):
        """_embed_query prepends query_prefix before calling model.encode."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 768

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]
        }

        engine = SearchEngine(
            embedding_model=mock_model,
            collection=mock_collection,
            query_prefix="search_query: ",
        )
        engine.search("armbar from guard")

        call_args = mock_model.encode.call_args
        text_passed = call_args.args[0] if call_args.args else call_args.kwargs.get("sentence") or call_args.kwargs.get("sentences")
        assert text_passed is not None
        assert text_passed.startswith("search_query: ")

    def test_empty_prefix_passes_query_unchanged(self):
        """With no prefix, the raw query text is passed to encode."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]
        }

        engine = SearchEngine(
            embedding_model=mock_model,
            collection=mock_collection,
            query_prefix="",
        )
        engine._embed_query("triangle choke")

        call_args = mock_model.encode.call_args
        text_passed = call_args.args[0] if call_args.args else call_args.kwargs.get("sentence") or call_args.kwargs.get("sentences")
        assert text_passed == "triangle choke"
