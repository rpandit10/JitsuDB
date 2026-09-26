"""HyDE query expansion and ColBERT late-interaction index."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jitsudb.retrieval.colbert_index import ColBERTIndex


# Helpers shared by multiple test classes
def _make_chunk(chunk_id: str, text: str = "Some BJJ text here."):
    from jitsudb.models import Chunk
    return Chunk(
        id=chunk_id,
        video_id="vid-test",
        start_time=0.0,
        end_time=45.0,
        transcript_text=text,
        gi_nogi="nogi",
    )


def _fake_token_embs(n_sentences: int, n_tokens: int = 6, dim: int = 16) -> list[np.ndarray]:
    """Return a list of (n_tokens × dim) float32 arrays — one per sentence."""
    rng = np.random.default_rng(seed=42)
    return [rng.random((n_tokens, dim)).astype(np.float32) for _ in range(n_sentences)]


class TestColBERTIndex:
    def test_build_creates_index_files(self, tmp_path):
        chunks = [_make_chunk("c-001"), _make_chunk("c-002")]
        mock_model = MagicMock()
        mock_model.encode.return_value = _fake_token_embs(2)

        index = ColBERTIndex(tmp_path / "colbert")
        n = index.build(chunks, mock_model)

        assert n == 2
        assert (tmp_path / "colbert" / "colbert_embeddings.bin").exists()
        assert (tmp_path / "colbert" / "colbert_meta.json").exists()

    def test_build_meta_contains_all_chunks(self, tmp_path):
        chunks = [_make_chunk(f"c-{i:03d}") for i in range(4)]
        mock_model = MagicMock()
        mock_model.encode.return_value = _fake_token_embs(4)

        index = ColBERTIndex(tmp_path / "colbert")
        index.build(chunks, mock_model)

        meta = json.loads((tmp_path / "colbert" / "colbert_meta.json").read_text())
        assert set(meta.keys()) == {"c-000", "c-001", "c-002", "c-003"}

    def test_build_empty_chunks_returns_zero(self, tmp_path):
        mock_model = MagicMock()
        index = ColBERTIndex(tmp_path / "colbert")
        assert index.build([], mock_model) == 0
        mock_model.encode.assert_not_called()

    def test_maxsim_unknown_chunk_returns_zero(self, tmp_path):
        index = ColBERTIndex(tmp_path / "colbert")
        q = np.ones((3, 16), dtype=np.float32)
        assert index.maxsim_score(q, "does-not-exist") == 0.0

    def test_maxsim_score_is_positive_for_indexed_chunk(self, tmp_path):
        chunk = _make_chunk("c-001")
        mock_model = MagicMock()
        mock_model.encode.return_value = _fake_token_embs(1)

        index = ColBERTIndex(tmp_path / "colbert")
        index.build([chunk], mock_model)

        # Query with a non-zero vector — MaxSim should be > 0
        q = np.ones((2, 16), dtype=np.float32)
        score = index.maxsim_score(q, "c-001")
        assert score > 0.0

    def test_rescore_returns_score_for_each_id(self, tmp_path):
        chunks = [_make_chunk("c-001"), _make_chunk("c-002")]
        token_embs = _fake_token_embs(2)

        build_model = MagicMock()
        build_model.encode.return_value = token_embs

        index = ColBERTIndex(tmp_path / "colbert")
        index.build(chunks, build_model)

        query_model = MagicMock()
        query_model.encode.return_value = _fake_token_embs(1, n_tokens=3)

        scores = index.rescore("heel hook setup", ["c-001", "c-002"], query_model)
        assert set(scores.keys()) == {"c-001", "c-002"}
        assert all(isinstance(v, float) for v in scores.values())

    def test_rescore_unindexed_chunk_returns_zero(self, tmp_path):
        index = ColBERTIndex(tmp_path / "colbert")
        mock_model = MagicMock()
        mock_model.encode.return_value = _fake_token_embs(1)
        scores = index.rescore("armbar", ["missing-id"], mock_model)
        assert scores["missing-id"] == 0.0

    def test_len_reflects_indexed_count(self, tmp_path):
        chunks = [_make_chunk(f"c-{i:03d}") for i in range(3)]
        mock_model = MagicMock()
        mock_model.encode.return_value = _fake_token_embs(3)

        index = ColBERTIndex(tmp_path / "colbert")
        assert len(index) == 0
        index.build(chunks, mock_model)
        assert len(index) == 3

    def test_incremental_build_appends(self, tmp_path):
        """Building in two passes produces the same result as one pass."""
        chunk_a = _make_chunk("c-A")
        chunk_b = _make_chunk("c-B")

        emb_a = _fake_token_embs(1, n_tokens=5)
        emb_b = _fake_token_embs(1, n_tokens=7)

        model_a = MagicMock()
        model_a.encode.return_value = emb_a
        model_b = MagicMock()
        model_b.encode.return_value = emb_b

        index = ColBERTIndex(tmp_path / "colbert")
        index.build([chunk_a], model_a)
        index.build([chunk_b], model_b)

        assert len(index) == 2

        # Reload from disk — meta should persist
        index2 = ColBERTIndex(tmp_path / "colbert")
        assert len(index2) == 2

    def test_maxsim_offsets_are_correct_for_multiple_chunks(self, tmp_path):
        """Each chunk's MaxSim uses its own token vectors, not a neighbour's."""
        dim = 4
        # Two chunks with orthogonal token embeddings
        emb_a = np.array([[1, 0, 0, 0]], dtype=np.float32)  # 1 token
        emb_b = np.array([[0, 1, 0, 0]], dtype=np.float32)  # 1 token

        mock_model = MagicMock()
        mock_model.encode.return_value = [emb_a, emb_b]

        index = ColBERTIndex(tmp_path / "colbert")
        index.build([_make_chunk("c-A"), _make_chunk("c-B")], mock_model)

        q_a = np.array([[1, 0, 0, 0]], dtype=np.float32)  # aligned with c-A
        q_b = np.array([[0, 1, 0, 0]], dtype=np.float32)  # aligned with c-B

        score_a_on_a = index.maxsim_score(q_a, "c-A")
        score_a_on_b = index.maxsim_score(q_a, "c-B")
        assert score_a_on_a > score_a_on_b  # q_a should match c-A better

        score_b_on_b = index.maxsim_score(q_b, "c-B")
        score_b_on_a = index.maxsim_score(q_b, "c-A")
        assert score_b_on_b > score_b_on_a  # q_b should match c-B better


# HyDE — SearchEngine._expand_query_hyde
class TestHyDE:
    def _make_engine(self, hyde_enabled: bool = True):
        from jitsudb.search.engine import SearchEngine

        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda text, **kw: np.array([0.5] * 384, dtype=np.float32)
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]
        }
        return SearchEngine(
            embedding_model=mock_model,
            collection=mock_collection,
            hyde_enabled=hyde_enabled,
            beam_width=0,
        )

    def test_hyde_returns_none_when_ollama_unavailable(self):
        engine = self._make_engine(hyde_enabled=True)
        with patch("jitsudb.rag.synthesizer.RagSynthesizer") as MockSynth:
            MockSynth.return_value.is_available.return_value = False
            result = engine._expand_query_hyde("heel hook setup")
        assert result is None

    def test_hyde_embeds_passage_when_available(self):
        engine = self._make_engine(hyde_enabled=True)
        fake_passage = "From the outside heel hook position, control the knee..."

        with patch("jitsudb.rag.synthesizer.RagSynthesizer") as MockSynth:
            synth_instance = MockSynth.return_value
            synth_instance.is_available.return_value = True
            synth_instance._prompt_hyde.return_value = iter([fake_passage])

            result = engine._expand_query_hyde("heel hook setup")

        assert result is not None
        assert len(result) == 384

    def test_hyde_returns_none_on_empty_passage(self):
        engine = self._make_engine(hyde_enabled=True)
        with patch("jitsudb.rag.synthesizer.RagSynthesizer") as MockSynth:
            synth_instance = MockSynth.return_value
            synth_instance.is_available.return_value = True
            synth_instance._prompt_hyde.return_value = iter(["   "])  # whitespace only

            result = engine._expand_query_hyde("heel hook setup")
        assert result is None

    def test_hyde_returns_none_on_exception(self):
        engine = self._make_engine(hyde_enabled=True)
        with patch("jitsudb.rag.synthesizer.RagSynthesizer") as MockSynth:
            synth_instance = MockSynth.return_value
            synth_instance.is_available.return_value = True
            synth_instance._prompt_hyde.side_effect = Exception("connection refused")

            result = engine._expand_query_hyde("heel hook setup")
        assert result is None

    def test_hyde_embedding_differs_from_query_embedding(self):
        """When HyDE generates a different passage, the averaged embedding differs
        from the raw query embedding."""
        from jitsudb.search.engine import SearchEngine

        call_count = 0

        def encode_side_effect(text, **kw):
            nonlocal call_count
            call_count += 1
            # First call (query): all 0.1; second call (hyde passage): all 0.9
            val = 0.1 if call_count == 1 else 0.9
            return np.array([val] * 384, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.encode.side_effect = encode_side_effect
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]
        }

        engine = SearchEngine(
            embedding_model=mock_model,
            collection=mock_collection,
            hyde_enabled=True,
            beam_width=0,
        )

        with patch("jitsudb.rag.synthesizer.RagSynthesizer") as MockSynth:
            synth_instance = MockSynth.return_value
            synth_instance.is_available.return_value = True
            synth_instance._prompt_hyde.return_value = iter(["heel hook passage text"])

            # Capture what the collection was actually queried with
            queried_embedding = None
            _empty = {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

            def capture_query(**kwargs):
                nonlocal queried_embedding
                queried_embedding = kwargs.get("query_embeddings", [None])[0]
                return _empty

            mock_collection.query.side_effect = capture_query
            engine.search("heel hook setup")

        # Raw query embedding is all 0.1; HyDE passage is all 0.9
        # Average should be all 0.5 — different from raw 0.1
        assert queried_embedding is not None
        avg_val = float(np.mean(queried_embedding))
        assert abs(avg_val - 0.5) < 0.01, f"Expected ~0.5 (HyDE average), got {avg_val}"
        assert abs(avg_val - 0.1) > 0.01, "Embedding should differ from raw query embedding"

    def test_hyde_disabled_does_not_call_synthesizer(self):
        """When hyde_enabled=False, _expand_query_hyde is never called."""
        engine = self._make_engine(hyde_enabled=False)
        with patch("jitsudb.rag.synthesizer.RagSynthesizer") as MockSynth:
            engine.search("heel hook setup")
        MockSynth.assert_not_called()


class TestM9Config:
    def test_hyde_enabled_defaults_false(self):
        from jitsudb.config import SearchConfig
        assert SearchConfig().hyde_enabled is False

    def test_colbert_enabled_defaults_false(self):
        from jitsudb.config import SearchConfig
        assert SearchConfig().colbert_enabled is False

    def test_hyde_enabled_can_be_set(self):
        from jitsudb.config import SearchConfig
        cfg = SearchConfig(hyde_enabled=True)
        assert cfg.hyde_enabled is True
