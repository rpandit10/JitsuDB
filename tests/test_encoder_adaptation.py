"""Domain-adapted encoder: triplet mining and re-embedding."""

from __future__ import annotations

import sqlite3

import pytest

from scripts.mine_negatives import mine_triplets
from scripts.reembed import reembed


def _insert_video(conn: sqlite3.Connection, video_id: str) -> None:
    conn.execute(
        "INSERT INTO videos (id, title, instructor, file_path, duration_seconds, "
        "gi_nogi, primary_category, status, added_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (video_id, "Test", "Tester", "/fake/v.mp4", 300.0, "nogi", "Passing",
         "indexed", "2024-01-01T00:00:00"),
    )


def _insert_chunk(conn: sqlite3.Connection, chunk_id: str, video_id: str, text: str) -> None:
    conn.execute(
        "INSERT INTO chunks (id, video_id, start_time, end_time, transcript_text, gi_nogi) "
        "VALUES (?,?,?,?,?,?)",
        (chunk_id, video_id, 0.0, 45.0, text, "nogi"),
    )


def _label(conn: sqlite3.Connection, chunk_id: str, label_type: str, label_value: str) -> None:
    conn.execute(
        "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?,?,?)",
        (chunk_id, label_type, label_value),
    )


@pytest.fixture
def labeled_db(tmp_db: sqlite3.Connection) -> sqlite3.Connection:
    """DB with two videos, three positions, and multiple technique sub-groups.

    Layout (position → technique → chunk_ids):
      guard    → armbar  → [g-a-1, g-a-2]
      guard    → kimura  → [g-k-1, g-k-2]
      mount    → armbar  → [m-a-1, m-a-2]
      mount    → keylock → [m-k-1, m-k-2]
      back     → rnc     → [b-r-1]           ← only 1 chunk: no triplets (anchor=positive)
      side_ctl → n/a     → no technique labels ← no triplets
    """
    vid = "vid-mine"
    _insert_video(tmp_db, vid)

    specs = [
        ("g-a-1", "guard", "armbar",  "Guard armbar text one"),
        ("g-a-2", "guard", "armbar",  "Guard armbar text two"),
        ("g-k-1", "guard", "kimura",  "Guard kimura text one"),
        ("g-k-2", "guard", "kimura",  "Guard kimura text two"),
        ("m-a-1", "mount", "armbar",  "Mount armbar text one"),
        ("m-a-2", "mount", "armbar",  "Mount armbar text two"),
        ("m-k-1", "mount", "keylock", "Mount keylock text one"),
        ("m-k-2", "mount", "keylock", "Mount keylock text two"),
        ("b-r-1", "back",  "rnc",     "Back rnc text one"),  # single chunk → skipped
    ]
    sc_id = "sc-no-tech"
    _insert_chunk(tmp_db, sc_id, vid, "Side control chunk")
    _label(tmp_db, sc_id, "position", "side_control")  # no technique label

    for chunk_id, position, technique, text in specs:
        _insert_chunk(tmp_db, chunk_id, vid, text)
        _label(tmp_db, chunk_id, "position", position)
        _label(tmp_db, chunk_id, "technique", technique)

    tmp_db.commit()
    return tmp_db


class TestMineTriplets:
    def test_returns_list_of_dicts(self, labeled_db):
        triplets = mine_triplets(labeled_db, seed=0)
        assert isinstance(triplets, list)
        assert len(triplets) > 0

    def test_triplet_structure(self, labeled_db):
        triplets = mine_triplets(labeled_db, seed=0)
        for t in triplets:
            assert set(t.keys()) == {"anchor", "positive", "negative"}
            assert isinstance(t["anchor"], str)
            assert isinstance(t["positive"], str)
            assert isinstance(t["negative"], str)

    def test_anchor_differs_from_negative(self, labeled_db):
        """Hard negative must come from a different technique — different text."""
        triplets = mine_triplets(labeled_db, seed=0)
        for t in triplets:
            assert t["anchor"] != t["negative"]

    def test_guard_and_mount_both_produce_triplets(self, labeled_db):
        """Both guard and mount have ≥2 techniques → both contribute triplets."""
        triplets = mine_triplets(labeled_db, seed=0)
        texts = {t["anchor"] for t in triplets}
        has_guard = any("Guard" in tx for tx in texts)
        has_mount = any("Mount" in tx for tx in texts)
        assert has_guard, "Expected guard position to produce triplets"
        assert has_mount, "Expected mount position to produce triplets"

    def test_single_chunk_technique_skipped(self, labeled_db):
        """back/rnc has only 1 chunk — cannot form anchor+positive, must be skipped."""
        triplets = mine_triplets(labeled_db, seed=0)
        for t in triplets:
            assert "Back rnc" not in t["anchor"]

    def test_position_without_technique_labels_skipped(self, labeled_db):
        """side_control chunk has no technique label → excluded."""
        triplets = mine_triplets(labeled_db, seed=0)
        for t in triplets:
            assert "Side control" not in t["anchor"]

    def test_limit_respected(self, labeled_db):
        all_triplets = mine_triplets(labeled_db, seed=0)
        limited = mine_triplets(labeled_db, seed=0, limit=2)
        assert len(limited) == min(2, len(all_triplets))

    def test_empty_db_returns_empty(self, tmp_db):
        """Database with no taxonomy labels produces no triplets."""
        triplets = mine_triplets(tmp_db, seed=0)
        assert triplets == []

    def test_seed_gives_reproducible_order(self, labeled_db):
        t1 = mine_triplets(labeled_db, seed=99)
        t2 = mine_triplets(labeled_db, seed=99)
        assert [t["anchor"] for t in t1] == [t["anchor"] for t in t2]


class TestReembed:
    def test_upserts_all_chunks(self, tmp_db):
        """reembed calls collection.upsert with every chunk in the DB."""
        from unittest.mock import MagicMock

        import numpy as np

        vid = "vid-reembed"
        _insert_video(tmp_db, vid)
        for i in range(3):
            _insert_chunk(tmp_db, f"re-chunk-{i:03d}", vid, f"Chunk {i} text")
        tmp_db.commit()

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((3, 384), dtype="float32")

        mock_collection = MagicMock()

        n = reembed(tmp_db, mock_model, mock_collection, batch_size=32)

        assert n == 3
        mock_model.encode.assert_called_once()
        mock_collection.upsert.assert_called_once()

        call_kwargs = mock_collection.upsert.call_args
        upserted_ids = call_kwargs.kwargs.get("ids") or call_kwargs.args[0]
        assert set(upserted_ids) == {"re-chunk-000", "re-chunk-001", "re-chunk-002"}

    def test_empty_db_returns_zero(self, tmp_db):
        from unittest.mock import MagicMock
        n = reembed(tmp_db, MagicMock(), MagicMock())
        assert n == 0

    def test_document_prefix_prepended(self, tmp_db):
        """reembed passes prefixed texts to model.encode when document_prefix is set."""
        from unittest.mock import MagicMock
        import numpy as np

        vid = "vid-prefix"
        _insert_video(tmp_db, vid)
        _insert_chunk(tmp_db, "px-1", vid, "triangle choke setup")
        tmp_db.commit()

        captured = {}
        mock_model = MagicMock()
        def fake_encode(texts, **kwargs):
            captured["texts"] = texts
            return np.zeros((len(texts), 768), dtype="float32")
        mock_model.encode.side_effect = fake_encode

        reembed(tmp_db, mock_model, MagicMock(), document_prefix="search_document: ")

        assert captured["texts"][0].startswith("search_document: ")


class TestEmbeddingConfig:
    def test_encoder_path_defaults_to_none(self):
        from jitsudb.config import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert cfg.encoder_path is None

    def test_ft_collection_default(self):
        from jitsudb.config import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert cfg.ft_collection == "text_ft"

    def test_encoder_path_set(self):
        from jitsudb.config import EmbeddingConfig
        cfg = EmbeddingConfig(encoder_path="models/bjj-nomic-ft/")
        assert cfg.encoder_path == "models/bjj-nomic-ft/"

    def test_nomic_defaults(self):
        from jitsudb.config import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert cfg.model_name == "nomic-ai/nomic-embed-text-v1.5"
        assert cfg.embedding_dim == 768
        assert cfg.query_prefix == "search_query: "
        assert cfg.document_prefix == "search_document: "
        assert cfg.trust_remote_code is True
        assert cfg.collection_name == "jitsudb_chunks"

    def test_prefixes_overridable(self):
        from jitsudb.config import EmbeddingConfig
        cfg = EmbeddingConfig(query_prefix="", document_prefix="")
        assert cfg.query_prefix == ""
        assert cfg.document_prefix == ""
