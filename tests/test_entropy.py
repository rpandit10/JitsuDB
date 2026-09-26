"""entropy scoring and drift detection."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jitsudb.pipeline.entropy import (
    _category_entropy,
    compute_chunk_entropy,
    score_and_store_chunks,
)
from jitsudb.pipeline.drift import _cosine_distance, detect_drift


def _make_chunk(chunk_id="c1", position=None, intent=None, techniques=None, context="top"):
    from jitsudb.models import Chunk
    from pathlib import Path
    return Chunk(
        id=chunk_id,
        video_id="vid-1",
        start_time=0.0,
        end_time=45.0,
        transcript_text="dummy text",
        position=position or [],
        intent=intent or [],
        techniques=techniques or [],
        context=context,
        gi_nogi="nogi",
    )


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            video_id TEXT,
            start_time REAL,
            end_time REAL,
            transcript_text TEXT,
            context TEXT,
            gi_nogi TEXT
        );
        CREATE TABLE uncertain_chunks (
            chunk_id TEXT PRIMARY KEY,
            entropy REAL NOT NULL,
            top_label TEXT NOT NULL,
            second_label TEXT,
            reviewed INTEGER NOT NULL DEFAULT 0
        );
    """)
    return conn


class TestCategoryEntropy:
    def test_zero_matches_returns_one(self):
        assert _category_entropy(0, 14) == 1.0

    def test_one_match_returns_zero(self):
        assert _category_entropy(1, 14) == 0.0

    def test_entropy_increases_with_matches(self):
        e2 = _category_entropy(2, 14)
        e5 = _category_entropy(5, 14)
        assert 0.0 < e2 < e5 < 1.0

    def test_all_matches_returns_one(self):
        assert _category_entropy(14, 14) == 1.0


class TestComputeChunkEntropy:
    def test_single_position_low_entropy(self):
        chunk = _make_chunk(position=["mount"], intent=["attack"], context="top")
        e = compute_chunk_entropy(chunk)
        assert e < 0.4

    def test_no_labels_high_entropy(self):
        chunk = _make_chunk(position=[], intent=[], techniques=[], context="neutral")
        e = compute_chunk_entropy(chunk)
        assert e > 0.6

    def test_many_positions_raises_entropy(self):
        chunk_single = _make_chunk(position=["mount"], intent=[], context="top")
        chunk_many   = _make_chunk(position=["mount", "side_control", "half_guard", "closed_guard"], intent=[], context="top")
        assert compute_chunk_entropy(chunk_many) > compute_chunk_entropy(chunk_single)

    def test_neutral_context_adds_entropy(self):
        chunk_clear   = _make_chunk(position=["mount"], intent=["attack"], context="top")
        chunk_neutral = _make_chunk(position=["mount"], intent=["attack"], context="neutral")
        assert compute_chunk_entropy(chunk_neutral) > compute_chunk_entropy(chunk_clear)

    def test_entropy_bounded_zero_to_one(self):
        for pos in [[], ["mount"], ["mount", "side_control"]]:
            for intent in [[], ["attack", "defense"]]:
                chunk = _make_chunk(position=pos, intent=intent)
                e = compute_chunk_entropy(chunk)
                assert 0.0 <= e <= 1.0


class TestScoreAndStoreChunks:
    def test_high_entropy_chunk_stored(self):
        conn = _make_db()
        chunk = _make_chunk("uc1", position=[], intent=[], context="neutral")
        written = score_and_store_chunks([chunk], conn, entropy_threshold=0.3)
        assert written == 1
        row = conn.execute("SELECT * FROM uncertain_chunks WHERE chunk_id = 'uc1'").fetchone()
        assert row is not None
        assert row["entropy"] > 0.3

    def test_low_entropy_chunk_not_stored(self):
        conn = _make_db()
        chunk = _make_chunk("lc1", position=["mount"], intent=["attack"], context="top")
        written = score_and_store_chunks([chunk], conn, entropy_threshold=0.8)
        assert written == 0
        assert conn.execute("SELECT COUNT(*) FROM uncertain_chunks").fetchone()[0] == 0

    def test_returns_count_of_stored(self):
        conn = _make_db()
        chunks = [
            _make_chunk("a", position=[], intent=[], context="neutral"),   # high entropy
            _make_chunk("b", position=["mount"], intent=["attack"], context="top"),  # low entropy
            _make_chunk("c", position=[], intent=[], context="neutral"),   # high entropy
        ]
        written = score_and_store_chunks(chunks, conn, entropy_threshold=0.3)
        assert written == 2

    def test_top_label_is_first_position(self):
        conn = _make_db()
        chunk = _make_chunk("tl1", position=["side_control", "mount"], intent=[], context="neutral")
        score_and_store_chunks([chunk], conn, entropy_threshold=0.0)
        row = conn.execute("SELECT top_label, second_label FROM uncertain_chunks").fetchone()
        assert row["top_label"] == "side_control"
        assert row["second_label"] == "mount"


class TestCosineDistance:
    def test_identical_vectors_distance_zero(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine_distance(v, v) == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_vectors_distance_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert _cosine_distance(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_zero_vector_returns_one(self):
        a = np.zeros(4, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine_distance(a, b) == 1.0


class TestDetectDrift:
    def _make_drift_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE chunks (id TEXT PRIMARY KEY, video_id TEXT);
            CREATE TABLE videos (id TEXT PRIMARY KEY, primary_category TEXT);
            CREATE TABLE drift_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                taxonomy_class TEXT,
                distance REAL,
                baseline_at TEXT,
                detected_at TEXT
            );
        """)
        return conn

    def test_no_baseline_returns_none(self):
        conn = self._make_drift_db()
        mock_coll = MagicMock()
        mock_coll.get.return_value = {"ids": [], "embeddings": []}
        chunks = [_make_chunk()]
        result = detect_drift("vid-1", "passing", chunks, mock_coll, conn)
        assert result is None

    def test_drift_written_when_above_threshold(self):
        from jitsudb.pipeline.drift import DRIFT_THRESHOLD
        conn = self._make_drift_db()

        # Populate baseline: 15 chunks in category
        for i in range(15):
            conn.execute("INSERT INTO chunks VALUES (?, ?)", (f"base{i}", "vid-base"))
            conn.execute("INSERT OR IGNORE INTO videos VALUES (?, ?)", ("vid-base", "passing"))

        # centroid at [1,0]; new chunk at [0,1] → distance = 1.0 > threshold
        base_embs = [[1.0, 0.0]] * 15
        new_embs  = [[0.0, 1.0]]

        mock_coll = MagicMock()
        mock_coll.get.side_effect = [
            {"ids": [f"base{i}" for i in range(15)], "embeddings": base_embs},
            {"ids": ["c1"], "embeddings": new_embs},
        ]

        chunk = _make_chunk("c1")
        result = detect_drift("vid-1", "passing", [chunk], mock_coll, conn)

        assert result is not None
        assert result > DRIFT_THRESHOLD
        assert conn.execute("SELECT COUNT(*) FROM drift_log").fetchone()[0] == 1

    def test_no_drift_log_when_below_threshold(self):
        conn = self._make_drift_db()
        for i in range(15):
            conn.execute("INSERT INTO chunks VALUES (?, ?)", (f"base{i}", "vid-base"))
            conn.execute("INSERT OR IGNORE INTO videos VALUES (?, ?)", ("vid-base", "passing"))

        # Nearly identical vectors → distance ≈ 0
        vec = [1.0, 0.0]
        mock_coll = MagicMock()
        mock_coll.get.side_effect = [
            {"ids": [f"base{i}" for i in range(15)], "embeddings": [vec] * 15},
            {"ids": ["c1"], "embeddings": [vec]},
        ]

        chunk = _make_chunk("c1")
        detect_drift("vid-1", "passing", [chunk], mock_coll, conn)
        assert conn.execute("SELECT COUNT(*) FROM drift_log").fetchone()[0] == 0
