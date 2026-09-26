"""HDBSCAN clustering and click-through reranker."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            start_time REAL NOT NULL DEFAULT 0,
            end_time REAL NOT NULL DEFAULT 60,
            transcript_text TEXT NOT NULL DEFAULT '',
            thumbnail_path TEXT,
            context TEXT,
            gi_nogi TEXT NOT NULL DEFAULT 'both'
        );
        CREATE TABLE videos (
            id TEXT PRIMARY KEY,
            title TEXT,
            instructor TEXT,
            file_path TEXT,
            duration_seconds REAL,
            gi_nogi TEXT,
            primary_category TEXT,
            status TEXT,
            added_at TEXT,
            processed_at TEXT
        );
        CREATE TABLE taxonomy_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            label_type TEXT NOT NULL,
            label_value TEXT NOT NULL
        );
        CREATE TABLE uncertain_chunks (
            chunk_id TEXT PRIMARY KEY,
            entropy REAL NOT NULL,
            top_label TEXT NOT NULL,
            second_label TEXT,
            reviewed INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE chunk_label_proposals (
            chunk_id TEXT PRIMARY KEY,
            cluster_id INTEGER NOT NULL,
            proposed_positions TEXT NOT NULL DEFAULT '[]',
            proposed_intents TEXT NOT NULL DEFAULT '[]',
            proposed_techniques TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE retrieval_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            query_embedding BLOB NOT NULL,
            chunk_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            clicked INTEGER NOT NULL DEFAULT 0,
            watch_duration_sec REAL NOT NULL DEFAULT 0.0,
            latency_ms REAL NOT NULL DEFAULT 0.0,
            logged_at TEXT NOT NULL
        );
    """)
    return conn


def _insert_chunk(conn, chunk_id, video_id="vid1", labels=None):
    conn.execute(
        "INSERT OR IGNORE INTO videos VALUES (?,?,?,?,?,?,?,?,?,?)",
        (video_id, "T", "I", "f", 60, "nogi", "passing", "indexed", "2025-01-01", None),
    )
    conn.execute(
        "INSERT INTO chunks VALUES (?,?,0,60,'',NULL,'top','nogi')",
        (chunk_id, video_id),
    )
    if labels:
        for ltype, lvalue in labels:
            conn.execute(
                "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?,?,?)",
                (chunk_id, ltype, lvalue),
            )
    conn.commit()


class TestMajorityLabels:
    def test_empty_cluster_returns_zero_confidence(self):
        from jitsudb.pipeline.cluster import _majority_labels
        positions, intents, techniques, confidence = _majority_labels([], [], {})
        assert positions == []
        assert intents == []
        assert techniques == []
        assert confidence == 0.0

    def test_single_confident_chunk_returns_its_labels(self):
        from jitsudb.pipeline.cluster import _majority_labels
        label_map = {"c1": {"position": ["mount"], "intent": ["attack"], "technique": []}}
        positions, intents, techniques, confidence = _majority_labels(
            [0], ["c1"], label_map, min_support=1
        )
        assert "mount" in positions
        assert "attack" in intents
        assert confidence > 0.0

    def test_majority_wins(self):
        from jitsudb.pipeline.cluster import _majority_labels
        label_map = {
            "c1": {"position": ["mount"], "intent": ["attack"], "technique": []},
            "c2": {"position": ["mount"], "intent": ["defense"], "technique": []},
            "c3": {"position": ["side_control"], "intent": ["attack"], "technique": []},
        }
        positions, intents, techniques, confidence = _majority_labels(
            [0, 1, 2], ["c1", "c2", "c3"], label_map, min_support=2
        )
        assert "mount" in positions
        assert "attack" in intents
        assert "side_control" not in positions  # only 1 occurrence, below min_support

    def test_no_confident_chunks_skipped(self):
        from jitsudb.pipeline.cluster import _majority_labels
        label_map = {}  # none are confident
        _, _, _, confidence = _majority_labels([0, 1], ["c1", "c2"], label_map)
        assert confidence == 0.0


class TestRunClusterProposals:
    def _make_mock_collection(self, n=20):
        embs = np.random.randn(n, 4).astype(np.float32)
        ids = [f"c{i}" for i in range(n)]
        mock = MagicMock()
        mock.count.return_value = n
        mock.get.return_value = {"ids": ids, "embeddings": embs.tolist()}
        return mock, ids

    def test_returns_zero_when_too_few_chunks(self):
        from jitsudb.pipeline.cluster import run_cluster_proposals
        conn = _make_db()
        mock_coll = MagicMock()
        mock_coll.count.return_value = 3
        mock_coll.get.return_value = {"ids": ["c0", "c1", "c2"], "embeddings": [[1,0,0,0]]*3}
        written = run_cluster_proposals(mock_coll, conn, min_cluster_size=5)
        assert written == 0

    def test_proposals_written_for_clustered_chunks(self):
        from jitsudb.pipeline.cluster import run_cluster_proposals
        conn = _make_db()

        n = 30
        for i in range(n):
            _insert_chunk(conn, f"c{i}", labels=[("position", "mount"), ("intent", "attack")])

        # Two tight clusters + noise
        embs = np.zeros((n, 4), dtype=np.float32)
        embs[:10] = [1, 0, 0, 0]
        embs[10:20] = [0, 1, 0, 0]
        # last 10 are noise (orthogonal singles)
        for i in range(20, n):
            embs[i] = np.random.randn(4).astype(np.float32) * 10

        ids = [f"c{i}" for i in range(n)]
        mock_coll = MagicMock()
        mock_coll.count.return_value = n
        mock_coll.get.return_value = {"ids": ids, "embeddings": embs.tolist()}

        written = run_cluster_proposals(mock_coll, conn, min_cluster_size=5, min_samples=2)
        assert written > 0
        count = conn.execute("SELECT COUNT(*) FROM chunk_label_proposals").fetchone()[0]
        assert count == written

    def test_upsert_on_rerun(self):
        from jitsudb.pipeline.cluster import run_cluster_proposals
        conn = _make_db()
        n = 12
        for i in range(n):
            _insert_chunk(conn, f"c{i}", labels=[("position", "mount")])
        embs = np.zeros((n, 4), dtype=np.float32)
        embs[:n//2] = [1, 0, 0, 0]
        embs[n//2:] = [0, 1, 0, 0]
        ids = [f"c{i}" for i in range(n)]
        mock_coll = MagicMock()
        mock_coll.count.return_value = n
        mock_coll.get.return_value = {"ids": ids, "embeddings": embs.tolist()}

        w1 = run_cluster_proposals(mock_coll, conn, min_cluster_size=4, min_samples=2)
        w2 = run_cluster_proposals(mock_coll, conn, min_cluster_size=4, min_samples=2)
        # Row count stays same (upsert, not insert)
        assert conn.execute("SELECT COUNT(*) FROM chunk_label_proposals").fetchone()[0] == w1


class TestGetProposals:
    def test_returns_none_when_absent(self):
        from jitsudb.pipeline.cluster import get_proposals
        conn = _make_db()
        _insert_chunk(conn, "c1")
        assert get_proposals("c1", conn) is None

    def test_returns_proposals_when_present(self):
        from jitsudb.pipeline.cluster import get_proposals
        conn = _make_db()
        _insert_chunk(conn, "c1")
        conn.execute(
            """INSERT INTO chunk_label_proposals
               (chunk_id, cluster_id, proposed_positions, proposed_intents,
                proposed_techniques, confidence, created_at)
               VALUES ('c1', 0, '["mount"]', '["attack"]', '[]', 0.8, '2025-01-01')"""
        )
        conn.commit()
        p = get_proposals("c1", conn)
        assert p is not None
        assert p["positions"] == ["mount"]
        assert p["intents"] == ["attack"]
        assert p["confidence"] == pytest.approx(0.8)


class TestClickLog:
    def _make_search_result(self, chunk_id):
        from pathlib import Path
        from jitsudb.models import Chunk, Video, SearchResult
        chunk = Chunk(
            id=chunk_id, video_id="v1", start_time=0, end_time=60,
            transcript_text="text", position=[], context="top",
            intent=[], techniques=[], gi_nogi="nogi",
        )
        video = Video(
            id="v1", title="T", instructor="I", file_path=Path("f.mp4"),
            duration_seconds=60, gi_nogi="nogi", primary_category="passing",
            status="indexed", added_at="2025-01-01",
        )
        return SearchResult(
            chunk=chunk, video=video, relevance_score=0.9, snippet="text",
            detected_filters={},
        )

    def test_impression_writes_rows(self):
        from jitsudb.search.click_reranker import log_impression
        conn = _make_db()
        _insert_chunk(conn, "c1")
        _insert_chunk(conn, "c2")
        results = [self._make_search_result("c1"), self._make_search_result("c2")]
        log_impression("sess1", [1.0, 0.0, 0.0, 0.0], results, conn)
        count = conn.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0]
        assert count == 2

    def test_click_marks_row(self):
        from jitsudb.search.click_reranker import log_impression, log_click
        conn = _make_db()
        _insert_chunk(conn, "c1")
        results = [self._make_search_result("c1")]
        log_impression("sess1", [1.0, 0.0, 0.0, 0.0], results, conn)
        log_click("sess1", "c1", conn)
        clicked = conn.execute(
            "SELECT clicked FROM retrieval_log WHERE chunk_id = 'c1'"
        ).fetchone()[0]
        assert clicked == 1

    def test_click_only_affects_own_session(self):
        from jitsudb.search.click_reranker import log_impression, log_click
        conn = _make_db()
        _insert_chunk(conn, "c1")
        results = [self._make_search_result("c1")]
        log_impression("sessA", [1.0, 0.0, 0.0, 0.0], results, conn)
        log_impression("sessB", [1.0, 0.0, 0.0, 0.0], results, conn)
        log_click("sessA", "c1", conn)
        rows = conn.execute(
            "SELECT session_id, clicked FROM retrieval_log WHERE chunk_id = 'c1'"
        ).fetchall()
        clicks_by_sess = {r["session_id"]: r["clicked"] for r in rows}
        assert clicks_by_sess["sessA"] == 1
        assert clicks_by_sess["sessB"] == 0


class TestClickBoost:
    def _make_result(self, chunk_id, score=0.5):
        from pathlib import Path
        from jitsudb.models import Chunk, Video, SearchResult
        chunk = Chunk(
            id=chunk_id, video_id="v1", start_time=0, end_time=60,
            transcript_text="text", position=[], context="top",
            intent=[], techniques=[], gi_nogi="nogi",
        )
        video = Video(
            id="v1", title="T", instructor="I", file_path=Path("f.mp4"),
            duration_seconds=60, gi_nogi="nogi", primary_category="passing",
            status="indexed", added_at="2025-01-01",
        )
        return SearchResult(
            chunk=chunk, video=video, relevance_score=score, snippet="text",
            detected_filters={},
        )

    def test_no_clicks_returns_same_order(self):
        from jitsudb.search.click_reranker import boost
        conn = _make_db()
        results = [self._make_result("c1", 0.9), self._make_result("c2", 0.5)]
        boosted = boost([1.0, 0.0], results, conn)
        assert [r.chunk.id for r in boosted] == ["c1", "c2"]

    def test_clicked_chunk_rises(self):
        from jitsudb.search.click_reranker import log_impression, log_click, boost
        conn = _make_db()
        _insert_chunk(conn, "c1")
        _insert_chunk(conn, "c2")

        q_emb = [1.0, 0.0, 0.0, 0.0]
        results_for_log = [self._make_result("c1"), self._make_result("c2")]
        log_impression("s1", q_emb, results_for_log, conn)
        log_click("s1", "c2", conn)  # c2 was clicked

        # c1 has higher base score, but c2 was clicked → c2 should rise after boost
        search_results = [self._make_result("c1", 0.6), self._make_result("c2", 0.5)]
        boosted = boost(q_emb, search_results, conn)
        ids = [r.chunk.id for r in boosted]
        assert ids[0] == "c2"

    def test_boost_capped(self):
        from jitsudb.search.click_reranker import log_impression, log_click, boost, _BOOST_WEIGHT
        conn = _make_db()
        _insert_chunk(conn, "c1")
        q_emb = [1.0, 0.0, 0.0, 0.0]
        results_for_log = [self._make_result("c1")]
        log_impression("s1", q_emb, results_for_log, conn)
        log_click("s1", "c1", conn)

        r = self._make_result("c1", 0.5)
        boosted = boost(q_emb, [r], conn)
        assert boosted[0].relevance_score <= 0.5 + _BOOST_WEIGHT + 1e-6
