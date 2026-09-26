"""Tests for search timing, metrics endpoint, and eval harness utilities."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from jitsudb.api.app import create_app
from jitsudb.api.dependencies import get_db, get_search_engine


class TestSearchTiming:
    def test_single_stage_recorded(self):
        from jitsudb.search.timing import SearchTiming
        t = SearchTiming()
        t.start("embed")
        time.sleep(0.01)
        t.stop("embed")
        assert "embed" in t.stages
        assert t.stages["embed"] >= 10.0  # ≥10 ms

    def test_total_ms_sums_stages(self):
        from jitsudb.search.timing import SearchTiming
        t = SearchTiming()
        t.start("a")
        t.stop("a")
        t.start("b")
        t.stop("b")
        assert t.total_ms == pytest.approx(t.stages["a"] + t.stages["b"], abs=0.01)

    def test_stop_without_start_is_noop(self):
        from jitsudb.search.timing import SearchTiming
        t = SearchTiming()
        t.stop("phantom")  # must not raise
        assert "phantom" not in t.stages

    def test_as_dict_structure(self):
        from jitsudb.search.timing import SearchTiming
        t = SearchTiming()
        t.start("x")
        t.stop("x")
        d = t.as_dict()
        assert "stages_ms" in d and "total_ms" in d
        assert d["stages_ms"]["x"] == t.stages["x"]


class TestMetricsHelpers:
    def test_percentile_empty(self):
        from jitsudb.api.routes.metrics import _percentile
        assert _percentile([], 50) == 0.0

    def test_percentile_single(self):
        from jitsudb.api.routes.metrics import _percentile
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 99) == 42.0

    def test_percentile_median(self):
        from jitsudb.api.routes.metrics import _percentile
        vals = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
        assert _percentile(vals, 50) == 30.0

    def test_percentile_p95(self):
        from jitsudb.api.routes.metrics import _percentile
        vals = sorted(float(i) for i in range(1, 101))
        assert _percentile(vals, 95) == 95.0


# Metrics API endpoint
def _make_metrics_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE chunks (id TEXT PRIMARY KEY, video_id TEXT, start_time REAL,
            end_time REAL, transcript_text TEXT, thumbnail_path TEXT,
            context TEXT, gi_nogi TEXT);
        CREATE TABLE videos (id TEXT PRIMARY KEY, title TEXT, instructor TEXT,
            file_path TEXT, duration_seconds REAL, gi_nogi TEXT,
            primary_category TEXT, status TEXT, added_at TEXT, processed_at TEXT);
        CREATE TABLE uncertain_chunks (chunk_id TEXT PRIMARY KEY, entropy REAL,
            top_label TEXT, second_label TEXT, reviewed INTEGER DEFAULT 0);
        CREATE TABLE retrieval_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, query_embedding BLOB, chunk_id TEXT,
            rank INTEGER, clicked INTEGER DEFAULT 0,
            watch_duration_sec REAL DEFAULT 0.0,
            latency_ms REAL DEFAULT 0.0,
            logged_at TEXT
        );
    """)
    return conn


class TestMetricsEndpoint:
    @pytest.fixture
    def client(self):
        app = create_app()
        db = _make_metrics_db()

        # Insert some test data
        db.execute(
            "INSERT INTO videos VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("v1", "T", "I", "f", 60, "nogi", "passing", "indexed", "2025-01-01", None),
        )
        db.execute(
            "INSERT INTO chunks VALUES (?,?,0,60,'txt',NULL,'top','nogi')", ("c1", "v1")
        )
        # Two search impressions with latency
        db.execute(
            "INSERT INTO retrieval_log VALUES (?,?,?,?,?,?,?,?,?)",
            (None, "s1", b"\x00", "c1", 0, 0, 0.0, 120.0, "2025-01-01T00:00:00"),
        )
        db.execute(
            "INSERT INTO retrieval_log VALUES (?,?,?,?,?,?,?,?,?)",
            (None, "s2", b"\x00", "c1", 0, 1, 0.0, 80.0, "2025-01-01T00:01:00"),
        )
        db.commit()

        mock_engine = MagicMock()
        mock_engine.search.return_value = []
        mock_engine._last_timing = {"embed": 10.0, "ann_text": 25.0}
        mock_engine._last_total_ms = 35.0

        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_search_engine] = lambda: mock_engine
        return TestClient(app)

    def test_returns_200(self, client):
        res = client.get("/api/metrics")
        assert res.status_code == 200

    def test_latency_section_present(self, client):
        data = client.get("/api/metrics").json()
        assert "latency" in data
        assert data["latency"]["n_queries"] == 2

    def test_latency_percentiles_correct(self, client):
        data = client.get("/api/metrics").json()
        lat = data["latency"]
        assert lat["p50_ms"] in (80.0, 120.0)  # one of the two values
        assert lat["max_ms"] == 120.0

    def test_last_search_stage_breakdown(self, client):
        data = client.get("/api/metrics").json()
        assert data["last_search"]["total_ms"] == 35.0
        assert "embed" in data["last_search"]["stages_ms"]

    def test_index_section(self, client):
        data = client.get("/api/metrics").json()
        idx = data["index"]
        assert idx["total_chunks"] == 1
        assert idx["total_videos"] == 1
        assert idx["indexed_videos"] == 1
        assert idx["total_impressions_logged"] == 2
        assert idx["total_clicks_logged"] == 1

    def test_vram_section_present(self, client):
        data = client.get("/api/metrics").json()
        assert "vram" in data
        assert "available" in data["vram"]


# Eval harness metric helpers
class TestEvalMetrics:
    def test_precision_perfect(self):
        from eval.run_eval import _precision_at_k
        assert _precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)

    def test_precision_partial(self):
        from eval.run_eval import _precision_at_k
        # 1 of 3 retrieved are relevant
        assert _precision_at_k(["a", "x", "y"], {"a"}, 3) == pytest.approx(1 / 3)

    def test_precision_miss(self):
        from eval.run_eval import _precision_at_k
        assert _precision_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_recall_perfect(self):
        from eval.run_eval import _recall_at_k
        assert _recall_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_recall_partial(self):
        from eval.run_eval import _recall_at_k
        # retrieved a but not b; |relevant|=2
        assert _recall_at_k(["a", "x"], {"a", "b"}, 2) == pytest.approx(0.5)

    def test_recall_empty_relevant(self):
        from eval.run_eval import _recall_at_k
        assert _recall_at_k(["a"], set(), 1) == 0.0

    def test_ndcg_perfect(self):
        from eval.run_eval import _ndcg_at_k
        assert _ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_ndcg_rank_matters(self):
        from eval.run_eval import _ndcg_at_k
        # Relevant at rank 2 vs rank 1 — rank 2 should score lower
        score_rank1 = _ndcg_at_k(["a", "x"], {"a"}, 2)
        score_rank2 = _ndcg_at_k(["x", "a"], {"a"}, 2)
        assert score_rank1 > score_rank2

    def test_mrr_first_hit(self):
        from eval.run_eval import _mrr
        assert _mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_mrr_second_hit(self):
        from eval.run_eval import _mrr
        assert _mrr(["x", "a", "b"], {"a"}) == pytest.approx(0.5)

    def test_mrr_no_hit(self):
        from eval.run_eval import _mrr
        assert _mrr(["x", "y"], {"a"}) == 0.0


# Synthetic query generation
def _make_eval_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE chunks (id TEXT PRIMARY KEY, video_id TEXT NOT NULL,
            start_time REAL, end_time REAL, transcript_text TEXT,
            thumbnail_path TEXT, context TEXT, gi_nogi TEXT);
        CREATE TABLE taxonomy_labels (id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL, label_type TEXT NOT NULL, label_value TEXT NOT NULL);
    """)
    return conn


def _seed_labeled(conn, chunk_id, position, technique, video_id="v1"):
    conn.execute(
        "INSERT OR IGNORE INTO chunks VALUES (?,?,0,60,'t',NULL,'top','nogi')",
        (chunk_id, video_id),
    )
    conn.execute(
        "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?,?,?)",
        (chunk_id, "position", position),
    )
    conn.execute(
        "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?,?,?)",
        (chunk_id, "technique", technique),
    )
    conn.commit()


class TestSyntheticQueryGeneration:
    def test_generates_query_for_pair(self):
        from eval.run_eval import generate_synthetic_queries
        conn = _make_eval_db()
        for i in range(4):
            _seed_labeled(conn, f"c{i}", "mount", "armbar")
        queries = generate_synthetic_queries(conn, n=10, min_relevant=3)
        assert len(queries) == 1
        assert queries[0]["query"] == "armbar from mount"
        assert len(queries[0]["relevant_chunk_ids"]) == 4

    def test_min_relevant_filters_pairs(self):
        from eval.run_eval import generate_synthetic_queries
        conn = _make_eval_db()
        # Only 2 chunks — below min_relevant=3
        for i in range(2):
            _seed_labeled(conn, f"c{i}", "mount", "armbar")
        queries = generate_synthetic_queries(conn, n=10, min_relevant=3)
        assert len(queries) == 0

    def test_multiple_pairs_returned(self):
        from eval.run_eval import generate_synthetic_queries
        conn = _make_eval_db()
        for i in range(3):
            _seed_labeled(conn, f"m{i}", "mount", "armbar")
        for i in range(3):
            _seed_labeled(conn, f"s{i}", "side_control", "kimura")
        queries = generate_synthetic_queries(conn, n=10, min_relevant=3)
        assert len(queries) == 2
        texts = {q["query"] for q in queries}
        assert "armbar from mount" in texts
        assert "kimura from side control" in texts

    def test_relevant_ids_match_labeled_chunks(self):
        from eval.run_eval import generate_synthetic_queries
        conn = _make_eval_db()
        chunk_ids = [f"c{i}" for i in range(5)]
        for cid in chunk_ids:
            _seed_labeled(conn, cid, "guard", "triangle")
        queries = generate_synthetic_queries(conn, n=10, min_relevant=3)
        assert len(queries) == 1
        assert set(queries[0]["relevant_chunk_ids"]) == set(chunk_ids)


# run_evaluation integration (with mock engine)
class TestRunEvaluation:
    def _mock_engine(self, hit_ids: list[str]):
        """Engine that always returns hit_ids in order as SearchResult-like objects."""
        results = []
        for cid in hit_ids:
            mock = MagicMock()
            mock.chunk.id = cid
            results.append(mock)
        engine = MagicMock()
        engine.search.return_value = results
        return engine

    def test_perfect_retrieval_gives_1(self):
        from eval.run_eval import run_evaluation
        queries = [{"query": "armbar", "relevant_chunk_ids": ["a", "b"]}]
        engine = self._mock_engine(["a", "b", "c"])
        results = run_evaluation(queries, engine, k=2)
        assert results["aggregate"]["P@2"] == pytest.approx(1.0)
        assert results["aggregate"]["nDCG@2"] == pytest.approx(1.0)
        assert results["aggregate"]["MRR"] == pytest.approx(1.0)

    def test_miss_gives_zero(self):
        from eval.run_eval import run_evaluation
        queries = [{"query": "armbar", "relevant_chunk_ids": ["z"]}]
        engine = self._mock_engine(["a", "b", "c"])
        results = run_evaluation(queries, engine, k=3)
        assert results["aggregate"]["P@3"] == 0.0
        assert results["aggregate"]["MRR"] == 0.0

    def test_latency_populated(self):
        from eval.run_eval import run_evaluation
        queries = [{"query": "q", "relevant_chunk_ids": ["a"]}]
        engine = self._mock_engine(["a"])
        results = run_evaluation(queries, engine, k=1)
        assert results["latency_ms"]["mean"] >= 0.0

    def test_n_queries_correct(self):
        from eval.run_eval import run_evaluation
        queries = [
            {"query": f"q{i}", "relevant_chunk_ids": ["a"]}
            for i in range(5)
        ]
        engine = self._mock_engine(["a", "b"])
        results = run_evaluation(queries, engine, k=1)
        assert results["n_queries"] == 5
