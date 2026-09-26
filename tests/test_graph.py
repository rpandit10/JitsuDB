"""Tests for jitsudb.db.graph — edge construction and neighbour traversal."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from jitsudb.db.graph import (
    _cosine_similarity,
    build_edges_for_video,
    get_contiguous_window,
    get_contiguous_window_cte,
    get_neighbours,
)
from jitsudb.models import Chunk


class _MockCollection:
    """Minimal ChromaDB collection stub that returns pre-baked embeddings."""

    def __init__(self, embeddings_by_id: dict[str, list[float]]):
        self._embs = embeddings_by_id

    def get(self, ids: list[str], include: list[str]) -> dict:
        found = [i for i in ids if i in self._embs]
        return {
            "ids": found,
            "embeddings": [self._embs[i] for i in found],
        }


def _make_chunks(video_id: str, n: int = 10) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{i:03d}",
            video_id=video_id,
            start_time=float(i * 45),
            end_time=float(i * 45 + 44),
            transcript_text=f"Chunk {i} text.",
            gi_nogi="nogi",
        )
        for i in range(n)
    ]


def _make_collection(chunks: list[Chunk], seed: int = 42) -> _MockCollection:
    rng = np.random.default_rng(seed=seed)
    return _MockCollection(
        {c.id: rng.random(384).astype(np.float32).tolist() for c in chunks}
    )


def _insert_video(conn: sqlite3.Connection, video_id: str) -> None:
    conn.execute(
        """
        INSERT INTO videos (id, title, instructor, file_path, duration_seconds,
                            gi_nogi, primary_category, status, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (video_id, "Test Video", "Tester", "/fake/v.mp4",
         450.0, "nogi", "Passing", "indexed", "2024-01-01T00:00:00"),
    )


def _insert_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    for c in chunks:
        conn.execute(
            """
            INSERT INTO chunks (id, video_id, start_time, end_time,
                                transcript_text, context, gi_nogi)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (c.id, c.video_id, c.start_time, c.end_time,
             c.transcript_text, "neutral", c.gi_nogi),
        )
    conn.commit()


# Fixtures

VIDEO_ID = "vid-graph-test"


@pytest.fixture
def graph_db(tmp_db):
    """tmp_db pre-populated with one video and 10 chunks."""
    chunks = _make_chunks(VIDEO_ID)
    _insert_video(tmp_db, VIDEO_ID)
    _insert_chunks(tmp_db, chunks)
    return tmp_db, chunks


def test_build_edges_count(graph_db):
    """10 ordered chunks → 9 directed edges."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    n = build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)
    assert n == 9
    row = conn.execute("SELECT COUNT(*) FROM chunk_edges WHERE video_id = ?", (VIDEO_ID,)).fetchone()
    assert row[0] == 9


def test_build_edges_continuity_scores_in_range(graph_db):
    """Every continuity_score must be a valid cosine similarity in [-1, 1]."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)
    rows = conn.execute(
        "SELECT continuity_score FROM chunk_edges WHERE video_id = ?", (VIDEO_ID,)
    ).fetchall()
    assert len(rows) == 9
    for row in rows:
        score = row[0]
        assert -1.0 <= score <= 1.0, f"continuity_score {score} out of range"


def test_build_edges_temporal_gap(graph_db):
    """temporal_gap_sec should equal dst.start_time - src.end_time for each pair."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)
    rows = conn.execute(
        """
        SELECT ce.temporal_gap_sec, c_src.end_time, c_dst.start_time
        FROM chunk_edges ce
        JOIN chunks c_src ON c_src.id = ce.src_chunk_id
        JOIN chunks c_dst ON c_dst.id = ce.dst_chunk_id
        WHERE ce.video_id = ?
        """,
        (VIDEO_ID,),
    ).fetchall()
    for row in rows:
        expected_gap = max(0.0, row[2] - row[1])
        assert abs(row[0] - expected_gap) < 1e-6


def test_build_edges_idempotent(graph_db):
    """Re-running build_edges_for_video should leave exactly 9 edges, not 18."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)
    row = conn.execute("SELECT COUNT(*) FROM chunk_edges WHERE video_id = ?", (VIDEO_ID,)).fetchone()
    assert row[0] == 9


def test_build_edges_fewer_than_two_chunks(tmp_db):
    """A video with one chunk produces 0 edges and returns 0."""
    video_id = "vid-single"
    _insert_video(tmp_db, video_id)
    chunk = _make_chunks(video_id, n=1)
    _insert_chunks(tmp_db, chunk)
    col = _make_collection(chunk)
    n = build_edges_for_video(video_id, chunk, col, conn=tmp_db)
    assert n == 0
    row = tmp_db.execute("SELECT COUNT(*) FROM chunk_edges WHERE video_id = ?", (video_id,)).fetchone()
    assert row[0] == 0


def test_build_edges_missing_embeddings_skipped(graph_db):
    """Edges for pairs where one embedding is missing are silently skipped."""
    conn, chunks = graph_db
    # Only give embeddings for even-indexed chunks → 4 adjacent pairs without both
    rng = np.random.default_rng(seed=7)
    partial_embs = {
        c.id: rng.random(384).astype(np.float32).tolist()
        for i, c in enumerate(chunks) if i % 2 == 0
    }
    col = _MockCollection(partial_embs)
    n = build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)
    # Even chunks: 0,2,4,6,8 → adjacent pairs (0,2),(2,4),(4,6),(6,8) → 4 edges
    # (chunk-000→chunk-002 skips chunk-001 which lacks embedding)
    # But edges are between CONSECUTIVE ordered chunks, not just even ones.
    # Ordered: 0,1,2,...9. Pairs: (0,1),(1,2),...
    # Even chunks present: 0,2,4,6,8. Adjacent even pairs: (0,2) not adjacent in order.
    # So actually edges between (i, i+1) where both i and i+1 have embeddings.
    # Even: {0,2,4,6,8}. Pairs with both: none (no two consecutive even numbers differ by 1).
    # So 0 edges.
    assert n == 0


def test_get_neighbours_next_hop1(graph_db):
    """From chunk-005, direction=next, max_hops=1 → only chunk-006."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    result = get_neighbours("chunk-005", direction="next", max_hops=1, conn=conn)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "chunk-006"
    assert result[0]["hop"] == 1
    assert result[0]["direction"] == "next"


def test_get_neighbours_prev_hop1(graph_db):
    """From chunk-005, direction=prev, max_hops=1 → only chunk-004."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    result = get_neighbours("chunk-005", direction="prev", max_hops=1, conn=conn)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "chunk-004"
    assert result[0]["hop"] == 1
    assert result[0]["direction"] == "prev"


def test_get_neighbours_both_hop2(graph_db):
    """From chunk-005, direction=both, max_hops=2 → chunks 003, 004, 006, 007."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    result = get_neighbours("chunk-005", direction="both", max_hops=2, conn=conn)
    ids = {r["chunk_id"] for r in result}
    assert ids == {"chunk-003", "chunk-004", "chunk-006", "chunk-007"}


def test_get_neighbours_boundary_first_chunk(graph_db):
    """chunk-000 has no predecessor — direction=prev must return []."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    result = get_neighbours("chunk-000", direction="prev", max_hops=2, conn=conn)
    assert result == []


def test_get_neighbours_boundary_last_chunk(graph_db):
    """chunk-009 has no successor — direction=next must return []."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    result = get_neighbours("chunk-009", direction="next", max_hops=2, conn=conn)
    assert result == []


def test_get_neighbours_invalid_direction(graph_db):
    conn, chunks = graph_db
    with pytest.raises(ValueError, match="direction must be"):
        get_neighbours("chunk-005", direction="sideways", conn=conn)


def test_get_contiguous_window_single_anchor(graph_db):
    """anchor=[chunk-005], beam=2 → [chunk-003, chunk-004, chunk-005, chunk-006, chunk-007]."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    window = get_contiguous_window(["chunk-005"], beam_width=2, conn=conn)
    assert window == ["chunk-003", "chunk-004", "chunk-005", "chunk-006", "chunk-007"]


def test_get_contiguous_window_temporal_order(graph_db):
    """Window must be sorted by start_time regardless of anchor/expansion order."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    window = get_contiguous_window(["chunk-007", "chunk-002"], beam_width=1, conn=conn)
    # chunk-002 expands to 001,003; chunk-007 expands to 006,008 → all unique ids sorted
    expected = ["chunk-001", "chunk-002", "chunk-003", "chunk-006", "chunk-007", "chunk-008"]
    assert window == expected


def test_get_contiguous_window_beam_zero(graph_db):
    """beam_width=0 returns the anchors themselves in temporal order, no expansion."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    window = get_contiguous_window(["chunk-007", "chunk-003"], beam_width=0, conn=conn)
    assert window == ["chunk-003", "chunk-007"]


def test_get_contiguous_window_empty_anchors(graph_db):
    conn, chunks = graph_db
    assert get_contiguous_window([], beam_width=2, conn=conn) == []


# Cascade delete regression
def test_cascade_delete_removes_edges(graph_db):
    """Deleting chunk-005 must remove all edges where it is src or dst."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("DELETE FROM chunks WHERE id = 'chunk-005'")
    conn.commit()

    rows = conn.execute(
        """
        SELECT COUNT(*) FROM chunk_edges
        WHERE src_chunk_id = 'chunk-005' OR dst_chunk_id = 'chunk-005'
        """
    ).fetchone()
    assert rows[0] == 0

    # The remaining 9-1=8 chunks should have (8-1)=7 edges but the two edges
    # touching chunk-005 were (chunk-004→chunk-005) and (chunk-005→chunk-006),
    # so 9 - 2 = 7 edges remain.
    remaining = conn.execute(
        "SELECT COUNT(*) FROM chunk_edges WHERE video_id = ?", (VIDEO_ID,)
    ).fetchone()
    assert remaining[0] == 7


# _cosine_similarity unit tests
def test_cosine_similarity_identical():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero_vector():
    z = np.array([0.0, 0.0], dtype=np.float32)
    v = np.array([1.0, 0.0], dtype=np.float32)
    assert _cosine_similarity(z, v) == 0.0
    assert _cosine_similarity(v, z) == 0.0


# get_contiguous_window_cte  (must match BFS output)
def test_cte_matches_bfs_single_anchor(graph_db):
    """CTE result for a single anchor must equal the BFS-based window."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    bfs = get_contiguous_window(["chunk-005"], beam_width=2, conn=conn)
    cte = get_contiguous_window_cte(["chunk-005"], beam_width=2, conn=conn)
    assert cte == bfs


def test_cte_matches_bfs_multi_anchor(graph_db):
    """CTE result for multiple anchors must equal the BFS-based window."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    bfs = get_contiguous_window(["chunk-002", "chunk-007"], beam_width=1, conn=conn)
    cte = get_contiguous_window_cte(["chunk-002", "chunk-007"], beam_width=1, conn=conn)
    assert cte == bfs


def test_cte_beam_zero_returns_sorted_anchors(graph_db):
    """CTE with beam_width=0 returns only the anchor IDs sorted by start_time."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    result = get_contiguous_window_cte(["chunk-007", "chunk-003"], beam_width=0, conn=conn)
    assert result == ["chunk-003", "chunk-007"]


def test_cte_empty_anchors_returns_empty(graph_db):
    conn, chunks = graph_db
    assert get_contiguous_window_cte([], beam_width=2, conn=conn) == []


def test_cte_single_query_execution(graph_db):
    """CTE expansion must execute exactly 1 SQL statement regardless of anchor count."""
    conn, chunks = graph_db
    col = _make_collection(chunks)
    build_edges_for_video(VIDEO_ID, chunks, col, conn=conn)

    query_count = 0

    def _count(stmt):
        nonlocal query_count
        query_count += 1

    conn.set_trace_callback(_count)
    get_contiguous_window_cte(["chunk-001", "chunk-004", "chunk-007"], beam_width=2, conn=conn)
    conn.set_trace_callback(None)

    assert query_count == 1


# Beam-walk context expansion (engine integration)
def test_beam_expansion_adds_context_chunks(tmp_db):
    """SearchEngine with beam_width=1 appends temporal neighbours to results."""
    from unittest.mock import MagicMock

    from jitsudb.db.graph import build_edges_for_video
    from jitsudb.search.engine import SearchEngine

    video_id = "vid-beam-expand"
    _insert_video(tmp_db, video_id)
    chunks = _make_chunks(video_id, n=5)
    _insert_chunks(tmp_db, chunks)
    col = _make_collection(chunks)
    build_edges_for_video(video_id, chunks, col, conn=tmp_db)

    # Mock ChromaDB returns only chunk-002 as the primary semantic hit.
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 384

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk-002"]],
        "distances": [[0.1]],
        "metadatas": [[{}]],
        "documents": [["Chunk 2 text."]],
    }

    engine = SearchEngine(
        embedding_model=mock_model,
        collection=mock_collection,
        top_k=10,
        beam_width=1,
        max_expand=4,
    )
    results = engine.search("basic drill test")
    result_ids = {r.chunk.id for r in results}

    assert "chunk-002" in result_ids
    # beam_width=1 → neighbours chunk-001 and chunk-003 should be added
    assert "chunk-001" in result_ids
    assert "chunk-003" in result_ids


def test_beam_expansion_disabled(tmp_db):
    """beam_width=0 must not add any extra chunks beyond the semantic hit."""
    from unittest.mock import MagicMock

    from jitsudb.db.graph import build_edges_for_video
    from jitsudb.search.engine import SearchEngine

    video_id = "vid-beam-zero"
    _insert_video(tmp_db, video_id)
    chunks = _make_chunks(video_id, n=5)
    _insert_chunks(tmp_db, chunks)
    col = _make_collection(chunks)
    build_edges_for_video(video_id, chunks, col, conn=tmp_db)

    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 384

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk-002"]],
        "distances": [[0.1]],
        "metadatas": [[{}]],
        "documents": [["Chunk 2 text."]],
    }

    engine = SearchEngine(
        embedding_model=mock_model,
        collection=mock_collection,
        top_k=10,
        beam_width=0,
        max_expand=4,
    )
    results = engine.search("basic drill")
    result_ids = {r.chunk.id for r in results}

    assert "chunk-002" in result_ids
    assert "chunk-001" not in result_ids
    assert "chunk-003" not in result_ids


def test_beam_expansion_capped_by_max_expand(tmp_db):
    """max_expand=1 limits newly added context chunks to at most 1."""
    from unittest.mock import MagicMock

    from jitsudb.db.graph import build_edges_for_video
    from jitsudb.search.engine import SearchEngine

    video_id = "vid-beam-cap"
    _insert_video(tmp_db, video_id)
    chunks = _make_chunks(video_id, n=5)
    _insert_chunks(tmp_db, chunks)
    col = _make_collection(chunks)
    build_edges_for_video(video_id, chunks, col, conn=tmp_db)

    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 384

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk-002"]],
        "distances": [[0.1]],
        "metadatas": [[{}]],
        "documents": [["Chunk 2 text."]],
    }

    engine = SearchEngine(
        embedding_model=mock_model,
        collection=mock_collection,
        top_k=10,
        beam_width=2,
        max_expand=1,
    )
    results = engine.search("basic technique")
    # Primary result is chunk-002. With beam_width=2, expansion would normally
    # produce chunk-000, chunk-001, chunk-003, chunk-004 — but max_expand=1 caps it.
    new_chunks = [r for r in results if r.chunk.id != "chunk-002"]
    assert len(new_chunks) <= 1


# Multi-hop temporal ordering
def test_multihop_query_sorts_by_start_time(tmp_db):
    """A temporal-connective query groups and sorts results by start_time."""
    from unittest.mock import MagicMock

    from jitsudb.db.graph import build_edges_for_video
    from jitsudb.search.engine import SearchEngine

    video_id = "vid-multihop"
    _insert_video(tmp_db, video_id)
    chunks = _make_chunks(video_id, n=5)
    _insert_chunks(tmp_db, chunks)
    col = _make_collection(chunks)
    build_edges_for_video(video_id, chunks, col, conn=tmp_db)

    # Return chunks 004, 001, 002 in that (non-temporal) order
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1] * 384

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk-004", "chunk-001", "chunk-002"]],
        "distances": [[0.05, 0.2, 0.3]],
        "metadatas": [[{}, {}, {}]],
        "documents": [["d4", "d1", "d2"]],
    }

    engine = SearchEngine(
        embedding_model=mock_model,
        collection=mock_collection,
        top_k=10,
        beam_width=0,  # no expansion, test ordering only
        max_expand=0,
    )
    # "before" triggers is_multihop=True
    results = engine.search("what comes before the final technique")
    result_ids = [r.chunk.id for r in results if r.chunk.id in {"chunk-001", "chunk-002", "chunk-004"}]
    # Within the single video group, chunks must appear in temporal (start_time) order
    assert result_ids == sorted(result_ids, key=lambda cid: int(cid.split("-")[1]))
