"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jitsudb.models import Chunk, RawChunk, Video, WordTimestamp


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a path to a temporary SQLite database file.

    Must match the filename that init_db() derives from DB_FILENAME so that
    the fixture connection and the thread-local connection point to the same file.
    """
    from jitsudb.constants import DB_FILENAME
    return tmp_path / DB_FILENAME


@pytest.fixture
def tmp_db(tmp_db_path: Path) -> sqlite3.Connection:
    """Open a temporary SQLite connection with the full JitsuDB schema applied."""
    import jitsudb.config as cfg_module
    import jitsudb.db as db_module
    from jitsudb.config import AppConfig, PathsConfig
    from jitsudb.db import get_connection, init_db

    # Close any stale thread-local connection from a previous test so that
    # get_connection() opens a fresh connection pointing at the new temp path.
    if hasattr(db_module._local, "conn") and db_module._local.conn is not None:
        try:
            db_module._local.conn.close()
        except Exception:
            pass
        db_module._local.conn = None

    cfg_module._config = AppConfig(paths=PathsConfig(data_dir=tmp_db_path.parent))
    init_db()

    # Return the same connection that init_db() used so inserts are immediately
    # visible without cross-connection synchronisation issues.
    conn = get_connection()
    yield conn

    # Teardown: close and clear so the next test gets a clean slate.
    try:
        conn.close()
    except Exception:
        pass
    db_module._local.conn = None
    cfg_module._config = None


@pytest.fixture
def sample_words() -> list[WordTimestamp]:
    """A short list of WordTimestamp objects representing a simple transcript."""
    return [
        WordTimestamp(word="So", start=0.0, end=0.2),
        WordTimestamp(word="from", start=0.2, end=0.4),
        WordTimestamp(word="side", start=0.4, end=0.7),
        WordTimestamp(word="control", start=0.7, end=1.1),
        WordTimestamp(word="you", start=1.1, end=1.3),
        WordTimestamp(word="need", start=1.3, end=1.5),
        WordTimestamp(word="to", start=1.5, end=1.6),
        WordTimestamp(word="escape.", start=1.6, end=2.0),
        WordTimestamp(word="Use", start=2.1, end=2.3),
        WordTimestamp(word="the", start=2.3, end=2.4),
        WordTimestamp(word="hip", start=2.4, end=2.6),
        WordTimestamp(word="escape", start=2.6, end=3.0),
        WordTimestamp(word="first.", start=3.0, end=3.3),
    ]


@pytest.fixture
def sample_video() -> Video:
    """A minimal Video fixture for use in pipeline and classifier tests."""
    return Video(
        title="Side Control Escapes",
        instructor="Test Instructor",
        file_path=Path("/fake/video.mp4"),
        duration_seconds=300.0,
        gi_nogi="nogi",
        primary_category="Escapes-Defense",
    )


@pytest.fixture
def sample_raw_chunk(sample_words: list[WordTimestamp]) -> RawChunk:
    """A RawChunk built from sample_words."""
    text = " ".join(w.word for w in sample_words)
    return RawChunk(
        start_time=sample_words[0].start,
        end_time=sample_words[-1].end,
        words=sample_words,
        transcript_text=text,
    )
