"""SQLite connection management and schema initialization."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from loguru import logger

from jitsudb.config import get_config
from jitsudb.constants import DB_FILENAME, TAXONOMY_VERSION

_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it on first access.

    Uses WAL journal mode for safe concurrent reads from the FastAPI thread
    alongside writes from the background pipeline thread.
    """
    if not hasattr(_local, "conn") or _local.conn is None:
        cfg = get_config()
        db_path: Path = cfg.paths.data_dir / DB_FILENAME
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        logger.debug(f"Opened SQLite connection to {db_path}")
    return _local.conn


def close_connection() -> None:
    """Close the thread-local connection if open."""
    if hasattr(_local, "conn") and _local.conn is not None:
        _local.conn.close()
        _local.conn = None


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call multiple times.

    Tables created:
        videos            — one row per video file
        chunks            — one row per transcript chunk
        taxonomy_labels   — normalised position/intent/technique tags per chunk
        processing_queue  — pipeline queue with priority ordering
        word_timestamps   — word-level timing data from Whisper
        schema_version    — key/value store for versioning (e.g. taxonomy version)
    """
    conn = get_connection()
    logger.info("Initialising database schema")

    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS videos (
            id               TEXT PRIMARY KEY,
            title            TEXT NOT NULL,
            instructor       TEXT NOT NULL,
            file_path        TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            gi_nogi          TEXT NOT NULL CHECK(gi_nogi IN ('gi', 'nogi', 'both')),
            primary_category TEXT NOT NULL,
            status           TEXT NOT NULL DEFAULT 'queued'
                             CHECK(status IN ('queued', 'processing', 'indexed', 'failed')),
            added_at         TEXT NOT NULL,
            processed_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id               TEXT PRIMARY KEY,
            video_id         TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            start_time       REAL NOT NULL,
            end_time         REAL NOT NULL,
            transcript_text  TEXT NOT NULL,
            thumbnail_path   TEXT,
            context          TEXT CHECK(context IN ('top', 'bottom', 'neutral', 'standing')),
            gi_nogi          TEXT NOT NULL DEFAULT 'both'
                             CHECK(gi_nogi IN ('gi', 'nogi', 'both'))
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);

        CREATE TABLE IF NOT EXISTS taxonomy_labels (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id    TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            label_type  TEXT NOT NULL,
            label_value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_taxonomy_chunk      ON taxonomy_labels(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_taxonomy_type_value ON taxonomy_labels(label_type, label_value);

        CREATE TABLE IF NOT EXISTS processing_queue (
            video_id        TEXT PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
            priority        INTEGER NOT NULL DEFAULT 0,
            status          TEXT NOT NULL DEFAULT 'pending'
                            CHECK(status IN ('pending', 'processing', 'done', 'failed')),
            error_message   TEXT,
            stage_completed TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_queue_priority ON processing_queue(priority, status);

        CREATE TABLE IF NOT EXISTS word_timestamps (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            word     TEXT NOT NULL,
            start_sec REAL NOT NULL,
            end_sec   REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_word_timestamps_chunk ON word_timestamps(chunk_id);

        CREATE TABLE IF NOT EXISTS schema_version (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        INSERT OR IGNORE INTO schema_version(key, value)
            VALUES ('taxonomy_version', '{TAXONOMY_VERSION}');

        -- Temporal graph edges between adjacent chunks within the same video.
        -- continuity_score = cosine similarity between adjacent chunk embeddings.
        CREATE TABLE IF NOT EXISTS chunk_edges (
            src_chunk_id      TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            dst_chunk_id      TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            video_id          TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            temporal_gap_sec  REAL NOT NULL,
            continuity_score  REAL NOT NULL,
            PRIMARY KEY (src_chunk_id, dst_chunk_id)
        );

        CREATE INDEX IF NOT EXISTS idx_edges_src   ON chunk_edges(src_chunk_id);
        CREATE INDEX IF NOT EXISTS idx_edges_dst   ON chunk_edges(dst_chunk_id);
        CREATE INDEX IF NOT EXISTS idx_edges_video ON chunk_edges(video_id);

        -- Chunks that the classifier assigned low confidence (high entropy).
        CREATE TABLE IF NOT EXISTS uncertain_chunks (
            chunk_id      TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            entropy       REAL NOT NULL,
            top_label     TEXT NOT NULL,
            second_label  TEXT,
            reviewed      INTEGER NOT NULL DEFAULT 0
        );

        -- Per-retrieval-event log for implicit click-through feedback.
        CREATE TABLE IF NOT EXISTS retrieval_log (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id         TEXT NOT NULL,
            query_text         TEXT NOT NULL DEFAULT '',
            query_embedding    BLOB NOT NULL,
            chunk_id           TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            rank               INTEGER NOT NULL,
            clicked            INTEGER NOT NULL DEFAULT 0,
            watch_duration_sec REAL NOT NULL DEFAULT 0.0,
            latency_ms         REAL NOT NULL DEFAULT 0.0,
            logged_at          TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_retrieval_log_session ON retrieval_log(session_id);

        -- Concept drift events detected per taxonomy class.
        CREATE TABLE IF NOT EXISTS drift_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            taxonomy_class  TEXT NOT NULL,
            distance        REAL NOT NULL,
            baseline_at     TEXT NOT NULL,
            detected_at     TEXT NOT NULL
        );

        -- LLM-proposed taxonomy labels pending human review.
        CREATE TABLE IF NOT EXISTS proposed_labels (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            label_text      TEXT NOT NULL UNIQUE,
            cluster_summary TEXT NOT NULL,
            nli_verdict     TEXT NOT NULL,
            activated       INTEGER NOT NULL DEFAULT 0,
            proposed_at     TEXT NOT NULL
        );

        -- HDBSCAN cluster label proposals per chunk.
        CREATE TABLE IF NOT EXISTS chunk_label_proposals (
            chunk_id            TEXT PRIMARY KEY,
            cluster_id          INTEGER NOT NULL,
            proposed_positions  TEXT NOT NULL DEFAULT '[]',
            proposed_intents    TEXT NOT NULL DEFAULT '[]',
            proposed_techniques TEXT NOT NULL DEFAULT '[]',
            confidence          REAL NOT NULL DEFAULT 0.0,
            created_at          TEXT NOT NULL
        );

        -- YouTube video IDs seen by the recommendation scraper.
        -- Prevents re-surfacing confirmed or dismissed videos.
        CREATE TABLE IF NOT EXISTS youtube_seen (
            yt_video_id  TEXT PRIMARY KEY,
            action       TEXT NOT NULL CHECK(action IN ('confirmed', 'dismissed')),
            title        TEXT,
            channel      TEXT,
            seen_at      TEXT NOT NULL
        );
    """)

    conn.commit()

    # Idempotent column migrations for existing databases.
    _migrations = [
        "ALTER TABLE retrieval_log ADD COLUMN latency_ms REAL NOT NULL DEFAULT 0.0",
        "ALTER TABLE retrieval_log ADD COLUMN query_text TEXT NOT NULL DEFAULT ''",
        # youtube recommendation scraper seen-list
        """CREATE TABLE IF NOT EXISTS youtube_seen (
            yt_video_id  TEXT PRIMARY KEY,
            action       TEXT NOT NULL CHECK(action IN ('confirmed', 'dismissed')),
            title        TEXT,
            channel      TEXT,
            seen_at      TEXT NOT NULL
        )""",
    ]
    for stmt in _migrations:
        try:
            conn.execute(stmt)
        except Exception:
            pass  # already exists or column already added
    conn.commit()

    logger.info("Database schema ready")
