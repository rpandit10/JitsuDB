"""Pydantic models for all data that crosses module boundaries or gets persisted."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Video(BaseModel):
    """A single instructional video in the library."""

    id: str = Field(default_factory=_new_uuid)
    title: str
    instructor: str
    file_path: Path
    duration_seconds: float
    gi_nogi: Literal["gi", "nogi", "both"]
    primary_category: str
    status: Literal["queued", "processing", "indexed", "failed"] = "queued"
    added_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime | None = None


class WordTimestamp(BaseModel):
    """A single word with its start/end timestamps from Whisper output."""

    word: str
    start: float
    end: float


class RawChunk(BaseModel):
    """Pre-classification transcript chunk with timing but no taxonomy labels yet."""

    start_time: float
    end_time: float
    words: list[WordTimestamp]
    transcript_text: str


class Chunk(BaseModel):
    """A classified, timestamped transcript segment ready for embedding and search."""

    id: str = Field(default_factory=_new_uuid)
    video_id: str
    start_time: float
    end_time: float
    transcript_text: str
    thumbnail_path: Path | None = None
    position: list[str] = Field(default_factory=list)
    context: Literal["top", "bottom", "neutral", "standing"] | None = None
    intent: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    gi_nogi: Literal["gi", "nogi", "both"] = "both"


class SearchResult(BaseModel):
    """A ranked search result pairing a chunk with its parent video."""

    chunk: Chunk
    video: Video
    relevance_score: float
    snippet: str
    detected_filters: dict[str, list[str]] = Field(default_factory=dict)
    visual_score: float = 0.0
    retrieved_by: Literal["text", "visual", "both"] = "text"


class QueueItem(BaseModel):
    """Processing queue entry for a single video."""

    video_id: str
    priority: int = 0
    status: Literal["pending", "processing", "done", "failed"] = "pending"
    error_message: str | None = None
    stage_completed: str | None = None


class PipelineProgress(BaseModel):
    """Real-time pipeline execution state, broadcast to the frontend via SSE."""

    video_id: str | None = None
    video_title: str | None = None
    stage: str | None = None
    stage_progress: float = 0.0
    overall_progress: float = 0.0
    is_running: bool = False
    is_paused: bool = False
    is_idle: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0
    video_progress: float = 0.0
    video_eta_seconds: float | None = None
    queue_eta_seconds: float | None = None
