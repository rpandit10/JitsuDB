"""POST /api/process — pipeline start, pause, and cancel controls."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from jitsudb.api import dependencies
from jitsudb.models import PipelineProgress

router = APIRouter(tags=["process"])


def _current_progress() -> PipelineProgress:
    raw = dependencies.get_pipeline_progress()
    if raw:
        return PipelineProgress(**raw)
    return PipelineProgress(
        is_running=dependencies.is_pipeline_running(),
        is_paused=dependencies.is_pipeline_paused(),
    )


@router.post("/process/start", response_model=PipelineProgress)
async def start_pipeline() -> PipelineProgress:
    """Start the processing pipeline in a background thread.

    If the pipeline is already running, returns the current progress
    without starting a second instance.
    """
    dependencies.start_pipeline()
    return _current_progress()


@router.post("/process/pause", response_model=PipelineProgress)
async def pause_pipeline() -> PipelineProgress:
    """Request the pipeline to pause after the current video finishes."""
    try:
        dependencies.pause_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _current_progress()


@router.post("/process/resume", response_model=PipelineProgress)
async def resume_pipeline() -> PipelineProgress:
    """Resume a paused pipeline."""
    try:
        dependencies.resume_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _current_progress()


@router.post("/process/cancel", response_model=PipelineProgress)
async def cancel_pipeline() -> PipelineProgress:
    """Stop the pipeline after the current stage completes."""
    try:
        dependencies.cancel_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _current_progress()


@router.post("/process/retry-failed", response_model=PipelineProgress)
async def retry_failed() -> PipelineProgress:
    """Reset all failed queue items back to pending so they are retried on next start."""
    from jitsudb.db import get_connection
    conn = get_connection()
    conn.execute(
        "UPDATE processing_queue SET status = 'pending', error_message = NULL WHERE status = 'failed'"
    )
    conn.execute("UPDATE videos SET status = 'queued' WHERE status = 'failed'")
    conn.commit()
    return _current_progress()
