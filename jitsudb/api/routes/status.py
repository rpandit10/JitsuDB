"""GET /api/status — Server-Sent Events stream for pipeline progress."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from jitsudb.api.dependencies import get_pipeline_progress

router = APIRouter(tags=["status"])


@router.get("/status")
async def pipeline_status() -> StreamingResponse:
    """Stream pipeline progress updates as Server-Sent Events.

    The client should connect with ``EventSource('/api/status')``.
    Each event is a JSON-serialised PipelineProgress payload.

    The stream emits one event per second while the pipeline is running
    and continues emitting heartbeat events (is_running=False) when idle
    so the client connection stays alive.
    """
    async def event_generator():
        while True:
            progress = get_pipeline_progress()
            data = json.dumps(progress)
            yield f"data: {data}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
