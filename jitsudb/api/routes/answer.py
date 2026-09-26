"""GET /api/answer — streaming RAG answer synthesis over retrieved BJJ chunks."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from loguru import logger

from jitsudb.api.dependencies import get_search_engine
from jitsudb.search.engine import SearchEngine

router = APIRouter(tags=["answer"])


@router.get("/answer")
async def answer(
    q: str = Query(..., description="Natural language question about BJJ technique"),
    engine: SearchEngine = Depends(get_search_engine),
) -> StreamingResponse:
    """Stream a RAG-synthesized answer from the top search results.

    SSE event types:
    - ``{"type": "citations", "citations": [...]}`` — sent first, lists source chunks
    - ``{"type": "token", "content": "..."}`` — one per generated token
    - ``{"type": "done"}`` — generation complete
    - ``{"type": "error", "message": "..."}`` — on failure
    """
    from jitsudb.config import get_config
    from jitsudb.rag.synthesizer import RagSynthesizer

    cfg = get_config()

    async def event_stream() -> AsyncGenerator[str, None]:
        # Search runs synchronously; delegate to thread pool
        results = await asyncio.to_thread(engine.search, q)

        if not results:
            yield _sse({"type": "error", "message": "No results found for this query."})
            return

        if not cfg.rag.enabled:
            yield _sse({"type": "error", "message": "RAG is disabled in config.yaml."})
            return

        synthesizer = RagSynthesizer(
            ollama_url=cfg.rag.ollama_url,
            model=cfg.rag.model,
            max_context_chunks=cfg.rag.max_context_chunks,
            max_tokens=cfg.rag.max_tokens,
            temperature=cfg.rag.temperature,
        )

        available = await asyncio.to_thread(synthesizer.is_available)
        if not available:
            yield _sse({
                "type": "error",
                "message": (
                    f"Ollama is not running or model '{cfg.rag.model}' is not pulled. "
                    f"Run: ollama serve && ollama pull {cfg.rag.model}"
                ),
            })
            return

        # Emit citations before streaming so the UI can render source links immediately
        citations = [
            {
                "index": i + 1,
                "video_title": r.video.title,
                "start_time": r.chunk.start_time,
                "chunk_id": r.chunk.id,
                "relevance_score": r.relevance_score,
            }
            for i, r in enumerate(results[: cfg.rag.max_context_chunks])
        ]
        yield _sse({"type": "citations", "citations": citations})

        # Bridge the synchronous generator to the async SSE stream via a queue.
        # When faithfulness_check is enabled the generator yields dicts;
        # otherwise it yields plain token strings.
        item_queue: queue.Queue[dict | str | None] = queue.Queue()

        def produce() -> None:
            try:
                if cfg.rag.faithfulness_check:
                    for item in synthesizer.synthesize_stream_with_faithfulness(q, results):
                        item_queue.put(item)
                else:
                    for token in synthesizer.synthesize_stream(q, results):
                        item_queue.put({"type": "token", "content": token})
            except Exception as exc:
                logger.error(f"RAG synthesis error: {exc}")
                item_queue.put({"type": "token", "content": f"\n\n[Error: {exc}]"})
            finally:
                item_queue.put(None)  # sentinel

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()

        while True:
            item = await asyncio.to_thread(item_queue.get)
            if item is None:
                break
            yield _sse(item)

        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
