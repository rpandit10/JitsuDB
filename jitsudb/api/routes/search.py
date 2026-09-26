"""GET /api/search — natural language + structured filter search."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Query

from jitsudb.api.dependencies import get_db, get_search_engine
from jitsudb.models import SearchResult
from jitsudb.search.engine import SearchEngine

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[SearchResult])
async def search(
    q: str = Query(..., description="Natural language search query"),
    position: list[str] = Query(default=[], description="Position filter labels"),
    intent: list[str] = Query(default=[], description="Intent filter labels"),
    technique: list[str] = Query(default=[], description="Technique filter labels"),
    context: str | None = Query(default=None, description="top | bottom | neutral | standing"),
    gi_nogi: str | None = Query(default=None, description="gi | nogi | both"),
    instructor: str | None = Query(default=None, description="Instructor name filter"),
    session_id: str | None = Query(default=None, description="Client session UUID for click tracking"),
    engine: SearchEngine = Depends(get_search_engine),
    db: sqlite3.Connection = Depends(get_db),
) -> list[SearchResult]:
    """Search the indexed chunk corpus with optional structured filters."""
    filters: dict[str, list[str]] = {}
    if position:
        filters["positions"] = position
    if intent:
        filters["intents"] = intent
    if technique:
        filters["techniques"] = technique
    if context:
        filters["context"] = [context]
    if gi_nogi:
        filters["gi_nogi"] = [gi_nogi]
    if instructor:
        filters["instructor"] = [instructor]

    results = engine.search(q, filters)

    if session_id and results and engine._last_query_embedding is not None:
        from jitsudb.search.click_reranker import log_impression
        log_impression(
            session_id,
            engine._last_query_embedding,
            results,
            db,
            latency_ms=engine._last_total_ms or 0.0,
            query_text=q,
        )

    return results
