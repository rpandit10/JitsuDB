"""Shared FastAPI dependencies — database connections and singleton services."""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING

from loguru import logger

from jitsudb.db import get_connection
from jitsudb.pipeline.orchestrator import PipelineOrchestrator

if TYPE_CHECKING:
    from jitsudb.retrieval.conformal import ConformalReranker
    from jitsudb.search.engine import SearchEngine

# Shared pipeline state written by the background thread, read by SSE
_pipeline_progress: dict = {}
_pipeline_lock = threading.Lock()

# Pipeline orchestrator singleton (created when processing starts)
_orchestrator: PipelineOrchestrator | None = None
_orchestrator_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None
_pause_event: threading.Event | None = None

# Search engine singleton (initialised on first search request)
_search_engine: "SearchEngine | None" = None

# Conformal reranker reference (set alongside _search_engine when enabled)
_conformal_reranker: "ConformalReranker | None" = None


def get_db() -> sqlite3.Connection:
    """FastAPI dependency that returns the thread-local SQLite connection."""
    return get_connection()


def get_search_engine() -> "SearchEngine":
    """FastAPI dependency that returns the shared SearchEngine singleton.

    Initialises the bi-encoder, ChromaDB collection, and (if enabled) the
    cross-encoder re-ranker on first call.
    """
    global _search_engine, _conformal_reranker
    if _search_engine is None:
        from jitsudb.config import get_config
        from jitsudb.pipeline.embedder import get_chroma_collection, load_embedding_model
        from jitsudb.search.engine import SearchEngine

        cfg = get_config()
        chroma_dir = cfg.paths.data_dir / "chromadb"

        model = load_embedding_model(
            model_name=cfg.embedding.model_name,
            device=cfg.embedding.device,
            model_path=cfg.embedding.encoder_path,
            trust_remote_code=cfg.embedding.trust_remote_code,
        )
        collection = get_chroma_collection(chroma_dir, cfg.embedding.collection_name)

        reranker = None
        if cfg.reranker.enabled:
            from jitsudb.search.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker(
                model_name=cfg.reranker.model_name,
                device=cfg.reranker.device,
            )
            if cfg.reranker.conformal_coverage is not None:
                from jitsudb.retrieval.conformal import (
                    ConformalReranker,
                    build_calibration_pairs,
                    build_calibration_pairs_from_clicks,
                )
                conformal = ConformalReranker(reranker, target_coverage=cfg.reranker.conformal_coverage)
                # Try golden QA file first, fall back to click data
                pairs = build_calibration_pairs()
                if not pairs:
                    pairs = build_calibration_pairs_from_clicks(get_connection())
                if pairs:
                    conformal.calibrate(pairs)
                _conformal_reranker = conformal
                reranker = conformal

        clip_model = None
        visual_collection = None
        if cfg.visual_embedding.enabled:
            from jitsudb.pipeline.visual_embedder import get_visual_collection, load_clip_model
            clip_model = load_clip_model(cfg.visual_embedding.model_name, cfg.visual_embedding.device)
            visual_collection = get_visual_collection(chroma_dir)

        colbert_index = None
        if cfg.search.colbert_enabled:
            from jitsudb.retrieval.colbert_index import ColBERTIndex
            colbert_dir = cfg.paths.data_dir / "colbert"
            colbert_index = ColBERTIndex(colbert_dir)
            logger.info(f"ColBERT index loaded: {len(colbert_index)} chunks indexed")

        _search_engine = SearchEngine(
            embedding_model=model,
            collection=collection,
            top_k=cfg.search.default_top_k,
            semantic_weight=cfg.search.semantic_weight,
            reranker=reranker,
            clip_model=clip_model,
            visual_collection=visual_collection,
            visual_weight=cfg.visual_embedding.visual_weight,
            beam_width=cfg.search.beam_width,
            max_expand=cfg.search.max_expand,
            reranker_top_k=cfg.search.reranker_top_k,
            candidate_k=cfg.search.candidate_k,
            hyde_enabled=cfg.search.hyde_enabled,
            hyde_n_hypotheticals=cfg.search.hyde_n_hypotheticals,
            query_prefix=cfg.embedding.query_prefix,
            colbert_index=colbert_index,
            colbert_weight=cfg.search.colbert_weight,
        )
    return _search_engine


def get_conformal_reranker() -> "ConformalReranker | None":
    """Return the active ConformalReranker, or None if conformal is disabled."""
    return _conformal_reranker


def get_pipeline_progress() -> dict:
    """Return a copy of the current pipeline progress state."""
    with _pipeline_lock:
        return dict(_pipeline_progress)


def get_orchestrator() -> PipelineOrchestrator | None:
    """Return the active orchestrator, or None if the pipeline is not running."""
    return _orchestrator


def is_pipeline_running() -> bool:
    """Return True if the pipeline thread is alive."""
    return _orchestrator_thread is not None and _orchestrator_thread.is_alive()


def is_pipeline_paused() -> bool:
    """Return True if the pipeline is paused."""
    return _pause_event is not None and _pause_event.is_set()


def start_pipeline() -> None:
    """Start the pipeline in a background thread.

    No-op if the pipeline is already running.
    """
    global _orchestrator, _orchestrator_thread, _stop_event, _pause_event
    if is_pipeline_running():
        return
    _stop_event = threading.Event()
    _pause_event = threading.Event()
    with _pipeline_lock:
        _pipeline_progress.clear()
    _orchestrator = PipelineOrchestrator(
        dry_run=False,
        progress_state=_pipeline_progress,
        stop_event=_stop_event,
        pause_event=_pause_event,
    )
    _orchestrator_thread = threading.Thread(
        target=_orchestrator.run,
        daemon=True,
        name="jitsudb-pipeline",
    )
    _orchestrator_thread.start()


def pause_pipeline() -> None:
    """Request the pipeline to pause after the current stage completes."""
    if not is_pipeline_running():
        raise RuntimeError("Pipeline is not running")
    if _pause_event is None:
        raise RuntimeError("Pipeline has no pause event")
    _pause_event.set()


def resume_pipeline() -> None:
    """Resume a paused pipeline."""
    if not is_pipeline_paused():
        raise RuntimeError("Pipeline is not paused")
    if _pause_event is None:
        raise RuntimeError("Pipeline has no pause event")
    _pause_event.clear()


def cancel_pipeline() -> None:
    """Stop the pipeline after the current stage completes."""
    if not is_pipeline_running():
        raise RuntimeError("Pipeline is not running")
    if _stop_event is None:
        raise RuntimeError("Pipeline has no stop event")
    _stop_event.set()
