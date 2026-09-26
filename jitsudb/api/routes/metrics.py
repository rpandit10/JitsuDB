"""GET /api/metrics — search latency, VRAM state, and index statistics."""

from __future__ import annotations

import math
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends

from jitsudb.api.dependencies import get_db, get_search_engine
from jitsudb.search.engine import SearchEngine

router = APIRouter(tags=["metrics"])


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile. P95 of [1..100] == 95.0."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = min(max(0, math.ceil(n * p / 100) - 1), n - 1)
    return round(sorted_vals[idx], 1)


def _vram_stats() -> dict[str, Any]:
    # System-level stats via pynvml — covers ALL processes (whisper, Ollama, torch, etc.)
    result: dict[str, Any] = {}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        result = {
            "available":      True,
            "device":         name,
            "total_gb":       round(mem.total / 1024 ** 3, 2),
            "system_used_gb": round(mem.used  / 1024 ** 3, 2),
            "system_free_gb": round(mem.free  / 1024 ** 3, 2),
            "gpu_util_pct":   util.gpu,
            "mem_util_pct":   util.memory,
        }
    except Exception:
        pass

    if not result:
        return {"available": False}

    # PyTorch allocator stats — only what this process placed on the GPU via torch
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            result.update({
                "pytorch_allocated_gb":      round(torch.cuda.memory_allocated(dev)     / 1024 ** 3, 3),
                "pytorch_reserved_gb":       round(torch.cuda.memory_reserved(dev)      / 1024 ** 3, 3),
                "pytorch_peak_allocated_gb": round(torch.cuda.max_memory_allocated(dev) / 1024 ** 3, 3),
            })
    except (ImportError, RuntimeError):
        pass

    return result


def _conformal_stats() -> dict[str, Any]:
    from jitsudb.api.dependencies import get_conformal_reranker
    conformal = get_conformal_reranker()
    if conformal is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "target_coverage": conformal.target_coverage,
        "q_hat": round(conformal._q_hat, 6) if conformal._q_hat is not None else None,
        "calibrated": conformal._q_hat is not None,
    }


@router.get("/metrics")
def get_metrics(
    db: sqlite3.Connection = Depends(get_db),
    engine: SearchEngine = Depends(get_search_engine),
) -> dict[str, Any]:
    """Return aggregate search latency (P50/P95/P99), VRAM state, and index counts.

    Latency is derived from ``retrieval_log.latency_ms`` stored on rank-0 rows
    by the search route. Rows with ``latency_ms = 0`` (pre-migration impressions
    or impressions logged without a session_id) are excluded.
    """
    # Latency percentiles from retrieval_log rank-0 rows
    rows = db.execute(
        "SELECT latency_ms FROM retrieval_log WHERE rank = 0 AND latency_ms > 0"
    ).fetchall()
    latencies = sorted(r["latency_ms"] for r in rows)
    n = len(latencies)
    latency: dict[str, Any] = {
        "n_queries": n,
        "p50_ms":  _percentile(latencies, 50),
        "p95_ms":  _percentile(latencies, 95),
        "p99_ms":  _percentile(latencies, 99),
        "mean_ms": round(sum(latencies) / n, 1) if n else 0.0,
        "min_ms":  round(latencies[0], 1) if latencies else 0.0,
        "max_ms":  round(latencies[-1], 1) if latencies else 0.0,
    }

    # Last search stage breakdown
    last_search: dict[str, Any] = {}
    if engine._last_timing is not None:
        last_search = {
            "stages_ms": engine._last_timing,
            "total_ms": engine._last_total_ms,
        }

    # Index statistics
    total_chunks   = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_videos   = db.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    indexed_videos = db.execute(
        "SELECT COUNT(*) FROM videos WHERE status = 'indexed'"
    ).fetchone()[0]
    uncertain_pending = db.execute(
        "SELECT COUNT(*) FROM uncertain_chunks WHERE reviewed = 0"
    ).fetchone()[0]
    total_impressions = db.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0]
    total_clicks      = db.execute(
        "SELECT COUNT(*) FROM retrieval_log WHERE clicked = 1"
    ).fetchone()[0]

    return {
        "latency":     latency,
        "last_search": last_search,
        "vram":        _vram_stats(),
        "conformal":   _conformal_stats(),
        "index": {
            "total_chunks":              total_chunks,
            "total_videos":              total_videos,
            "indexed_videos":            indexed_videos,
            "uncertain_chunks_pending":  uncertain_pending,
            "total_impressions_logged":  total_impressions,
            "total_clicks_logged":       total_clicks,
            "ctr": round(total_clicks / total_impressions, 4) if total_impressions else 0.0,
        },
    }
