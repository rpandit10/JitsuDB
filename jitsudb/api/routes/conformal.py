"""POST /api/conformal/calibrate — re-calibrate the conformal reranker on demand."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from jitsudb.api.dependencies import get_conformal_reranker, get_db

router = APIRouter(tags=["conformal"])


@router.post("/conformal/calibrate")
def calibrate(
    db: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    """Re-calibrate the conformal threshold using current click data.

    Reads (query_text, clicked_chunk, non_clicked_chunk) triples from
    retrieval_log and refits q_hat.  Returns the new threshold plus an
    estimate of empirical coverage on the calibration set.
    """
    conformal = get_conformal_reranker()
    if conformal is None:
        raise HTTPException(
            status_code=404,
            detail="Conformal prediction is not enabled. "
                   "Set reranker.conformal_coverage in config.yaml and restart.",
        )

    from jitsudb.retrieval.conformal import build_calibration_pairs_from_clicks

    pairs = build_calibration_pairs_from_clicks(db)
    if not pairs:
        raise HTTPException(
            status_code=422,
            detail="No click data with query_text available. "
                   "Run some searches and click results to collect training signal.",
        )

    q_hat = conformal.calibrate(pairs)
    coverage = conformal.empirical_coverage(pairs)
    n_pos = sum(1 for _, _, r in pairs if r)
    n_neg = sum(1 for _, _, r in pairs if not r)

    return {
        "q_hat": round(q_hat, 6),
        "target_coverage": conformal.target_coverage,
        "empirical_coverage": round(coverage, 4),
        "n_pairs": len(pairs),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


@router.get("/conformal/status")
def status() -> dict[str, Any]:
    """Return current conformal reranker state without triggering calibration."""
    conformal = get_conformal_reranker()
    if conformal is None:
        return {"enabled": False}

    return {
        "enabled": True,
        "target_coverage": conformal.target_coverage,
        "q_hat": round(conformal._q_hat, 6) if conformal._q_hat is not None else None,
        "calibrated": conformal._q_hat is not None,
    }
