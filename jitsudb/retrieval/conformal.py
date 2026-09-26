"""Conformal prediction wrapper for the cross-encoder re-ranker.

Conformal prediction gives a statistically guaranteed retrieval coverage: at
least `target_coverage` fraction of truly relevant chunks will appear in the
returned prediction set, regardless of the query distribution — provided the
calibration set is i.i.d. with the test distribution.

Usage::

    conformal = ConformalReranker(cross_encoder_reranker, target_coverage=0.90)
    pairs = build_calibration_pairs("eval/golden_qa.jsonl", db_conn)
    q_hat = conformal.calibrate(pairs)
    # conformal.rerank() now acts as a drop-in for CrossEncoderReranker.rerank()
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from jitsudb.models import SearchResult
    from jitsudb.search.reranker import CrossEncoderReranker


class ConformalReranker:
    """Wraps CrossEncoderReranker with a calibrated score threshold.

    The threshold q_hat is chosen so that at least `target_coverage` of all
    relevant (query, chunk) pairs from the calibration set score >= q_hat.
    At inference, every candidate whose raw cross-encoder score >= q_hat is
    included in the result set; the set falls back to top-1 if it would
    otherwise be empty.
    """

    def __init__(
        self,
        reranker: "CrossEncoderReranker",
        target_coverage: float = 0.90,
    ) -> None:
        self.reranker = reranker
        self.target_coverage = target_coverage
        self._q_hat: float | None = None

    def calibrate(self, calibration_pairs: list[tuple[str, str, bool]]) -> float:
        """Fit q_hat from (query, chunk_text, is_relevant) triples.

        q_hat is set to the (1 − target_coverage) quantile of the raw
        cross-encoder scores for all positive (relevant=True) pairs.
        This guarantees that at least target_coverage of positives score
        above q_hat on the calibration set.
        """
        positive_pairs = [(q, t) for q, t, r in calibration_pairs if r]
        if not positive_pairs:
            self._q_hat = -float("inf")
            logger.warning("ConformalReranker: no positive pairs — q_hat set to -inf")
            return self._q_hat

        scores = self.reranker.model.predict(positive_pairs, show_progress_bar=False)
        # q_hat = 10th percentile of positive scores when target_coverage=0.90
        self._q_hat = float(np.quantile(scores, 1.0 - self.target_coverage))
        logger.info(
            f"ConformalReranker calibrated: q_hat={self._q_hat:.4f} "
            f"from {len(positive_pairs)} positive pairs "
            f"(target coverage {self.target_coverage:.0%})"
        )
        return self._q_hat

    def predict_set(
        self,
        query: str,
        candidates: list["SearchResult"],
    ) -> list["SearchResult"]:
        """Return all candidates with raw cross-encoder score >= q_hat.

        Falls back to the single highest-scoring candidate if the set would
        be empty (guarantees at least one result is always returned).
        """
        if not candidates:
            return candidates

        if self._q_hat is None:
            # Not calibrated — fall back to regular reranking
            return self.reranker.rerank(query, candidates, top_k=len(candidates))

        pairs = [(query, r.chunk.transcript_text) for r in candidates]
        raw_scores = self.reranker.model.predict(pairs, show_progress_bar=False)

        for result, score in zip(candidates, raw_scores):
            result.relevance_score = float(score)

        filtered = [r for r in candidates if r.relevance_score >= self._q_hat]
        if not filtered:
            filtered = [max(candidates, key=lambda r: r.relevance_score)]

        # Normalise scores within the filtered set for display
        if len(filtered) > 1:
            lo = min(r.relevance_score for r in filtered)
            hi = max(r.relevance_score for r in filtered)
            if hi > lo:
                for r in filtered:
                    r.relevance_score = (r.relevance_score - lo) / (hi - lo)
            else:
                for r in filtered:
                    r.relevance_score = 1.0
        else:
            filtered[0].relevance_score = 1.0

        filtered.sort(key=lambda r: r.relevance_score, reverse=True)
        return filtered

    def rerank(
        self,
        query: str,
        results: list["SearchResult"],
        top_k: int,
    ) -> list["SearchResult"]:
        """Drop-in replacement for CrossEncoderReranker.rerank().

        Ignores top_k — the conformal set size is determined by q_hat.
        The engine always receives a non-empty list.
        """
        return self.predict_set(query, results)

    def empirical_coverage(
        self,
        calibration_pairs: list[tuple[str, str, bool]],
    ) -> float:
        """Return fraction of positive pairs that score >= q_hat."""
        if self._q_hat is None:
            raise RuntimeError("Must call calibrate() before empirical_coverage()")

        positive_pairs = [(q, t) for q, t, r in calibration_pairs if r]
        if not positive_pairs:
            return 1.0

        scores = self.reranker.model.predict(positive_pairs, show_progress_bar=False)
        coverage = float(np.mean(np.array(scores) >= self._q_hat))
        return coverage

    def __repr__(self) -> str:
        status = f"q_hat={self._q_hat:.4f}" if self._q_hat is not None else "uncalibrated"
        return (
            f"ConformalReranker(coverage={self.target_coverage:.0%}, {status})"
        )


def build_calibration_pairs_from_clicks(
    conn: sqlite3.Connection,
    n_negatives_per_positive: int = 4,
    seed: int = 42,
) -> list[tuple[str, str, bool]]:
    """Build calibration triples from retrieval_log click data.

    Positives  = (query_text, clicked_chunk_text).
    Negatives  = non-clicked chunks shown in the same session/query.

    Requires clicks with query_text populated. Falls back to random negatives
    from the corpus if a session has no non-clicked impressions.
    """
    rng = random.Random(seed)

    rows = conn.execute(
        """
        SELECT rl.session_id, rl.query_text, rl.chunk_id,
               rl.clicked, c.transcript_text
        FROM retrieval_log rl
        JOIN chunks c ON c.id = rl.chunk_id
        WHERE rl.query_text != ''
        ORDER BY rl.session_id, rl.query_text, rl.rank
        """
    ).fetchall()

    if not rows:
        logger.warning("ConformalReranker: no click data with query_text — returning empty set")
        return []

    # Group by (session_id, query_text)
    groups: dict[tuple[str, str], dict[str, list[str]]] = {}
    for row in rows:
        key = (row["session_id"], row["query_text"])
        if key not in groups:
            groups[key] = {"clicked": [], "shown": []}
        bucket = "clicked" if row["clicked"] else "shown"
        groups[key][bucket].append(row["transcript_text"])

    pairs: list[tuple[str, str, bool]] = []

    for (_, query_text), g in groups.items():
        if not g["clicked"]:
            continue
        for pos_text in g["clicked"]:
            pairs.append((query_text, pos_text, True))
            neg_pool = g["shown"]
            if not neg_pool:
                # No in-session negatives — skip (will be supplemented below)
                continue
            for _ in range(min(n_negatives_per_positive, len(neg_pool))):
                pairs.append((query_text, rng.choice(neg_pool), False))

    # If we have positives but insufficient negatives, pad with random corpus chunks
    n_pos = sum(1 for _, _, r in pairs if r)
    n_neg = sum(1 for _, _, r in pairs if not r)
    needed = max(0, n_pos * n_negatives_per_positive - n_neg)
    if needed > 0:
        neg_rows = conn.execute(
            "SELECT transcript_text FROM chunks ORDER BY RANDOM() LIMIT ?",
            (needed,),
        ).fetchall()
        # Use the most common query as the anchor for corpus negatives
        from collections import Counter
        query_counts: Counter = Counter(q for q, _, r in pairs if r)
        if query_counts:
            anchor_q = query_counts.most_common(1)[0][0]
            for row in neg_rows:
                pairs.append((anchor_q, row[0], False))

    rng.shuffle(pairs)
    n_pos = sum(1 for _, _, r in pairs if r)
    n_neg = sum(1 for _, _, r in pairs if not r)
    logger.info(
        f"Click-based calibration set: {len(pairs)} pairs "
        f"({n_pos} positive, {n_neg} negative)"
    )
    return pairs


def build_calibration_pairs(
    qa_path: str | Path = "eval/golden_qa.jsonl",
    conn: sqlite3.Connection | None = None,
    n_negatives_per_positive: int = 4,
    seed: int = 42,
) -> list[tuple[str, str, bool]]:
    """Build (query, chunk_text, is_relevant) triples from the golden QA set.

    For each golden QA entry:
    - Positive pairs: (question, relevant_chunk_text) for each relevant_chunk_id.
    - Negative pairs: randomly sampled non-relevant chunks from the DB.
    """
    qa_path = Path(qa_path)
    if not qa_path.exists():
        logger.warning(f"Golden QA not found at {qa_path} — returning empty calibration set")
        return []

    if conn is None:
        from jitsudb.db import get_connection
        conn = get_connection()

    qa_entries = [
        json.loads(line)
        for line in qa_path.read_text().splitlines()
        if line.strip()
    ]

    rng = random.Random(seed)
    pairs: list[tuple[str, str, bool]] = []

    for entry in qa_entries:
        question = entry["question"]
        relevant_ids: list[str] = entry["relevant_chunk_ids"]

        if relevant_ids:
            ph = ",".join("?" * len(relevant_ids))
            rows = conn.execute(
                f"SELECT id, transcript_text FROM chunks WHERE id IN ({ph})",
                relevant_ids,
            ).fetchall()
            for row in rows:
                pairs.append((question, row[1], True))

            # Negative pairs: random chunks NOT in the relevant set
            neg_rows = conn.execute(
                f"SELECT transcript_text FROM chunks "
                f"WHERE id NOT IN ({ph}) "
                f"ORDER BY RANDOM() LIMIT ?",
                (*relevant_ids, n_negatives_per_positive * len(relevant_ids)),
            ).fetchall()
            for row in neg_rows:
                pairs.append((question, row[0], False))

    rng.shuffle(pairs)
    n_pos = sum(1 for _, _, r in pairs if r)
    n_neg = sum(1 for _, _, r in pairs if not r)
    logger.info(
        f"Calibration set: {len(pairs)} pairs "
        f"({n_pos} positive, {n_neg} negative)"
    )
    return pairs
