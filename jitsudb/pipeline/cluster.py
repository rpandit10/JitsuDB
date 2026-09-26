"""HDBSCAN clustering for taxonomy label proposals.

Fetches all chunk embeddings from ChromaDB, runs HDBSCAN, then for each
cluster computes majority-vote labels from confidently-labeled chunks and
stores them in chunk_label_proposals for surfacing in the Label Review UI.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    import chromadb

_BATCH_SIZE = 500
_NOISE_LABEL = -1


def _fetch_all_embeddings(
    collection: "chromadb.Collection",
) -> tuple[list[str], np.ndarray]:
    """Return (ids, embeddings) for every chunk in the collection."""
    total = collection.count()
    if total == 0:
        return [], np.empty((0, 0), dtype=np.float32)

    all_ids: list[str] = []
    all_embs: list[np.ndarray] = []
    offset = 0
    while offset < total:
        result = collection.get(
            limit=_BATCH_SIZE,
            offset=offset,
            include=["embeddings"],
        )
        batch_ids = result["ids"]
        batch_embs = result["embeddings"]
        if not batch_ids:
            break
        all_ids.extend(batch_ids)
        all_embs.append(np.array(batch_embs, dtype=np.float32))
        offset += len(batch_ids)

    if not all_ids:
        return [], np.empty((0, 0), dtype=np.float32)

    return all_ids, np.vstack(all_embs)


def _confident_labels_by_chunk(conn: sqlite3.Connection) -> dict[str, dict[str, list[str]]]:
    """Return label map for chunks that are confidently labeled.

    A chunk is confident if it was never uncertain (not in uncertain_chunks)
    OR it has been reviewed (reviewed=1).
    """
    rows = conn.execute(
        """
        SELECT tl.chunk_id, tl.label_type, tl.label_value
        FROM taxonomy_labels tl
        WHERE tl.chunk_id NOT IN (
            SELECT chunk_id FROM uncertain_chunks WHERE reviewed = 0
        )
        """
    ).fetchall()

    label_map: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        cid = row["chunk_id"]
        if cid not in label_map:
            label_map[cid] = {"position": [], "intent": [], "technique": []}
        label_map[cid][row["label_type"]].append(row["label_value"])

    return label_map


def _majority_labels(
    cluster_ids: list[int],
    id_list: list[str],
    label_map: dict[str, dict[str, list[str]]],
    min_support: int = 2,
) -> tuple[list[str], list[str], list[str], float]:
    """Compute majority-vote labels for a cluster.

    Returns (positions, intents, techniques, confidence) where confidence
    is the fraction of confident chunks in the cluster that agree on the
    modal label (0.0 if the cluster has no confident chunks).
    """
    pos_counter: Counter = Counter()
    int_counter: Counter = Counter()
    tec_counter: Counter = Counter()
    confident_count = 0

    for idx in cluster_ids:
        cid = id_list[idx]
        labels = label_map.get(cid)
        if labels is None:
            continue
        confident_count += 1
        for p in labels["position"]:
            pos_counter[p] += 1
        for i in labels["intent"]:
            int_counter[i] += 1
        for t in labels["technique"]:
            tec_counter[t] += 1

    if confident_count == 0:
        return [], [], [], 0.0

    def _take(counter: Counter, threshold: int) -> list[str]:
        return [label for label, cnt in counter.most_common() if cnt >= threshold]

    threshold = max(1, min_support)
    positions  = _take(pos_counter, threshold)
    intents    = _take(int_counter, threshold)
    techniques = _take(tec_counter, threshold)

    top_count = max(
        (pos_counter.most_common(1)[0][1] if pos_counter else 0),
        (int_counter.most_common(1)[0][1] if int_counter else 0),
        1,
    )
    confidence = round(top_count / confident_count, 4)

    return positions, intents, techniques, confidence


def run_cluster_proposals(
    collection: "chromadb.Collection",
    conn: sqlite3.Connection,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> int:
    """Cluster all chunk embeddings with HDBSCAN and write label proposals.

    For each cluster (noise points skipped), computes majority-vote labels
    from confidently-labeled members and upserts a row into
    chunk_label_proposals for every chunk in the cluster.
    """
    from sklearn.cluster import HDBSCAN

    logger.info("Fetching embeddings for HDBSCAN clustering…")
    ids, embeddings = _fetch_all_embeddings(collection)
    if len(ids) < min_cluster_size:
        logger.warning(f"Only {len(ids)} chunks — skipping clustering")
        return 0

    logger.info(f"Running HDBSCAN on {len(ids)} chunks")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        store_centers="centroid",
    )
    labels_arr = clusterer.fit_predict(embeddings)

    n_clusters = int(labels_arr.max()) + 1
    n_noise    = int((labels_arr == _NOISE_LABEL).sum())
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

    label_map = _confident_labels_by_chunk(conn)

    cluster_members: dict[int, list[int]] = {}
    for idx, cluster_id in enumerate(labels_arr):
        if cluster_id == _NOISE_LABEL:
            continue
        cluster_members.setdefault(int(cluster_id), []).append(idx)

    now = datetime.now(timezone.utc).isoformat()
    written = 0

    for cluster_id, member_indices in cluster_members.items():
        positions, intents, techniques, confidence = _majority_labels(
            member_indices, ids, label_map
        )

        pos_json  = json.dumps(positions)
        int_json  = json.dumps(intents)
        tec_json  = json.dumps(techniques)

        for idx in member_indices:
            chunk_id = ids[idx]
            conn.execute(
                """
                INSERT INTO chunk_label_proposals
                    (chunk_id, cluster_id, proposed_positions, proposed_intents,
                     proposed_techniques, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    cluster_id          = excluded.cluster_id,
                    proposed_positions  = excluded.proposed_positions,
                    proposed_intents    = excluded.proposed_intents,
                    proposed_techniques = excluded.proposed_techniques,
                    confidence          = excluded.confidence,
                    created_at          = excluded.created_at
                """,
                (chunk_id, cluster_id, pos_json, int_json, tec_json, confidence, now),
            )
            written += 1

    conn.commit()
    logger.info(f"Cluster proposals written: {written}")
    return written


def get_proposals(
    chunk_id: str, conn: sqlite3.Connection
) -> dict[str, list[str]] | None:
    """Return HDBSCAN label proposals for a single chunk, or None if absent."""
    row = conn.execute(
        """
        SELECT proposed_positions, proposed_intents, proposed_techniques, confidence
        FROM chunk_label_proposals WHERE chunk_id = ?
        """,
        (chunk_id,),
    ).fetchone()

    if row is None:
        return None

    return {
        "positions":  json.loads(row["proposed_positions"]),
        "intents":    json.loads(row["proposed_intents"]),
        "techniques": json.loads(row["proposed_techniques"]),
        "confidence": row["confidence"],
    }
