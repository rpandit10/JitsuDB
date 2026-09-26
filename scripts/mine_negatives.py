"""Mine hard-negative triplets from taxonomy_labels for encoder fine-tuning.

Usage:
    python -m scripts.mine_negatives [--output data/bjj_triplets.jsonl] [--limit 5000]

Output format — one JSON object per line:
    {"anchor": "<text>", "positive": "<text>", "negative": "<text>"}

Strategy:
    Group chunks by (position_label, technique_label). For each anchor chunk in a
    group, the positive is another chunk from the *same* group and the hard negative
    is a chunk from the *same position* but a *different technique* — maximally
    confusable for a vanilla retriever but semantically distinct.

    Minimum data requirements to produce a triplet:
      - A position label with ≥2 distinct technique sub-groups.
      - Each technique sub-group must contain ≥2 chunks (so anchor ≠ positive).
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path


def mine_triplets(
    conn: sqlite3.Connection,
    limit: int | None = None,
    seed: int | None = None,
) -> list[dict[str, str]]:
    """Return hard-negative triplets mined from the taxonomy_labels table."""
    if seed is not None:
        random.seed(seed)

    rows = conn.execute(
        """
        SELECT tl_pos.chunk_id,
               c.transcript_text,
               tl_pos.label_value  AS position,
               tl_tech.label_value AS technique
        FROM taxonomy_labels tl_pos
        JOIN taxonomy_labels tl_tech
            ON  tl_tech.chunk_id  = tl_pos.chunk_id
            AND tl_tech.label_type = 'technique'
        JOIN chunks c ON c.id = tl_pos.chunk_id
        WHERE tl_pos.label_type = 'position'
        """,
    ).fetchall()

    # position → technique → [(chunk_id, text), ...]
    groups: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        groups[row["position"]][row["technique"]].append(
            (row["chunk_id"], row["transcript_text"])
        )

    triplets: list[dict[str, str]] = []

    for position, tech_map in groups.items():
        if len(tech_map) < 2:
            # Cannot form a hard negative without a second technique in this position.
            continue

        for technique, candidates in tech_map.items():
            if len(candidates) < 2:
                # Need at least two chunks to form (anchor, positive) within the group.
                continue

            neg_pool = [
                text
                for t, items in tech_map.items()
                if t != technique
                for _, text in items
            ]
            if not neg_pool:
                continue

            random.shuffle(candidates)
            for i, (_, anchor_text) in enumerate(candidates):
                pos_text = candidates[(i + 1) % len(candidates)][1]
                neg_text = random.choice(neg_pool)
                triplets.append(
                    {"anchor": anchor_text, "positive": pos_text, "negative": neg_text}
                )

    random.shuffle(triplets)
    return triplets[:limit] if limit is not None else triplets


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine BJJ hard-negative triplets")
    parser.add_argument(
        "--output",
        default="data/bjj_triplets.jsonl",
        help="Output JSONL path (default: data/bjj_triplets.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum triplets to write (default: all found)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    from jitsudb.config import get_config
    from jitsudb.db import get_connection

    cfg = get_config()
    conn = get_connection()

    print("Mining hard-negative triplets …")
    triplets = mine_triplets(conn, limit=args.limit, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(triplets)} triplets → {out_path}")
    if len(triplets) < 2000:
        print(
            f"WARNING: only {len(triplets)} triplets found (target ≥ 2,000). "
            "Index more videos or reduce the minimum-group-size threshold."
        )


if __name__ == "__main__":
    main()
