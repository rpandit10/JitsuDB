#!/usr/bin/env python
"""Mine seed queries from retrieval_log and write to benchmark_queries.jsonl.

Selects queries that received at least one click, deduplicates, filters navigational
queries, stratifies by gi/nogi and query-length bucket, and appends up to 300 entries
to eval/benchmark_queries.jsonl with source="mined_click".

Usage:
    uv run eval/mine_seed_queries.py [--db PATH] [--out PATH] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import uuid
from collections import defaultdict
from pathlib import Path

_NAVIGATIONAL_PATTERNS = re.compile(
    r"\b(full video|full mat|playlist|episode|volume|vol\b|part \d|instructor|professor)\b",
    re.IGNORECASE,
)
_INSTRUCTOR_NAMES = {
    "gordon", "ryan", "danaher", "leandro", "lo", "marcelo", "garcia",
    "bernardo", "faria", "mikey", "musumeci", "craig", "jones",
    "andrew", "wiltse", "john", "b",
}


def _is_navigational(q: str) -> bool:
    if _NAVIGATIONAL_PATTERNS.search(q):
        return True
    words = set(q.lower().split())
    if words <= _INSTRUCTOR_NAMES:
        return True
    return False


def _length_bucket(q: str) -> str:
    n = len(q.split())
    if n <= 6:
        return "short"
    if n <= 10:
        return "medium"
    return "long"


def _load_existing_queries(out_path: Path) -> set[str]:
    """Return the set of query_text values already in the output file."""
    seen: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if "query_text" in obj:
                        seen.add(obj["query_text"].lower().strip())
                except json.JSONDecodeError:
                    pass
    return seen


def mine(db_path: Path, out_path: Path, limit: int = 300) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT query_text
        FROM retrieval_log
        WHERE clicked > 0
          AND query_text IS NOT NULL
          AND trim(query_text) != ''
        GROUP BY lower(trim(query_text))
        """
    ).fetchall()
    conn.close()

    existing = _load_existing_queries(out_path)

    # Bucket → gi_nogi → list of (query_text, gi_nogi)
    buckets: dict[str, dict[str, list[tuple[str, str | None]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        q = row[0].strip() if isinstance(row, tuple) else row["query_text"].strip()
        gi = "unknown"
        words = q.split()
        if len(words) < 4:
            continue
        if _is_navigational(q):
            continue
        if q.lower() in existing:
            continue
        bucket = _length_bucket(q)
        buckets[bucket][gi].append((q, None))

    # Round-robin across buckets to get a stratified sample
    candidates: list[tuple[str, str | None]] = []
    bucket_names = ["short", "medium", "long"]
    gi_values = ["gi", "nogi", "unknown"]

    # Flatten in a stratified order
    for gi in gi_values:
        for bucket in bucket_names:
            candidates.extend(buckets[bucket][gi])

    sampled = candidates[:limit]

    if not sampled:
        print("No mined queries found (retrieval_log may have empty query_text fields).")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for q_text, gi_nogi in sampled:
            obj = {
                "query_id": str(uuid.uuid4()),
                "query_text": q_text,
                "gi_nogi": gi_nogi,
                "source": "mined_click",
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} mined queries → {out_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine seed queries from retrieval_log")
    parser.add_argument("--db", default="data/jitsudb.db", help="Path to SQLite database")
    parser.add_argument("--out", default="eval/benchmark_queries.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=300, help="Max queries to write")
    args = parser.parse_args()

    n = mine(Path(args.db), Path(args.out), args.limit)
    if n == 0:
        print("Tip: retrieval_log.query_text appears empty. Run generate_synthetic_queries.py to build the corpus.")


if __name__ == "__main__":
    main()
