"""Retrieval evaluation harness for JitsuDB.

Computes Precision@k, Recall@k, nDCG@k, and MRR against a held-out query set.

Two modes:

  1. **Synthetic** (default when ``eval/golden_qa.jsonl`` is empty or absent):
     Auto-generates queries from (position, technique) label pairs that have at
     least ``--min-relevant`` matching chunks in the database.  This requires no
     manual annotation and gives real numbers immediately after indexing.

  2. **Golden** (used when ``eval/golden_qa.jsonl`` contains entries):
     Each line must be JSON with keys ``query`` (str) and
     ``relevant_chunk_ids`` (list[str]).

Usage::

    cd /path/to/JitsuDB
    python eval/run_eval.py                     # synthetic, k=10
    python eval/run_eval.py --k 5 --n 30        # 30 synthetic queries, P@5
    python eval/run_eval.py --golden            # force golden mode
    python eval/run_eval.py --out eval/results/my_run.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / k if k else 0.0


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / len(relevant)


def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        (1.0 / math.log2(rank + 2))
        for rank, cid in enumerate(retrieved[:k])
        if cid in relevant
    )
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0.0


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def generate_synthetic_queries(conn, n: int = 50, min_relevant: int = 3) -> list[dict]:
    """Build queries from (position, technique) label pairs with >= min_relevant chunks."""
    pairs = conn.execute(
        """
        SELECT p.label_value AS position,
               t.label_value AS technique,
               COUNT(DISTINCT p.chunk_id) AS cnt
        FROM taxonomy_labels p
        JOIN taxonomy_labels t ON t.chunk_id = p.chunk_id
        WHERE p.label_type = 'position' AND t.label_type = 'technique'
        GROUP BY p.label_value, t.label_value
        HAVING cnt >= ?
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (min_relevant, n),
    ).fetchall()

    if not pairs:
        return []

    queries: list[dict] = []
    for row in pairs:
        pos = row["position"].replace("_", " ")
        tec = row["technique"].replace("_", " ")
        relevant = conn.execute(
            """
            SELECT DISTINCT p.chunk_id
            FROM taxonomy_labels p
            JOIN taxonomy_labels t ON t.chunk_id = p.chunk_id
            WHERE p.label_type = 'position' AND p.label_value = ?
              AND t.label_type = 'technique' AND t.label_value = ?
            """,
            (row["position"], row["technique"]),
        ).fetchall()
        queries.append(
            {
                "query": f"{tec} from {pos}",
                "relevant_chunk_ids": [r["chunk_id"] for r in relevant],
                "_source": "synthetic",
                "_position": row["position"],
                "_technique": row["technique"],
            }
        )
    return queries


def load_golden_queries(path: Path) -> list[dict]:
    queries = []
    if not path.exists():
        return queries
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
            if "query" in obj and "relevant_chunk_ids" in obj:
                queries.append(obj)
        except json.JSONDecodeError:
            continue
    return queries


def run_evaluation(
    queries: list[dict],
    engine,
    k: int = 10,
    verbose: bool = False,
) -> dict:
    """Run each query through the engine and compute retrieval metrics.

    Returns a results dict suitable for JSON serialisation.
    """
    per_query = []
    slow_threshold_ms = 500.0

    for item in queries:
        query_text = item["query"]
        relevant = set(item["relevant_chunk_ids"])

        t0 = time.perf_counter()
        results = engine.search(query_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        retrieved = [r.chunk.id for r in results]

        entry = {
            "query":       query_text,
            "latency_ms":  round(elapsed_ms, 1),
            "n_relevant":  len(relevant),
            "n_retrieved": len(retrieved),
            f"P@{k}":      round(_precision_at_k(retrieved, relevant, k), 4),
            f"R@{k}":      round(_recall_at_k(retrieved, relevant, k), 4),
            f"nDCG@{k}":   round(_ndcg_at_k(retrieved, relevant, k), 4),
            "MRR":         round(_mrr(retrieved, relevant), 4),
        }
        if "_source" in item:
            entry["_source"] = item["_source"]

        per_query.append(entry)

        if verbose or elapsed_ms > slow_threshold_ms:
            flag = "  [SLOW]" if elapsed_ms > slow_threshold_ms else ""
            print(
                f"  {elapsed_ms:6.0f}ms  P@{k}={entry[f'P@{k}']:.2f}  "
                f"nDCG={entry[f'nDCG@{k}']:.2f}  {query_text!r}{flag}"
            )

    if not per_query:
        return {"error": "no queries evaluated", "per_query": []}

    n = len(per_query)

    def _avg(key: str) -> float:
        return round(sum(q[key] for q in per_query) / n, 4)

    latencies = sorted(q["latency_ms"] for q in per_query)

    def _pct(p: float) -> float:
        idx = min(int(n * p / 100), n - 1)
        return round(latencies[idx], 1)

    return {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "n_queries":   n,
        "k":           k,
        "aggregate": {
            f"P@{k}":      _avg(f"P@{k}"),
            f"R@{k}":      _avg(f"R@{k}"),
            f"nDCG@{k}":   _avg(f"nDCG@{k}"),
            "MRR":         _avg("MRR"),
        },
        "latency_ms": {
            "mean": round(sum(latencies) / n, 1),
            "p50":  _pct(50),
            "p95":  _pct(95),
            "p99":  _pct(99),
            "max":  round(latencies[-1], 1),
        },
        "per_query": per_query,
    }


def _print_report(results: dict) -> None:
    k = results["k"]
    agg = results["aggregate"]
    lat = results["latency_ms"]

    bar = "─" * 42
    print(f"\n{'═' * 42}")
    print(f"  JitsuDB Retrieval Evaluation — k={k}")
    print(f"{'═' * 42}")
    print(f"  Queries evaluated : {results['n_queries']}")
    print()
    print(f"  {'Metric':<14}  {'Value':>8}")
    print(f"  {bar[:38]}")
    print(f"  {'P@' + str(k):<14}  {agg[f'P@{k}']:>8.4f}")
    print(f"  {'R@' + str(k):<14}  {agg[f'R@{k}']:>8.4f}")
    print(f"  {'nDCG@' + str(k):<14}  {agg[f'nDCG@{k}']:>8.4f}")
    print(f"  {'MRR':<14}  {agg['MRR']:>8.4f}")
    print()
    print(f"  {'Latency':>14}  {'ms':>8}")
    print(f"  {bar[:38]}")
    print(f"  {'mean':<14}  {lat['mean']:>8.1f}")
    print(f"  {'p50':<14}  {lat['p50']:>8.1f}")
    print(f"  {'p95':<14}  {lat['p95']:>8.1f}")
    print(f"  {'p99':<14}  {lat['p99']:>8.1f}")
    print(f"  {'max':<14}  {lat['max']:>8.1f}")

    slow = sorted(results["per_query"], key=lambda q: q["latency_ms"], reverse=True)[:5]
    if slow:
        print()
        print(f"  Slowest queries:")
        for q in slow:
            print(f"    {q['latency_ms']:6.0f}ms  {q['query']!r}")
    print(f"{'═' * 42}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="JitsuDB retrieval evaluation")
    parser.add_argument("--k",           type=int,  default=10,   help="Rank cutoff for metrics")
    parser.add_argument("--n",           type=int,  default=50,   help="Max synthetic queries to generate")
    parser.add_argument("--min-relevant",type=int,  default=3,    help="Min matching chunks per synthetic query")
    parser.add_argument("--golden",      action="store_true",      help="Force golden mode (skip synthetic)")
    parser.add_argument("--verbose",     action="store_true",      help="Print per-query results")
    parser.add_argument("--out",         type=str,  default=None, help="Write JSON results to this path")
    args = parser.parse_args()

    from jitsudb.api.dependencies import get_search_engine
    from jitsudb.db import get_connection, init_db

    print("Initialising database and search engine…")
    init_db()
    conn = get_connection()
    engine = get_search_engine()

    golden_path = Path(__file__).parent / "golden_qa.jsonl"
    golden = load_golden_queries(golden_path)

    if args.golden and not golden:
        print(f"ERROR: --golden specified but {golden_path} has no valid entries.", file=sys.stderr)
        sys.exit(1)

    if golden and not args.golden:
        queries = golden
        mode = f"golden ({len(golden)} entries from {golden_path.name})"
    else:
        print(f"Generating synthetic queries (n={args.n}, min_relevant={args.min_relevant})…")
        queries = generate_synthetic_queries(conn, n=args.n, min_relevant=args.min_relevant)
        mode = f"synthetic ({len(queries)} queries auto-generated from taxonomy labels)"
        if not queries:
            print(
                "ERROR: No (position, technique) pairs with enough matching chunks found.\n"
                "Index more videos first, or lower --min-relevant.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Eval set : {mode}")
    print(f"Running {len(queries)} queries through search engine…\n")

    results = run_evaluation(queries, engine, k=args.k, verbose=args.verbose)
    _print_report(results)

    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / "results" /
        f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
