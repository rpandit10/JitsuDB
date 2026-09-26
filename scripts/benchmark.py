"""Standalone search latency benchmark.

Runs a fixed query list through the search engine N times each and reports
P50/P95/P99 wall-clock latency per query and overall.  Does not require a
running server — loads the engine directly.

Usage::

    cd /path/to/JitsuDB
    python scripts/benchmark.py                     # 3 warmup + 10 timed runs
    python scripts/benchmark.py --runs 20           # 20 timed runs per query
    python scripts/benchmark.py --queries "kimura" "armbar from guard"
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_QUERIES = [
    "kimura from side control",
    "rear naked choke from back control",
    "triangle choke from guard",
    "heel hook from outside ashi",
    "armbar from mount",
    "single leg x guard sweep",
    "half guard pass",
    "de la riva guard sweep",
    "closed guard break and pass",
    "back take from turtle",
]


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    idx = min(int(len(vals) * p / 100), len(vals) - 1)
    return vals[idx]


def run_benchmark(
    queries: list[str],
    engine,
    warmup: int = 3,
    runs: int = 10,
) -> dict:
    print(f"Warming up ({warmup} run{'s' if warmup != 1 else ''} discarded)…")
    for q in queries[:1]:
        for _ in range(warmup):
            engine.search(q)

    print(f"Benchmarking {len(queries)} queries × {runs} runs each…\n")

    all_latencies: list[float] = []
    per_query: list[dict] = []

    for query in queries:
        lats: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            engine.search(query)
            lats.append((time.perf_counter() - t0) * 1000)
        lats.sort()
        all_latencies.extend(lats)

        row = {
            "query":   query,
            "mean_ms": round(sum(lats) / len(lats), 1),
            "p50_ms":  round(_pct(lats, 50), 1),
            "p95_ms":  round(_pct(lats, 95), 1),
            "min_ms":  round(lats[0], 1),
            "max_ms":  round(lats[-1], 1),
        }
        per_query.append(row)
        print(
            f"  {row['mean_ms']:6.1f}ms mean  "
            f"p50={row['p50_ms']:.1f}  p95={row['p95_ms']:.1f}  "
            f"[{row['min_ms']:.1f}–{row['max_ms']:.1f}]  {query!r}"
        )

    all_latencies.sort()
    n = len(all_latencies)
    overall = {
        "n_samples": n,
        "mean_ms":   round(sum(all_latencies) / n, 1),
        "p50_ms":    round(_pct(all_latencies, 50), 1),
        "p95_ms":    round(_pct(all_latencies, 95), 1),
        "p99_ms":    round(_pct(all_latencies, 99), 1),
        "min_ms":    round(all_latencies[0], 1),
        "max_ms":    round(all_latencies[-1], 1),
    }
    return {"overall": overall, "per_query": per_query}


def _print_summary(results: dict) -> None:
    o = results["overall"]
    print(f"\n{'═' * 44}")
    print("  Overall latency summary")
    print(f"{'═' * 44}")
    print(f"  Samples : {o['n_samples']}")
    print(f"  Mean    : {o['mean_ms']:>8.1f} ms")
    print(f"  P50     : {o['p50_ms']:>8.1f} ms")
    print(f"  P95     : {o['p95_ms']:>8.1f} ms")
    print(f"  P99     : {o['p99_ms']:>8.1f} ms")
    print(f"  Min     : {o['min_ms']:>8.1f} ms")
    print(f"  Max     : {o['max_ms']:>8.1f} ms")

    # Stage breakdown from last search
    try:
        from jitsudb.api.dependencies import get_search_engine
        engine = get_search_engine()
        if engine._last_timing:
            print()
            print("  Stage breakdown (last search):")
            for stage, ms in sorted(engine._last_timing.items(), key=lambda x: -x[1]):
                bar = "█" * max(1, int(ms / (engine._last_total_ms or 1) * 20))
                print(f"    {stage:<14} {ms:>7.1f} ms  {bar}")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(dev) / 1024 ** 3
            peak  = torch.cuda.max_memory_allocated(dev) / 1024 ** 3
            name  = torch.cuda.get_device_properties(dev).name
            print()
            print(f"  VRAM ({name}): {alloc:.2f} GB allocated, {peak:.2f} GB peak")
    except (ImportError, RuntimeError):
        pass

    print(f"{'═' * 44}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="JitsuDB search latency benchmark")
    parser.add_argument("--runs",    type=int, default=10, help="Timed runs per query")
    parser.add_argument("--warmup",  type=int, default=3,  help="Warmup runs (discarded)")
    parser.add_argument("queries",   nargs="*",            help="Queries to benchmark")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES

    from jitsudb.api.dependencies import get_search_engine
    from jitsudb.db import init_db

    print("Initialising…")
    init_db()
    engine = get_search_engine()

    results = run_benchmark(queries, engine, warmup=args.warmup, runs=args.runs)
    _print_summary(results)


if __name__ == "__main__":
    main()
