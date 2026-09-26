"""CLI entry point: python -m jitsudb.search "<query>" [--top-k N]"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m jitsudb.search",
        description="Search the JitsuDB index from the command line.",
    )
    parser.add_argument("query", nargs="+", help="Natural language search query")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum results to show")
    args = parser.parse_args()

    query = " ".join(args.query)

    from jitsudb.api.dependencies import get_search_engine
    from jitsudb.db import init_db

    init_db()
    engine = get_search_engine()
    results = engine.search(query)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results[: args.top_k], 1):
        mins, secs = divmod(int(result.chunk.start_time), 60)
        timestamp = f"{mins}:{secs:02d}"
        print(f"\n{i}. {result.video.title}")
        print(f"   Instructor : {result.video.instructor}")
        print(f"   Timestamp  : {timestamp}")
        print(f"   Score      : {result.relevance_score:.3f}")
        if result.chunk.position:
            print(f"   Position   : {', '.join(result.chunk.position)}")
        if result.chunk.techniques:
            print(f"   Techniques : {', '.join(result.chunk.techniques)}")
        print(f"   {result.snippet}")


if __name__ == "__main__":
    main()
