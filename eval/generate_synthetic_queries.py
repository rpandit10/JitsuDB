#!/usr/bin/env python
"""Generate synthetic benchmark queries via Ollama from sampled chunks.

Samples 100 chunks weighted toward those absent from retrieval_log, sends each to
the local Ollama llama3.2:3b endpoint to generate one BJJ technique-retrieval query,
post-processes the query through the abbreviation expander, and appends results to
eval/benchmark_queries.jsonl with source="synthetic".

Usage:
    uv run eval/generate_synthetic_queries.py [--db PATH] [--out PATH] [--sample N] [--model NAME]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import time
import uuid
from pathlib import Path

import httpx

_PROMPT_TEMPLATE = """\
You are a BJJ practitioner writing search queries for a technique library.

Read this chunk of transcribed BJJ instructional video and write ONE search query \
that a student would type to find this exact clip. Requirements:
- 5 to 12 words
- Do NOT name the instructor
- Do NOT quote verbatim phrases from the transcript
- Focus on the specific technique or position being taught

Transcript chunk:
\"\"\"
{transcript}
\"\"\"

Respond with ONLY the query text. No explanation, no punctuation at the end."""

_WORD_COUNT_RE = re.compile(r"\s+")


def _word_count(s: str) -> int:
    return len(_WORD_COUNT_RE.split(s.strip()))


def _load_existing_ids(out_path: Path) -> set[str]:
    seen: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if obj.get("source") == "synthetic" and "chunk_id" in obj:
                        seen.add(obj["chunk_id"])
                except json.JSONDecodeError:
                    pass
    return seen


def _sample_chunks(
    conn: sqlite3.Connection,
    n: int,
    already_generated: set[str],
) -> list[dict]:
    """Sample n chunks weighted toward those absent from retrieval_log."""
    # Chunks that appear in retrieval_log (already seen by the system)
    seen_in_log: set[str] = set(
        row[0]
        for row in conn.execute("SELECT DISTINCT chunk_id FROM retrieval_log").fetchall()
    )

    rows = conn.execute(
        """
        SELECT c.id, c.transcript_text, c.video_id, c.context,
               v.primary_category, v.gi_nogi
        FROM chunks c
        JOIN videos v ON v.id = c.video_id
        WHERE v.status = 'indexed'
          AND length(c.transcript_text) > 100
        """
    ).fetchall()

    unseen = [r for r in rows if r[0] not in seen_in_log and r[0] not in already_generated]
    seen = [r for r in rows if r[0] in seen_in_log and r[0] not in already_generated]

    # 80% from unseen, 20% from seen (to cover distribution gaps)
    n_unseen = min(int(n * 0.8), len(unseen))
    n_seen = min(n - n_unseen, len(seen))

    sampled = random.sample(unseen, n_unseen) + random.sample(seen, n_seen)
    random.shuffle(sampled)
    return [
        {
            "id": r[0],
            "transcript_text": r[1],
            "video_id": r[2],
            "context": r[3],
            "category": r[4],
            "gi_nogi": r[5],
        }
        for r in sampled
    ]


def _call_ollama(
    transcript: str,
    ollama_url: str,
    model: str,
    timeout: float = 30.0,
) -> str | None:
    prompt = _PROMPT_TEMPLATE.format(transcript=transcript[:800])
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        # Strip surrounding quotes if Ollama added them
        text = text.strip('"').strip("'")
        return text
    except Exception as exc:
        print(f"  Ollama error: {exc}")
        return None


def _post_process(query: str) -> str:
    from jitsudb.search.query_preprocessor import expand_abbreviations
    return expand_abbreviations(query.strip())


def generate(
    db_path: Path,
    out_path: Path,
    sample_n: int = 100,
    ollama_url: str = "http://localhost:11434",
    model: str = "llama3.2:3b",
    delay_sec: float = 0.2,
) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    already_generated = _load_existing_ids(out_path)
    chunks = _sample_chunks(conn, sample_n, already_generated)
    conn.close()

    print(f"Sampled {len(chunks)} chunks for synthetic query generation")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("a", encoding="utf-8") as fh:
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}/{len(chunks)}] chunk {chunk['id'][:8]}…", end=" ", flush=True)

            raw = _call_ollama(chunk["transcript_text"], ollama_url, model)
            if not raw:
                print("SKIP (no response)")
                continue

            try:
                query = _post_process(raw)
            except Exception:
                query = raw.strip()

            # Validate word count
            wc = _word_count(query)
            if wc < 4 or wc > 20:
                print(f"SKIP (bad length: {wc} words: {query!r})")
                continue

            obj = {
                "query_id": str(uuid.uuid4()),
                "query_text": query,
                "gi_nogi": chunk.get("gi_nogi"),
                "source": "synthetic",
                "chunk_id": chunk["id"],
                "category": chunk.get("category"),
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
            print(f"OK  [{wc}w] {query!r}")

            if delay_sec > 0:
                time.sleep(delay_sec)

    print(f"\nWrote {written} synthetic queries → {out_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark queries via Ollama")
    parser.add_argument("--db", default="data/jitsudb.db", help="Path to SQLite database")
    parser.add_argument("--out", default="eval/benchmark_queries.jsonl", help="Output JSONL path")
    parser.add_argument("--sample", type=int, default=100, help="Chunks to sample")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between Ollama calls")
    args = parser.parse_args()

    generate(
        db_path=Path(args.db),
        out_path=Path(args.out),
        sample_n=args.sample,
        ollama_url=args.ollama_url,
        model=args.model,
        delay_sec=args.delay,
    )


if __name__ == "__main__":
    main()
