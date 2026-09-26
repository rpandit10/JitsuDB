"""Result ranking and deduplication."""

from __future__ import annotations

from collections import defaultdict

from jitsudb.models import SearchResult


def rank_results(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Sort results by relevance_score descending, then deduplicate overlapping chunks."""
    deduped = deduplicate(results)
    deduped.sort(key=lambda r: r.relevance_score, reverse=True)
    return deduped[:top_k]


def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """Remove overlapping chunks from the same video, keeping the highest-scoring one.

    Two chunks overlap when their time ranges intersect. Among overlapping
    chunks from the same video, only the highest relevance_score is kept.
    """
    by_video: dict[str, list[SearchResult]] = defaultdict(list)
    for r in results:
        by_video[r.chunk.video_id].append(r)

    deduped: list[SearchResult] = []
    for group in by_video.values():
        group.sort(key=lambda r: r.relevance_score, reverse=True)
        selected: list[SearchResult] = []
        for candidate in group:
            c = candidate.chunk
            overlaps = any(
                c.start_time < sel.chunk.end_time and sel.chunk.start_time < c.end_time
                for sel in selected
            )
            if not overlaps:
                selected.append(candidate)
        deduped.extend(selected)

    return deduped


def build_snippet(transcript_text: str, query: str, context_chars: int = 200) -> str:
    """Extract a highlighted snippet from a transcript around the best query match.

    Finds the substring of ``transcript_text`` closest to the query terms and
    returns up to ``context_chars`` characters centred on that match.
    """
    text_lower = transcript_text.lower()
    best_pos = -1

    for word in query.lower().split():
        if len(word) < 3:
            continue
        pos = text_lower.find(word)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos == -1:
        if len(transcript_text) <= context_chars:
            return transcript_text
        return transcript_text[:context_chars] + "…"

    half = context_chars // 2
    start = max(0, best_pos - half)
    end = min(len(transcript_text), best_pos + half)

    snippet = transcript_text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(transcript_text):
        snippet = snippet + "…"
    return snippet
