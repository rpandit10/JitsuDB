"""Transcript chunker — sliding window with sentence boundary snapping."""

from __future__ import annotations

from jitsudb.constants import NATURAL_PAUSE_THRESHOLD_SECONDS
from jitsudb.models import RawChunk, WordTimestamp

# Verbal cues that typically open a new instructional section
_SECTION_PHRASES = [
    "now let's look at",
    "now we're going to",
    "now i want to show",
    "next technique",
    "the next thing",
    "next we're going to",
    "let's move on",
    "moving on to",
    "okay so next",
    "alright so next",
    "for the next",
]

_SENTENCE_ENDERS = frozenset(".?!")


def chunk_transcript(
    words: list[WordTimestamp],
    chunk_duration: float,
    overlap: float,
    snap_window: float,
    chapter_breaks: list[float] | None = None,
) -> list[RawChunk]:
    """Divide a word-level transcript into overlapping chunks.

    Algorithm:
    1. Slide a window of ``chunk_duration`` seconds, advancing by
       ``chunk_duration - overlap`` seconds each step.
    2. At each proposed boundary, search within ``snap_window`` seconds for a
       sentence-ending word (period / ? / !) or a natural pause
       > ``NATURAL_PAUSE_THRESHOLD_SECONDS``. Snap to the closest one found.
    3. If a chapter break falls inside the current window, force a boundary
       there instead (regardless of the snap window).
    """
    if not words:
        return []

    total_duration = words[-1].end
    step = chunk_duration - overlap
    forced = sorted(chapter_breaks or [])

    chunks: list[RawChunk] = []
    chunk_start = words[0].start

    while chunk_start < total_duration - 0.5:
        proposed_end = chunk_start + chunk_duration

        # Check for a forced chapter break inside this window
        forced_break = next(
            (b for b in forced if chunk_start < b <= proposed_end),
            None,
        )

        if forced_break is not None:
            actual_end = forced_break
        else:
            actual_end = _snap_boundary(words, proposed_end, snap_window)
            actual_end = min(actual_end, total_duration)

        chunk_words = [w for w in words if w.start >= chunk_start and w.start < actual_end]

        if chunk_words:
            chunks.append(
                RawChunk(
                    start_time=chunk_words[0].start,
                    end_time=chunk_words[-1].end,
                    words=chunk_words,
                    transcript_text=words_to_text(chunk_words),
                )
            )

        if forced_break is not None:
            chunk_start = forced_break
        else:
            chunk_start = chunk_start + step

    # Ensure the tail of the audio is captured if the last chunk ended early
    if chunks and chunks[-1].end_time < total_duration - 1.0:
        tail_start = max(chunks[-1].end_time - overlap, words[0].start)
        tail_words = [w for w in words if w.start >= tail_start]
        if tail_words:
            chunks.append(
                RawChunk(
                    start_time=tail_words[0].start,
                    end_time=tail_words[-1].end,
                    words=tail_words,
                    transcript_text=words_to_text(tail_words),
                )
            )

    return chunks


def detect_chapter_breaks(words: list[WordTimestamp], silence_gap: float = 3.0) -> list[float]:
    """Detect chapter boundaries from silence gaps and verbal section markers."""
    breaks: list[float] = []

    # Silence-gap detection
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if gap >= silence_gap:
            breaks.append(words[i].end)

    # Verbal section-marker detection
    # Build a character-offset → timestamp map so we can convert text positions
    char_positions: list[tuple[int, float]] = []  # (char_start_in_joined, word_start_time)
    pos = 0
    joined_parts: list[str] = []
    for w in words:
        char_positions.append((pos, w.start))
        part = w.word.strip().lower()
        joined_parts.append(part)
        pos += len(part) + 1  # +1 for the joining space

    joined_text = " ".join(joined_parts)

    for phrase in _SECTION_PHRASES:
        search_start = 0
        while True:
            idx = joined_text.find(phrase, search_start)
            if idx == -1:
                break
            # Find the word whose char range covers idx
            ts = _char_pos_to_time(idx, char_positions)
            if ts is not None:
                breaks.append(ts)
            search_start = idx + len(phrase)

    return sorted(set(breaks))


def words_to_text(words: list[WordTimestamp]) -> str:
    """Reconstruct a plain text string from a list of WordTimestamp objects."""
    return " ".join(w.word.strip() for w in words)


def _snap_boundary(
    words: list[WordTimestamp],
    proposed: float,
    window: float,
) -> float:
    """Return the best sentence boundary near ``proposed``, or ``proposed`` itself.

    Searches within [proposed - window, proposed + window] for:
    - A word ending with a sentence-terminal character (., ?, !)
    - A natural pause between adjacent words > NATURAL_PAUSE_THRESHOLD_SECONDS

    Returns the candidate closest to ``proposed``.
    """
    lo = proposed - window
    hi = proposed + window

    best_time: float | None = None
    best_dist = float("inf")

    def _consider(t: float) -> None:
        nonlocal best_time, best_dist
        if lo <= t <= hi:
            dist = abs(t - proposed)
            if dist < best_dist:
                best_dist = dist
                best_time = t

    for i, w in enumerate(words):
        # Sentence-terminal word
        if w.word.rstrip()[-1:] in _SENTENCE_ENDERS:
            _consider(w.end)

        # Natural pause after this word
        if i + 1 < len(words):
            gap = words[i + 1].start - w.end
            if gap >= NATURAL_PAUSE_THRESHOLD_SECONDS:
                _consider(w.end)

    return best_time if best_time is not None else proposed


def _char_pos_to_time(char_idx: int, char_positions: list[tuple[int, float]]) -> float | None:
    """Map a character offset in the joined text back to a word start timestamp."""
    result: float | None = None
    for pos, ts in reversed(char_positions):
        if pos <= char_idx:
            result = ts
            break
    return result
