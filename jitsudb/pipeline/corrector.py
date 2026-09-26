"""Post-transcription correction pass — fixes known Whisper BJJ mistranscriptions."""

from __future__ import annotations

import re

from jitsudb.models import WordTimestamp
from jitsudb.taxonomy.corrections import CORRECTIONS


def apply_corrections(words: list[WordTimestamp]) -> list[WordTimestamp]:
    """Run the CORRECTIONS dictionary over a word-level transcript.

    Scans the word list sequentially and replaces consecutive words that match
    a correction phrase with the corrected words. Timestamps for replaced words
    are distributed evenly across the original span.
    """
    if not words:
        return []

    result = list(words)

    for wrong, right in CORRECTIONS.items():
        wrong_parts = wrong.lower().split()
        right_parts = right.split()
        n = len(wrong_parts)

        new_result: list[WordTimestamp] = []
        i = 0
        while i < len(result):
            if (
                i + n <= len(result)
                and all(
                    result[i + j].word.strip().lower() == wrong_parts[j]
                    for j in range(n)
                )
            ):
                # Distribute the original span evenly across the replacement words
                span_start = result[i].start
                span_end = result[i + n - 1].end
                span = span_end - span_start
                m = len(right_parts)
                for k, rword in enumerate(right_parts):
                    new_result.append(
                        WordTimestamp(
                            word=rword,
                            start=span_start + span * k / m,
                            end=span_start + span * (k + 1) / m,
                        )
                    )
                i += n
            else:
                new_result.append(result[i])
                i += 1

        result = new_result

    return result


def correct_text(text: str) -> str:
    """Apply CORRECTIONS to a plain text string (used for testing and snippets).

    Replacements are case-insensitive and phrase-matched (not word-boundary
    constrained, so suitable for already-joined transcript text).
    """
    result = text
    for wrong, right in CORRECTIONS.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        result = pattern.sub(right, result)
    return result
