"""Unit tests for jitsudb.pipeline.chunker."""

from __future__ import annotations

import pytest

from jitsudb.models import WordTimestamp
from jitsudb.pipeline.chunker import chunk_transcript, detect_chapter_breaks, words_to_text


def _make_words(n: int, word_duration: float = 0.5, gap: float = 0.0) -> list[WordTimestamp]:
    """Generate a synthetic word list with uniform timing."""
    words = []
    t = 0.0
    for i in range(n):
        words.append(WordTimestamp(word=f"word{i}", start=t, end=t + word_duration))
        t += word_duration + gap
    return words


class TestWordsToText:
    def test_joins_words_with_spaces(self):
        words = [WordTimestamp(word="hello", start=0, end=0.5),
                 WordTimestamp(word="world", start=0.6, end=1.0)]
        assert words_to_text(words) == "hello world"

    def test_strips_whitespace_from_words(self):
        words = [WordTimestamp(word="  hello  ", start=0, end=0.5)]
        assert words_to_text(words) == "hello"

    def test_empty_list(self):
        assert words_to_text([]) == ""


class TestChunkTranscript:
    def test_short_transcript_produces_one_chunk(self):
        words = _make_words(10, word_duration=1.0)
        chunks = chunk_transcript(words, chunk_duration=45.0, overlap=15.0, snap_window=5.0)
        assert len(chunks) == 1

    def test_long_transcript_produces_multiple_chunks(self):
        # 200 words × 1s each = 200s total → should yield multiple chunks
        words = _make_words(200, word_duration=1.0)
        chunks = chunk_transcript(words, chunk_duration=45.0, overlap=15.0, snap_window=5.0)
        assert len(chunks) > 1

    def test_chunks_have_non_zero_text(self):
        words = _make_words(60, word_duration=1.0)
        chunks = chunk_transcript(words, chunk_duration=45.0, overlap=15.0, snap_window=5.0)
        for chunk in chunks:
            assert chunk.transcript_text.strip() != ""

    def test_chunk_timing_is_ordered(self):
        words = _make_words(100, word_duration=1.0)
        chunks = chunk_transcript(words, chunk_duration=45.0, overlap=15.0, snap_window=5.0)
        for chunk in chunks:
            assert chunk.start_time < chunk.end_time

    def test_empty_transcript_returns_empty_list(self):
        assert chunk_transcript([], 45.0, 15.0, 5.0) == []


class TestDetectChapterBreaks:
    def test_detects_large_gap(self):
        words = _make_words(10, word_duration=0.5)
        # Insert a 5-second gap between word 4 and word 5
        for i in range(5, 10):
            words[i] = WordTimestamp(
                word=words[i].word,
                start=words[i].start + 5.0,
                end=words[i].end + 5.0,
            )
        breaks = detect_chapter_breaks(words, silence_gap=3.0)
        assert len(breaks) >= 1

    def test_no_breaks_in_continuous_speech(self):
        words = _make_words(20, word_duration=0.5, gap=0.05)
        breaks = detect_chapter_breaks(words, silence_gap=3.0)
        assert breaks == []
