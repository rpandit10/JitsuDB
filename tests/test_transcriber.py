"""Unit tests for jitsudb.pipeline.transcriber."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jitsudb.pipeline.transcriber import load_glossary, transcribe


class TestLoadGlossary:
    def test_returns_content_when_file_exists(self, tmp_path: Path):
        glossary = tmp_path / "glossary.txt"
        glossary.write_text("kimura, armbar, guillotine")
        result = load_glossary(glossary)
        assert "kimura" in result

    def test_returns_empty_string_when_missing(self, tmp_path: Path):
        result = load_glossary(tmp_path / "nonexistent.txt")
        assert result == ""


class TestTranscribe:
    def test_returns_word_timestamps(self, tmp_path: Path):
        wav = tmp_path / "audio.wav"
        wav.touch()

        mock_model = MagicMock()
        # faster-whisper returns (segments_generator, info)
        mock_segment = MagicMock()
        mock_segment.words = [
            MagicMock(word="hello", start=0.0, end=0.4),
            MagicMock(word="world", start=0.5, end=0.9),
        ]
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        result = transcribe(wav, mock_model, glossary="", language="en")
        assert len(result) == 2
        assert result[0].word == "hello"
        assert result[1].word == "world"

    def test_raises_if_wav_missing(self):
        mock_model = MagicMock()
        with pytest.raises(FileNotFoundError):
            transcribe(Path("/nonexistent/audio.wav"), mock_model)
