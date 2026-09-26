"""Unit tests for jitsudb.pipeline.classifier."""

from __future__ import annotations

import pytest

from jitsudb.pipeline.classifier import infer_context, normalise_text, scan_taxonomy
from jitsudb.taxonomy.dictionary import INTENTS, POSITIONS, TECHNIQUES


class TestNormaliseText:
    def test_lowercases(self):
        assert normalise_text("Side Control") == "side control"

    def test_strips_punctuation(self):
        assert normalise_text("armbar!") == "armbar"

    def test_collapses_whitespace(self):
        assert normalise_text("half   guard") == "half guard"


class TestScanTaxonomy:
    def test_detects_side_control(self):
        text = "you need to escape from side control"
        labels = scan_taxonomy(text, POSITIONS)
        assert "side_control" in labels

    def test_detects_multiple_positions(self):
        text = "from half guard transition to mount"
        labels = scan_taxonomy(text, POSITIONS)
        assert "half_guard" in labels
        assert "mount" in labels

    def test_no_false_positives_on_unrelated_text(self):
        text = "the weather is nice today"
        labels = scan_taxonomy(text, POSITIONS)
        assert labels == []

    def test_detects_kimura_in_techniques(self):
        labels = scan_taxonomy("grab the kimura grip", TECHNIQUES)
        assert "kimura" in labels

    @pytest.mark.parametrize("text,expected_intent", [
        ("you need to escape", "defense"),
        ("attack and submit", "attack"),
        ("sweep to come on top", "sweep"),
    ])
    def test_detects_intent(self, text, expected_intent):
        labels = scan_taxonomy(text, INTENTS)
        assert expected_intent in labels


class TestInferContext:
    def test_escape_plus_position_is_bottom(self):
        ctx = infer_context(
            "escape from side control",
            detected_positions=["side_control"],
            detected_intents=["defense"],
            primary_category="Escapes",
        )
        assert ctx == "bottom"

    def test_attack_plus_position_is_top(self):
        ctx = infer_context(
            "attack from mount",
            detected_positions=["mount"],
            detected_intents=["attack"],
            primary_category="Submissions",
        )
        assert ctx == "top"

    def test_standing_context(self):
        ctx = infer_context(
            "takedown from standing",
            detected_positions=["standing"],
            detected_intents=["takedown"],
            primary_category="Takedowns",
        )
        assert ctx == "standing"

    def test_category_hint_bottom(self):
        ctx = infer_context(
            "some neutral text",
            detected_positions=[],
            detected_intents=[],
            primary_category="Escapes-Defense",
        )
        assert ctx == "bottom"
