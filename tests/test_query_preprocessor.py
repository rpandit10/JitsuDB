"""Unit tests for jitsudb.search.query_preprocessor."""

from __future__ import annotations

import pytest

from jitsudb.search.query_preprocessor import (
    detect_filters,
    detect_temporal_connectives,
    expand_abbreviations,
    preprocess_query,
)


class TestExpandAbbreviations:
    def test_expands_sc(self):
        assert "side control" in expand_abbreviations("SC escape")

    def test_expands_rnc(self):
        assert "rear naked choke" in expand_abbreviations("RNC from back")

    def test_expands_dlr(self):
        assert "de la riva" in expand_abbreviations("dlr sweeps")

    def test_case_insensitive(self):
        result = expand_abbreviations("Slx entry")
        assert "single leg x" in result.lower()

    def test_no_change_for_unknown_input(self):
        result = expand_abbreviations("armbar from mount")
        assert result == "armbar from mount"

    def test_does_not_expand_partial_match(self):
        # "sc" should not replace inside "escape" or "muscle"
        result = expand_abbreviations("escape from the position")
        assert "side control" not in result


class TestDetectFilters:
    def test_detects_side_control_position(self):
        filters = detect_filters("side control escape")
        assert "side_control" in filters.get("positions", [])

    def test_detects_defense_intent_from_escape(self):
        filters = detect_filters("escaping from half guard")
        assert "defense" in filters.get("intents", [])

    def test_detects_kimura_technique(self):
        filters = detect_filters("kimura from side control")
        assert "kimura" in filters.get("techniques", [])

    def test_returns_empty_on_generic_query(self):
        filters = detect_filters("hello world")
        assert filters.get("positions", []) == []
        assert filters.get("intents", []) == []


class TestPreprocessQuery:
    def test_returns_tuple(self):
        expanded, filters = preprocess_query("SC escape")
        assert isinstance(expanded, str)
        assert isinstance(filters, dict)

    def test_expansion_happens_before_filter_detection(self):
        expanded, filters = preprocess_query("SC escape")
        assert "side control" in expanded.lower()
        assert "side_control" in filters.get("positions", [])


class TestDetectTemporalConnectives:
    def test_before(self):
        assert detect_temporal_connectives("what should I do before the sweep?")

    def test_after(self):
        assert detect_temporal_connectives("after a failed armbar, what next?")

    def test_leads_to(self):
        assert detect_temporal_connectives("kimura leads to the sweep")

    def test_leads_to_singular(self):
        assert detect_temporal_connectives("this lead to the triangle")

    def test_first_then(self):
        assert detect_temporal_connectives("first grip the collar then pull guard")

    def test_from_to(self):
        assert detect_temporal_connectives("transition from side control to mount")

    def test_transition_word(self):
        assert detect_temporal_connectives("transition to back control")

    def test_sequence(self):
        assert detect_temporal_connectives("guard passing sequence")

    def test_setup_finish(self):
        assert detect_temporal_connectives("setup the kimura then finish with an armbar")

    def test_case_insensitive(self):
        assert detect_temporal_connectives("BEFORE the guard pass")

    def test_no_false_positive_simple_query(self):
        assert not detect_temporal_connectives("kimura from side control")

    def test_no_false_positive_generic(self):
        assert not detect_temporal_connectives("how to escape the back mount")
