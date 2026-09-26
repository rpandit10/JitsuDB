"""Query preprocessing — abbreviation expansion and auto-filter detection."""

from __future__ import annotations

import re

from jitsudb.taxonomy.dictionary import INTENTS, POSITIONS, TECHNIQUES

TEMPORAL_PATTERNS: list[str] = [
    r"\bbefore\b",
    r"\bafter\b",
    r"\bleads?\s+to\b",
    r"\bfirst\b.{1,50}\bthen\b",
    r"\bfrom\b.{1,50}\bto\b",
    r"\btransition",
    r"\bsequence\b",
    r"\bsetup\b.{1,50}\bfinish\b",
]

ABBREVIATIONS: dict[str, str] = {
    "sc": "side control",
    "hg": "half guard",
    "cg": "closed guard",
    "og": "open guard",
    "dlr": "de la riva",
    "rdlr": "reverse de la riva",
    "slx": "single leg x",
    "bg": "butterfly guard",
    "bc": "back control",
    "kob": "knee on belly",
    "ns": "north south",
    "rnc": "rear naked choke",
    "tkd": "takedown",
    "bjj": "jiu-jitsu",
}


def expand_abbreviations(query: str) -> str:
    """Replace known BJJ abbreviations with their full forms.

    Replacement is case-insensitive and whole-word only so "SC" in the
    middle of a word is not replaced.
    """
    result = query
    for abbr, expansion in ABBREVIATIONS.items():
        pattern = re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE)
        result = pattern.sub(expansion, result)
    return result


def detect_filters(query: str) -> dict[str, list[str]]:
    """Scan the expanded query for taxonomy terms and return detected filters.

    Runs the same normalise + scan logic as the classifier against POSITIONS,
    INTENTS, and TECHNIQUES so the UI can show auto-detected filter chips.
    """
    from jitsudb.pipeline.classifier import normalise_text, scan_taxonomy

    norm = normalise_text(query)
    return {
        "positions": scan_taxonomy(norm, POSITIONS),
        "intents": scan_taxonomy(norm, INTENTS),
        "techniques": scan_taxonomy(norm, TECHNIQUES),
    }


def detect_temporal_connectives(query: str) -> bool:
    """Return True if the query contains temporal connectives suggesting multi-hop retrieval.

    Matches phrases like "before", "after", "leads to", "first … then",
    "from X to Y", "transition", "sequence", and "setup … finish".
    """
    lowered = query.lower()
    for pattern in TEMPORAL_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


def preprocess_query(query: str) -> tuple[str, dict[str, list[str]]]:
    """Expand abbreviations, detect filters, and return both to the search engine."""
    expanded = expand_abbreviations(query)
    filters = detect_filters(expanded)
    return expanded, filters
