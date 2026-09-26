"""Taxonomy classifier — keyword scan assigns structured BJJ labels to each chunk."""

from __future__ import annotations

import re

from jitsudb.models import Chunk, RawChunk, Video
from jitsudb.taxonomy.dictionary import (
    ATTACK_CATEGORY_HINTS,
    CONTEXTS,
    DEFENSE_CATEGORY_HINTS,
    INTENTS,
    POSITIONS,
    TECHNIQUES,
)


def classify_chunk(raw: RawChunk, video: Video) -> Chunk:
    """Assign taxonomy labels to a raw chunk and return a fully classified Chunk."""
    norm = normalise_text(raw.transcript_text)

    positions = scan_taxonomy(norm, POSITIONS)
    intents = scan_taxonomy(norm, INTENTS)
    techniques = scan_taxonomy(norm, TECHNIQUES)
    context = infer_context(norm, positions, intents, video.primary_category)

    return Chunk(
        video_id=video.id,
        start_time=raw.start_time,
        end_time=raw.end_time,
        transcript_text=raw.transcript_text,
        position=positions,
        context=context,
        intent=intents,
        techniques=techniques,
        gi_nogi=video.gi_nogi,
    )


def normalise_text(text: str) -> str:
    """Lowercase and strip punctuation from transcript text for keyword matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def scan_taxonomy(normalised_text: str, taxonomy: dict[str, list[str]]) -> list[str]:
    """Return all taxonomy keys whose term lists appear in the normalised text.

    Each term in the value list is searched as a substring of the normalised text.
    Terms with spaces are matched as phrases; single-word terms use word boundaries
    to avoid false positives (e.g. "mount" inside "amount").
    """
    matches: list[str] = []

    for key, terms in taxonomy.items():
        for term in terms:
            norm_term = term.lower()
            if " " in norm_term:
                # Multi-word phrase: substring search
                if norm_term in normalised_text:
                    matches.append(key)
                    break
            else:
                # Single word: require word boundaries to avoid partial matches
                pattern = r"\b" + re.escape(norm_term) + r"\b"
                if re.search(pattern, normalised_text):
                    matches.append(key)
                    break

    return matches


def infer_context(
    normalised_text: str,
    detected_positions: list[str],
    detected_intents: list[str],
    primary_category: str,
) -> str | None:
    """Determine top/bottom/neutral/standing context from co-occurrence signals.

    Scoring approach:
    - Standing intents (takedown) or positions → "standing"
    - Defensive intents (defense, sweep) + any position → "bottom"
    - Offensive intents (attack, pass, control) + any position → "top"
    - Ties or no positional context → check primary_category hints
    - Still ambiguous → "neutral"
    """
    # Standing is unambiguous
    if "standing" in detected_positions or "takedown" in detected_intents:
        return "standing"

    # Score bottom vs top signals from intents
    bottom_intents = {"defense", "sweep"}
    top_intents = {"attack", "pass", "control"}

    bottom_score = sum(1 for i in detected_intents if i in bottom_intents)
    top_score = sum(1 for i in detected_intents if i in top_intents)

    # Also check raw text for context keywords from CONTEXTS dict
    context_norm = normalise_text(" ".join(CONTEXTS.get("bottom", [])))
    for kw in CONTEXTS.get("bottom", []):
        if f" {kw.lower()} " in f" {normalised_text} ":
            bottom_score += 1
            break

    for kw in CONTEXTS.get("top", []):
        if f" {kw.lower()} " in f" {normalised_text} ":
            top_score += 1
            break

    if bottom_score > top_score and detected_positions:
        return "bottom"
    if top_score > bottom_score and detected_positions:
        return "top"

    # Fall back to video category hints
    cat_lower = primary_category.lower()
    for hint in DEFENSE_CATEGORY_HINTS:
        if hint in cat_lower:
            return "bottom"
    for hint in ATTACK_CATEGORY_HINTS:
        if hint in cat_lower:
            return "top"

    # Genuinely ambiguous
    return "neutral"
