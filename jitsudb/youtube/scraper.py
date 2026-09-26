"""YouTube recommendation scraper.

Uses yt-dlp's extract_flat mode to collect video metadata without downloading.
Results are filtered against the youtube_seen DB table to avoid surfacing
already-confirmed or dismissed videos.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from jitsudb.config import YouTubeScraperConfig


def _entry_to_candidate(entry: dict, source: str) -> dict | None:
    """Normalise a yt-dlp flat-playlist entry into a candidate dict."""
    vid_id = entry.get("id")
    if not vid_id:
        return None
    title = entry.get("title") or "Unknown"
    url = entry.get("webpage_url") or entry.get("url") or f"https://www.youtube.com/watch?v={vid_id}"
    duration = float(entry.get("duration") or 0)
    return {
        "yt_video_id":      vid_id,
        "title":            title,
        "uploader":         entry.get("uploader") or entry.get("channel") or "",
        "url":              url,
        "thumbnail_url":    entry.get("thumbnail") or "",
        "duration_seconds": duration,
        "view_count":       entry.get("view_count") or 0,
        "upload_date":      entry.get("upload_date") or "",
        "source":           source,
    }


def _search_tag(tag: str, max_results: int) -> list[dict]:
    """Run a yt-dlp keyword search and return candidate dicts."""
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
    }
    search_url = f"ytsearch{max_results}:{tag}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False) or {}
    except Exception as exc:
        logger.warning(f"yt-dlp search failed for tag {tag!r}: {exc}")
        return []

    candidates = []
    for entry in info.get("entries") or []:
        c = _entry_to_candidate(entry, f"search:{tag}")
        if c:
            candidates.append(c)
    return candidates


def _scrape_channel(channel_url: str, max_results: int) -> list[dict]:
    """Fetch the most recent videos from a channel URL."""
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": max_results,
    }
    # Append /videos to land on the uploads playlist
    url = channel_url.rstrip("/") + "/videos"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception as exc:
        logger.warning(f"yt-dlp channel scrape failed for {channel_url!r}: {exc}")
        return []

    candidates = []
    for entry in info.get("entries") or []:
        c = _entry_to_candidate(entry, f"channel:{channel_url}")
        if c:
            candidates.append(c)
    return candidates


def fetch_recommendations(cfg: YouTubeScraperConfig) -> list[dict]:
    """Scrape tags + channels, deduplicate, filter seen, return candidates.

    This is a blocking call — run it in a background thread.
    Results are NOT filtered by youtube_seen here; the caller should do that
    after obtaining a DB connection (DB connections must not cross threads
    in the same way yt-dlp calls can).
    """
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for tag in cfg.search_tags:
        logger.info(f"Scraper: searching tag {tag!r} ({cfg.max_results_per_tag} results)")
        for c in _search_tag(tag, cfg.max_results_per_tag):
            if c["yt_video_id"] not in seen_ids:
                seen_ids.add(c["yt_video_id"])
                candidates.append(c)

    for channel_url in cfg.reputable_channels:
        logger.info(f"Scraper: scraping channel {channel_url!r} ({cfg.max_per_channel} videos)")
        for c in _scrape_channel(channel_url, cfg.max_per_channel):
            if c["yt_video_id"] not in seen_ids:
                seen_ids.add(c["yt_video_id"])
                candidates.append(c)

    # Filter out videos that are too short to be instructionals
    candidates = [c for c in candidates if c["duration_seconds"] >= cfg.min_duration_seconds]

    logger.info(f"Scraper: {len(candidates)} candidates after dedup + duration filter")
    return candidates


def mark_seen(conn, yt_video_id: str, action: str, title: str = "", channel: str = "") -> None:
    """Insert or replace a video ID in youtube_seen."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO youtube_seen (yt_video_id, action, title, channel, seen_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (yt_video_id, action, title, channel, now),
    )
    conn.commit()


def get_seen_ids(conn) -> set[str]:
    """Return all yt_video_ids from youtube_seen."""
    rows = conn.execute("SELECT yt_video_id FROM youtube_seen").fetchall()
    return {r[0] for r in rows}
