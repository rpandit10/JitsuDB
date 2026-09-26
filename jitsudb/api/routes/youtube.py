"""YouTube routes: manual download and recommendation scraping."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from jitsudb.api.dependencies import get_db

router = APIRouter(tags=["youtube"])

# recommendation cache
_rec_cache: list[dict] = []
_rec_cache_at: float = 0.0
_rec_lock = threading.Lock()

# In-memory download job tracker
_downloads: dict[str, dict] = {}
_dl_lock = threading.Lock()


class YouTubeRequest(BaseModel):
    url: str
    gi_nogi: Literal["gi", "nogi", "both"] = "nogi"
    instructor: str = ""
    primary_category: str = "Uncategorised"


class YouTubeSubmitResponse(BaseModel):
    download_id: str
    status: str
    message: str


@router.post("/youtube", response_model=YouTubeSubmitResponse)
async def submit_youtube(body: YouTubeRequest) -> YouTubeSubmitResponse:
    """Accept a YouTube URL, start background download, and auto-enqueue for pipeline."""
    if not body.url.strip():
        raise HTTPException(status_code=422, detail="url must not be empty")

    download_id = str(uuid.uuid4())
    with _dl_lock:
        _downloads[download_id] = {
            "download_id": download_id,
            "status": "starting",
            "url": body.url,
            "title": None,
            "video_id": None,
            "progress_pct": 0,
            "error": None,
        }

    thread = threading.Thread(
        target=_download_worker,
        args=(download_id, body),
        daemon=True,
        name=f"yt-dl-{download_id[:8]}",
    )
    thread.start()

    return YouTubeSubmitResponse(
        download_id=download_id,
        status="starting",
        message="Download started in background.",
    )


class RecommendationConfirmRequest(BaseModel):
    yt_video_id: str
    url: str
    title: str = ""
    uploader: str = ""
    gi_nogi: Literal["gi", "nogi", "both"] = "nogi"
    instructor: str = ""
    primary_category: str = "Uncategorised"


class RecommendationDismissRequest(BaseModel):
    yt_video_id: str
    title: str = ""
    channel: str = ""


@router.get("/youtube/recommendations")
async def get_recommendations(refresh: bool = False) -> list[dict]:
    """Return unseen YouTube video candidates from the configured scraper.

    Results are cached for cache_ttl_seconds; pass ?refresh=true to force
    a new yt-dlp fetch.
    """
    from jitsudb.config import get_config
    from jitsudb.db import get_connection
    from jitsudb.youtube.scraper import fetch_recommendations, get_seen_ids

    cfg = get_config()
    scraper_cfg = cfg.youtube.scraper
    ttl = scraper_cfg.cache_ttl_seconds

    global _rec_cache, _rec_cache_at

    with _rec_lock:
        cache_stale = (time.time() - _rec_cache_at) > ttl
        if not refresh and not cache_stale and _rec_cache:
            cached = list(_rec_cache)
        else:
            cached = None

    if cached is not None:
        # Filter against current seen table and return
        conn = get_connection()
        seen = get_seen_ids(conn)
        return [c for c in cached if c["yt_video_id"] not in seen]

    # Blocking scrape — runs synchronously here but the route is async so
    # FastAPI will run it in a threadpool if called with run_in_threadpool.
    # For simplicity (yt-dlp is already its own I/O), run inline.
    logger.info("Recommendations: running scraper (cache miss or forced refresh)")
    candidates = fetch_recommendations(scraper_cfg)

    with _rec_lock:
        _rec_cache = candidates
        _rec_cache_at = time.time()

    conn = get_connection()
    seen = get_seen_ids(conn)
    return [c for c in candidates if c["yt_video_id"] not in seen]


@router.post("/youtube/recommendations/confirm")
async def confirm_recommendation(body: RecommendationConfirmRequest) -> dict:
    """Mark a recommended video as confirmed and enqueue it for download."""
    from jitsudb.db import get_connection
    from jitsudb.youtube.scraper import mark_seen

    conn = get_connection()
    mark_seen(conn, body.yt_video_id, "confirmed", body.title, body.uploader)

    # Kick off the same background download used by manual ingest
    download_id = str(uuid.uuid4())
    with _dl_lock:
        _downloads[download_id] = {
            "download_id": download_id,
            "status": "starting",
            "url": body.url,
            "title": body.title or None,
            "video_id": None,
            "progress_pct": 0,
            "error": None,
        }

    dl_body = YouTubeRequest(
        url=body.url,
        gi_nogi=body.gi_nogi,
        instructor=body.instructor,
        primary_category=body.primary_category,
    )
    thread = threading.Thread(
        target=_download_worker,
        args=(download_id, dl_body),
        daemon=True,
        name=f"yt-rec-{download_id[:8]}",
    )
    thread.start()
    return {"download_id": download_id, "status": "starting", "message": "Download queued."}


@router.post("/youtube/recommendations/dismiss")
async def dismiss_recommendation(body: RecommendationDismissRequest) -> dict:
    """Mark a recommended video as dismissed so it won't appear again."""
    from jitsudb.db import get_connection
    from jitsudb.youtube.scraper import mark_seen

    conn = get_connection()
    mark_seen(conn, body.yt_video_id, "dismissed", body.title, body.channel)
    return {"ok": True}


# ── per-job endpoints (keep after /recommendations/* to avoid path collision) ─

@router.get("/youtube/{download_id}")
async def get_download_status(download_id: str) -> dict:
    """Poll status of a background YouTube download job."""
    with _dl_lock:
        job = _downloads.get(download_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found")
    return dict(job)


@router.get("/youtube")
async def list_downloads() -> list[dict]:
    """Return all tracked YouTube download jobs (most recent first)."""
    with _dl_lock:
        return list(reversed(list(_downloads.values())))


def _set(download_id: str, patch: dict) -> None:
    with _dl_lock:
        _downloads[download_id].update(patch)


def _download_worker(download_id: str, body: YouTubeRequest) -> None:
    from jitsudb.config import get_config
    from jitsudb.db import get_connection
    from jitsudb.youtube.downloader import download_video, extract_metadata

    try:
        from jitsudb.youtube.scraper import mark_seen

        cfg = get_config()

        _set(download_id, {"status": "fetching_metadata"})
        meta = extract_metadata(body.url)
        title = meta["title"]
        uploader = meta.get("uploader", "")
        duration = meta["duration_seconds"]
        yt_vid_id = meta.get("video_id", "")
        logger.info(f"YouTube [{download_id[:8]}] metadata: {title!r} ({duration:.0f}s)")
        _set(download_id, {"status": "downloading", "title": title})

        def _progress(d: dict) -> None:
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
                downloaded = d.get("downloaded_bytes", 0)
                pct = int(min(100, downloaded / total * 100))
                _set(download_id, {"progress_pct": pct})

        download_dir = cfg.paths.data_dir / "youtube_downloads"
        video_path = download_video(
            body.url, download_dir, cfg.youtube.default_format, progress_hook=_progress
        )
        logger.info(f"YouTube [{download_id[:8]}] downloaded to {video_path}")

        _set(download_id, {"status": "registering", "progress_pct": 100})

        instructor = body.instructor.strip() or uploader or "Unknown"
        video_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = get_connection()
        conn.execute(
            """
            INSERT OR IGNORE INTO videos
              (id, title, instructor, file_path, duration_seconds, gi_nogi, primary_category, status, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?)
            """,
            (
                video_id,
                title,
                instructor,
                str(video_path),
                duration,
                body.gi_nogi,
                body.primary_category,
                now,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO processing_queue (video_id, priority, status) VALUES (?, 0, 'pending')",
            (video_id,),
        )
        conn.commit()

        logger.info(f"YouTube [{download_id[:8]}] registered video_id={video_id}")

        # Mark in youtube_seen so the recommendation scraper won't resurface it
        if yt_vid_id:
            mark_seen(conn, yt_vid_id, "confirmed", title, uploader)

        _set(download_id, {"status": "complete", "video_id": video_id})

    except Exception as exc:
        logger.exception(f"YouTube download {download_id} failed: {exc}")
        _set(download_id, {"status": "error", "error": str(exc)})
