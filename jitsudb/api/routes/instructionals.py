"""GET /api/instructionals/available, POST /api/instructionals/seed."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jitsudb.api.dependencies import get_db
from jitsudb.config import get_config
from jitsudb.db import get_connection, init_db

router = APIRouter(tags=["instructionals"])

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def _load_seed_module():
    scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
    spec = importlib.util.spec_from_file_location("seed_catalog", scripts_dir / "seed_catalog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _video_count(folder: Path) -> int:
    return sum(1 for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS)


@router.get("/instructionals/available")
def list_available_folders() -> list[dict[str, Any]]:
    """Return un-indexed subfolders in the configured library root."""
    cfg = get_config()
    library = Path(cfg.paths.video_library).expanduser().resolve()

    if not library.exists():
        return []

    conn = get_connection()
    indexed_paths: set[str] = {
        row["file_path"]
        for row in conn.execute("SELECT file_path FROM videos").fetchall()
    }

    results: list[dict] = []
    for folder in sorted(library.rglob("*")):
        if not folder.is_dir():
            continue
        has_video = any(
            f.suffix.lower() in VIDEO_EXTENSIONS
            for f in folder.iterdir()
            if f.is_file()
        )
        if not has_video:
            continue
        already_indexed = any(str(folder) in p for p in indexed_paths)
        if already_indexed:
            continue
        try:
            rel = str(folder.relative_to(library))
        except ValueError:
            rel = str(folder)
        results.append({
            "path": str(folder),
            "name": folder.name,
            "relative_path": rel,
            "video_count": _video_count(folder),
        })

    return results


class SeedRequest(BaseModel):
    folder_path: str
    auto_start: bool = True


class SeedResponse(BaseModel):
    seeded: int
    already_indexed: int
    video_ids: list[str]
    message: str


@router.post("/instructionals/seed", response_model=SeedResponse)
def seed_folder(req: SeedRequest) -> SeedResponse:
    """Seed a folder of videos into the DB and enqueue them for processing."""
    folder = Path(req.folder_path).expanduser().resolve()

    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder}")

    n_videos = _video_count(folder)
    if n_videos == 0:
        raise HTTPException(status_code=400, detail="No video files found in folder")

    seed_mod = _load_seed_module()
    cfg = get_config()
    library = Path(cfg.paths.video_library).expanduser().resolve()
    index_path = library / "Index.md"

    init_db()
    conn = get_connection()

    before_ids = {
        row["id"]
        for row in conn.execute("SELECT id FROM videos WHERE status = 'queued'").fetchall()
    }

    seed_mod.seed(folder, index_path, dry_run=False)

    after_rows = conn.execute(
        "SELECT id FROM videos WHERE status = 'queued' AND file_path LIKE ?",
        (str(folder) + "%",),
    ).fetchall()
    new_ids = [r["id"] for r in after_rows if r["id"] not in before_ids]
    already_indexed = n_videos - len(new_ids)

    if new_ids:
        for vid in new_ids:
            conn.execute(
                "INSERT OR IGNORE INTO processing_queue (video_id, priority, status) VALUES (?, 0, 'pending')",
                (vid,),
            )
        conn.commit()

        if req.auto_start:
            from jitsudb.api.dependencies import start_pipeline
            start_pipeline()

    return SeedResponse(
        seeded=len(new_ids),
        already_indexed=already_indexed,
        video_ids=new_ids,
        message=(
            f"Seeded {len(new_ids)} video(s) and queued for processing."
            if new_ids
            else "All videos in this folder are already indexed."
        ),
    )
