"""Add an instructional to JitsuDB: pick a folder, seed, and enqueue.

Usage:
    python scripts/add_instructional.py [--path /folder] [--no-start]

Without --path, opens a zenity folder picker or falls back to a numbered
terminal list of un-indexed subfolders found in the configured library root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from jitsudb.config import get_config
from jitsudb.db import get_connection, init_db

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
API_BASE = "http://127.0.0.1:8000/api"


def _pick_with_zenity(start_dir: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory",
             "--title=Select instructional folder",
             f"--filename={start_dir}/"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _pick_from_terminal(library: Path) -> Path | None:
    conn = get_connection()
    indexed_paths = {
        row["file_path"]
        for row in conn.execute("SELECT file_path FROM videos WHERE status = 'indexed'").fetchall()
    }

    search_root = library if library.exists() else Path.home()
    candidates: list[Path] = []
    for folder in sorted(search_root.rglob("*")):
        if not folder.is_dir():
            continue
        has_video = any(f.suffix.lower() in VIDEO_EXTENSIONS for f in folder.iterdir() if f.is_file())
        already_done = any(str(folder) in p for p in indexed_paths)
        if has_video and not already_done:
            candidates.append(folder)

    if not candidates:
        print("No un-indexed video folders found. Use --path to specify one manually.")
        return None

    print("\nAvailable instructionals:\n")
    for i, folder in enumerate(candidates, 1):
        try:
            rel = folder.relative_to(search_root)
        except ValueError:
            rel = folder
        print(f"  {i:>3}.  {rel}")

    print()
    raw = input("Enter number (or paste a full path): ").strip()

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
        print("Invalid selection.")
        return None

    p = Path(raw).expanduser().resolve()
    return p if p.is_dir() else None


def pick_folder(library: Path) -> Path | None:
    start = library if library.exists() else Path.home()
    folder = _pick_with_zenity(start)
    if folder is None:
        folder = _pick_from_terminal(library)
    return folder


def _seed_folder(folder: Path, index_path: Path) -> list[str]:
    """Seed videos from folder and return IDs of newly inserted videos."""
    spec = importlib.util.spec_from_file_location(
        "seed_catalog", Path(__file__).parent / "seed_catalog.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.seed(folder, index_path, dry_run=False)

    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title FROM videos WHERE status = 'queued' AND file_path LIKE ?",
        (str(folder) + "%",),
    ).fetchall()
    return [r["id"] for r in rows]


def _api_post(path: str, body: dict) -> bool:
    try:
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except urllib.error.URLError:
        return False


def _enqueue_direct(video_ids: list[str]) -> None:
    conn = get_connection()
    for vid in video_ids:
        conn.execute(
            "INSERT OR IGNORE INTO processing_queue (video_id, priority, status) VALUES (?, 0, 'pending')",
            (vid,),
        )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Add an instructional to the JitsuDB pipeline")
    parser.add_argument("--path", type=Path, default=None, help="Folder to add (skips picker)")
    parser.add_argument("--no-start", action="store_true", help="Enqueue but don't auto-start the pipeline")
    args = parser.parse_args()

    cfg = get_config()
    library = Path(cfg.paths.video_library).expanduser().resolve()
    index_path = library / "Index.md"

    init_db()

    folder = args.path.expanduser().resolve() if args.path else pick_folder(library)
    if folder is None:
        sys.exit(0)

    if not folder.exists():
        logger.error(f"Folder does not exist: {folder}")
        sys.exit(1)

    n_videos = sum(1 for f in folder.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS)
    if n_videos == 0:
        logger.error(f"No video files found in {folder}")
        sys.exit(1)

    print(f"\nFound {n_videos} video(s) in: {folder.name}")

    print("Seeding catalog...")
    new_ids = _seed_folder(folder, index_path)

    if not new_ids:
        print("All videos in this folder are already indexed — nothing to do.")
        sys.exit(0)

    print(f"Seeded {len(new_ids)} new video(s).")

    api_up = _api_post("/queue", {"video_ids": new_ids, "priority": 0})
    if not api_up:
        _enqueue_direct(new_ids)
        print(f"Enqueued {len(new_ids)} video(s) directly (API not running).")
    else:
        print(f"Enqueued {len(new_ids)} video(s) via API.")

    if not args.no_start:
        if api_up:
            _api_post("/process/start", {})
            print("Pipeline started — monitor progress in the browser → Processing Queue.")
        else:
            print("Start the backend and the queue will process automatically:")
            print("  uvicorn jitsudb.api.app:app --host 127.0.0.1 --port 8000 --reload")

    print()


if __name__ == "__main__":
    main()
