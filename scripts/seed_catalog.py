"""Scan a video library and upsert discovered videos into SQLite.

Usage:
    python scripts/seed_catalog.py [--dry-run] [--path /some/folder]

Reads Index.md from the library root (or the scan path) to get canonical
instructor names, series titles, and gi/nogi classification. Falls back to
path inference for entries not found in the index.

Expected folder structure:
    <library>/<NN-Category>/<Instructor - Series Title>/<part>.mp4
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from jitsudb.config import get_config
from jitsudb.db import get_connection, init_db
from jitsudb.models import Video

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def _normalize(text: str) -> str:
    """Lowercase, keep only alphanumeric + spaces, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def parse_index(index_path: Path) -> list[dict]:
    """Parse the markdown table in Index.md into a list of entry dicts.

    Each entry has: instructor, series_title, category, gi_nogi, tags, key.
    ``key`` is a normalised string used for fuzzy folder matching.
    """
    if not index_path.exists():
        logger.warning(f"Index.md not found at {index_path} — falling back to path inference only")
        return []

    entries: list[dict] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or not cells[0].isdigit():
            continue

        instructor   = cells[1].strip()
        series_title = cells[2].strip()
        primary_folder = cells[3].strip()
        tags = [t.strip() for t in cells[4].split(",") if t.strip()]

        # Strip leading "NN-" from category name
        category = re.sub(r"^\d+-", "", primary_folder.split("/")[0]).strip()

        gi_nogi = "gi" if primary_folder.startswith("07-Gi") or "Gi" in tags else "nogi"

        entries.append({
            "instructor":   instructor,
            "series_title": series_title,
            "category":     category,
            "gi_nogi":      gi_nogi,
            "tags":         tags,
            # Normalised "instructor title" used for fuzzy folder matching
            "key": _normalize(f"{instructor} {series_title}"),
        })

    logger.info(f"Loaded {len(entries)} entries from {index_path.name}")
    return entries


def find_index_entry(folder_name: str, entries: list[dict], threshold: float = 0.60) -> dict | None:
    """Return the best-matching Index.md entry for a folder name, or None."""
    if not entries:
        return None
    norm = _normalize(folder_name)
    best_score = 0.0
    best_entry = None
    for entry in entries:
        score = difflib.SequenceMatcher(None, norm, entry["key"]).ratio()
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_score >= threshold:
        logger.debug(f"Matched {folder_name!r} → {best_entry['series_title']!r} (score={best_score:.2f})")
        return best_entry
    logger.debug(f"No index match for {folder_name!r} (best score={best_score:.2f})")
    return None


def _extract_part(stem: str) -> str:
    """Return a clean part suffix like ' — Part 3', or '' if none found."""
    m = re.search(r"[Pp]art[\s_]?(\d+)", stem)
    return f" — Part {m.group(1)}" if m else ""


def infer_metadata(file_path: Path, scan_root: Path, entries: list[dict]) -> dict:
    """Derive title, instructor, category, and gi_nogi for a video file.

    Tries Index.md fuzzy lookup first; falls back to path-based inference.
    """
    parts = file_path.relative_to(scan_root).parts
    dirs = parts[:-1]

    # When --path points directly at a series folder the file sits at the root
    # of scan_root with no sub-directories.  Fall back to scan_root itself.
    if not dirs:
        parent_dir  = scan_root.name
        category_raw = scan_root.parent.name
    else:
        parent_dir   = dirs[-1]
        category_raw = dirs[0]

    category = re.sub(r"^\d+-", "", category_raw.split("/")[0]).strip()

    # Try Index.md lookup
    entry = find_index_entry(parent_dir, entries)
    if entry:
        instructor   = entry["instructor"]
        series_title = entry["series_title"]
        gi_nogi      = entry["gi_nogi"]
        category     = entry["category"]
        title        = series_title + _extract_part(file_path.stem)
        return {
            "title":            title,
            "instructor":       instructor,
            "primary_category": category,
            "gi_nogi":          gi_nogi,
        }

    # Path-based fallback
    # Extract instructor from "Instructor - Title" pattern in parent folder
    if " - " in parent_dir:
        instructor = parent_dir.split(" - ", 1)[0].strip()
    elif len(dirs) >= 2:
        # Try grandparent for deeply nested structures (e.g. Submeta collection)
        grandparent = dirs[-2]
        if " - " in grandparent:
            instructor = grandparent.split(" - ", 1)[0].strip()
        else:
            instructor = parent_dir
    else:
        instructor = parent_dir or "Unknown"

    # gi if in the Gi category
    gi_nogi = "gi" if "gi" in category.lower() else "nogi"

    title = file_path.stem.replace("_", " ").replace("-", " ").strip()

    return {
        "title":            title,
        "instructor":       instructor,
        "primary_category": category,
        "gi_nogi":          gi_nogi,
    }


def seed(scan_path: Path, index_path: Path, dry_run: bool = False) -> None:
    if not scan_path.exists():
        logger.error(f"Path does not exist: {scan_path}")
        sys.exit(1)

    entries = parse_index(index_path)

    if not dry_run:
        init_db()
    conn = get_connection() if not dry_run else None

    found = inserted = 0

    for video_path in sorted(scan_path.rglob("*")):
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        found += 1

        meta = infer_metadata(video_path, scan_path, entries)

        if dry_run:
            logger.info(
                f"[dry-run] {video_path.parent.name}/{video_path.name}\n"
                f"          instructor={meta['instructor']!r}  "
                f"gi_nogi={meta['gi_nogi']!r}  "
                f"category={meta['primary_category']!r}\n"
                f"          title={meta['title']!r}"
            )
            continue

        existing = conn.execute(
            "SELECT id FROM videos WHERE file_path = ?", (str(video_path),)
        ).fetchone()
        if existing:
            logger.debug(f"Skipping (already indexed): {video_path.name}")
            continue

        video = Video(
            title=meta["title"],
            instructor=meta["instructor"],
            file_path=video_path,
            duration_seconds=0.0,
            gi_nogi=meta["gi_nogi"],
            primary_category=meta["primary_category"],
            status="queued",
        )

        conn.execute(
            """INSERT INTO videos
               (id, title, instructor, file_path, duration_seconds, gi_nogi,
                primary_category, status, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                video.id, video.title, video.instructor, str(video.file_path),
                video.duration_seconds, video.gi_nogi, video.primary_category,
                video.status, video.added_at.isoformat(),
            ),
        )
        conn.commit()
        inserted += 1
        logger.info(f"Inserted: [{meta['instructor']}] {meta['title']}")

    logger.info(f"Done. Found {found} video file(s); inserted {inserted} new record(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the JitsuDB catalog from a video folder")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Folder to scan. Defaults to config video_library.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to Index.md. Defaults to <scan-path>/Index.md.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.path:
        target = args.path.expanduser().resolve()
    else:
        cfg = get_config()
        target = Path(cfg.paths.video_library).expanduser().resolve()

    index_file = args.index.expanduser().resolve() if args.index else target / "Index.md"

    seed(target, index_file, dry_run=args.dry_run)
