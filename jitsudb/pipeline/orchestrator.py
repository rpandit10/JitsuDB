"""Pipeline orchestrator — chains all processing stages with resume support."""

from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from jitsudb.config import get_config
from jitsudb.constants import (
    MIN_SILENCE_GAP_SECONDS,
)
from jitsudb.db import get_connection
from jitsudb.models import Chunk, PipelineProgress, QueueItem, RawChunk, Video, WordTimestamp
from jitsudb.pipeline import audio, classifier, corrector, embedder, thumbnails, transcriber
from jitsudb.pipeline.chunker import chunk_transcript, detect_chapter_breaks

if TYPE_CHECKING:
    import chromadb
    from faster_whisper import WhisperModel
    from sentence_transformers import SentenceTransformer

# Ordered list of pipeline stages — order is significant for resume logic
PIPELINE_STAGES = [
    "validate",
    "audio",
    "transcribe",
    "correct",
    "chunk",
    "classify",
    "embed",
    "thumbnail",
    "visual_embed",
    "graph_edges",
]


class PipelineOrchestrator:
    """Runs the full processing pipeline for queued videos.

    Loads the queue from SQLite, processes one video at a time (to stay within
    VRAM limits), and tracks per-video stage completion so the pipeline can
    resume after a crash or user pause.

    Heavy models (Whisper, sentence-transformer) are loaded once in ``run()``
    and reused across all videos. They are NOT reloaded per video.
    """

    def __init__(
        self,
        dry_run: bool = False,
        progress_state: dict | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> None:
        self.dry_run = dry_run
        self._progress: dict = progress_state if progress_state is not None else {}
        self._stop = stop_event or threading.Event()
        self._pause = pause_event or threading.Event()
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._video_start_time: float | None = None

    def run(self, poll_interval: float = 5.0) -> None:
        """Process queued videos continuously, idling between batches.

        Loads heavy models once on first real batch of work, then keeps them
        in memory across idle periods.  Polls SQLite every ``poll_interval``
        seconds when the queue is empty so newly-added videos are picked up
        automatically.
        """
        cfg = get_config()

        if self.dry_run:
            logger.info("[dry-run] Pipeline would process the following queue:")
            for item, vid in self._load_queue():
                logger.info(
                    f"  [dry-run] {vid.title} "
                    f"(priority={item.priority}, last_stage={item.stage_completed})"
                )
            return

        # Models are loaded lazily on the first batch and kept for reuse.
        models_loaded = False
        whisper_model = None
        glossary: str = ""
        embed_model = None
        chroma_collection = None
        clip_model = None
        visual_collection = None

        chroma_dir = cfg.paths.data_dir / "chromadb"

        self._update_progress(PipelineProgress(is_running=True, is_idle=True, overall_progress=1.0))

        while not self._stop.is_set():
            queue = self._load_queue()

            if not queue:
                self._update_progress(
                    PipelineProgress(is_running=True, is_idle=True, overall_progress=1.0)
                )
                self._stop.wait(timeout=poll_interval)
                continue

            # Load models on first encounter of real work.
            if not models_loaded:
                try:
                    logger.info("Loading Whisper model …")
                    whisper_model = transcriber.load_model(
                        cfg.transcription.model_size,
                        cfg.transcription.compute_type,
                    )
                    glossary = transcriber.load_glossary(cfg.transcription.glossary_path)

                    logger.info("Loading embedding model …")
                    embed_model = embedder.load_embedding_model(
                        model_name=cfg.embedding.model_name,
                        device=cfg.embedding.device,
                        model_path=cfg.embedding.encoder_path,
                        trust_remote_code=cfg.embedding.trust_remote_code,
                    )
                    chroma_collection = embedder.get_chroma_collection(
                        chroma_dir, cfg.embedding.collection_name
                    )

                    if cfg.visual_embedding.enabled:
                        from jitsudb.pipeline.visual_embedder import get_visual_collection, load_clip_model
                        logger.info("Loading CLIP model …")
                        clip_model = load_clip_model(
                            cfg.visual_embedding.model_name, cfg.visual_embedding.device
                        )
                        visual_collection = get_visual_collection(chroma_dir)

                    models_loaded = True
                except Exception as exc:
                    logger.error(
                        f"Model loading failed: {exc}\n{traceback.format_exc()}"
                        f"\nRetrying in {poll_interval}s …"
                    )
                    self._stop.wait(timeout=poll_interval)
                    continue

            total = len(queue)
            logger.info(f"Processing queue: {total} video(s)")

            self._start_time = time.monotonic()
            self._update_progress(PipelineProgress(is_running=True, is_idle=False, overall_progress=0.0))

            for idx, (queue_item, video) in enumerate(queue):
                if self._stop.is_set():
                    logger.info("Stop requested — halting pipeline")
                    break

                while self._pause.is_set():
                    self._update_progress(
                        PipelineProgress(
                            is_running=True,
                            is_paused=True,
                            video_title=video.title,
                            overall_progress=idx / total,
                        )
                    )
                    self._pause.wait(timeout=1.0)

                overall = idx / total
                self._update_progress(
                    PipelineProgress(
                        is_running=True,
                        video_id=video.id,
                        video_title=video.title,
                        overall_progress=overall,
                    )
                )

                try:
                    self._process_video(
                        video,
                        queue_item,
                        whisper_model,
                        glossary,
                        embed_model,
                        chroma_collection,
                        clip_model=clip_model,
                        visual_collection=visual_collection,
                        overall_base=overall,
                        overall_share=1.0 / total,
                    )
                except Exception as exc:
                    logger.error(
                        f"Pipeline failed on '{video.title}': {exc}\n{traceback.format_exc()}"
                    )
                    self._mark_failed(video.id, str(exc))

            if self._stop.is_set():
                break

        self._update_progress(PipelineProgress(is_running=False, overall_progress=1.0))
        logger.info("Pipeline stopped")

    def _process_video(
        self,
        video: Video,
        queue_item: QueueItem,
        whisper_model: "WhisperModel",
        glossary: str,
        embed_model: "SentenceTransformer",
        chroma_collection: "chromadb.Collection",
        clip_model: "SentenceTransformer | None" = None,
        visual_collection: "chromadb.Collection | None" = None,
        overall_base: float = 0.0,
        overall_share: float = 1.0,
    ) -> None:
        """Run all pipeline stages for a single video, skipping completed ones."""
        cfg = get_config()
        conn = get_connection()
        last_completed = queue_item.stage_completed

        def stage_done(stage: str) -> bool:
            if last_completed is None:
                return False
            return PIPELINE_STAGES.index(stage) <= PIPELINE_STAGES.index(last_completed)

        def update(stage: str, stage_pct: float) -> None:
            stage_idx = PIPELINE_STAGES.index(stage)
            stage_fraction = (stage_idx + stage_pct) / len(PIPELINE_STAGES)
            self._update_progress(
                PipelineProgress(
                    is_running=True,
                    video_id=video.id,
                    video_title=video.title,
                    stage=stage,
                    stage_progress=stage_pct,
                    overall_progress=overall_base + stage_fraction * overall_share,
                    video_progress=stage_fraction,
                )
            )

        self._video_start_time = time.monotonic()
        self._mark_processing(video.id)

        if not stage_done("validate"):
            update("validate", 0.0)
            if self._stop.is_set():
                return
            self._validate_video(video)
            self._advance_stage(video.id, "validate")
            update("validate", 1.0)

        wav_path = cfg.paths.data_dir / "transcripts" / f"{video.id}.wav"
        if not stage_done("audio"):
            update("audio", 0.0)
            if self._stop.is_set():
                return
            audio.extract_audio(video.file_path, wav_path)
            self._advance_stage(video.id, "audio")
            update("audio", 1.0)

        transcript_path = cfg.paths.data_dir / "transcripts" / f"{video.id}.json"
        if not stage_done("transcribe"):
            update("transcribe", 0.0)
            if self._stop.is_set():
                return
            words = transcriber.transcribe(
                wav_path, whisper_model, glossary, cfg.transcription.language,
                progress_callback=lambda pct: update("transcribe", pct),
            )
            _save_transcript(transcript_path, words)
            # Delete WAV immediately to save disk space
            wav_path.unlink(missing_ok=True)
            self._advance_stage(video.id, "transcribe")
            update("transcribe", 1.0)

        corrected_path = cfg.paths.data_dir / "transcripts" / f"{video.id}_corrected.json"
        if not stage_done("correct"):
            update("correct", 0.0)
            if self._stop.is_set():
                return
            words = _load_transcript(transcript_path)
            corrected_words = corrector.apply_corrections(words)
            _save_transcript(corrected_path, corrected_words)
            self._advance_stage(video.id, "correct")
            update("correct", 1.0)

        if not stage_done("chunk"):
            update("chunk", 0.0)
            if self._stop.is_set():
                return
            corrected_words = _load_transcript(corrected_path)
            chapter_breaks = detect_chapter_breaks(corrected_words, MIN_SILENCE_GAP_SECONDS)
            raw_chunks = chunk_transcript(
                corrected_words,
                cfg.chunking.chunk_duration_seconds,
                cfg.chunking.overlap_seconds,
                cfg.chunking.sentence_snap_window_seconds,
                chapter_breaks=chapter_breaks,
            )
            logger.info(f"'{video.title}': {len(raw_chunks)} chunks")
            self._advance_stage(video.id, "chunk")
            update("chunk", 1.0)

        if not stage_done("classify"):
            update("classify", 0.0)
            if self._stop.is_set():
                return
            # Re-load raw_chunks if we resumed past chunk stage
            if stage_done("chunk"):
                corrected_words = _load_transcript(corrected_path)
                chapter_breaks = detect_chapter_breaks(corrected_words, MIN_SILENCE_GAP_SECONDS)
                raw_chunks = chunk_transcript(
                    corrected_words,
                    cfg.chunking.chunk_duration_seconds,
                    cfg.chunking.overlap_seconds,
                    cfg.chunking.sentence_snap_window_seconds,
                    chapter_breaks=chapter_breaks,
                )
            classified_chunks = [classifier.classify_chunk(rc, video) for rc in raw_chunks]
            self._save_chunks_to_db(classified_chunks, raw_chunks, conn)
            from jitsudb.pipeline.entropy import score_and_store_chunks
            score_and_store_chunks(classified_chunks, conn)
            conn.commit()
            self._advance_stage(video.id, "classify")
            update("classify", 1.0)

        if not stage_done("embed"):
            update("embed", 0.0)
            if self._stop.is_set():
                return
            classified_chunks = self._load_chunks_from_db(video.id, conn)
            embedder.embed_and_index(
                classified_chunks, embed_model, chroma_collection,
                document_prefix=cfg.embedding.document_prefix,
            )
            from jitsudb.pipeline.drift import detect_drift
            detect_drift(video.id, video.primary_category, classified_chunks, chroma_collection, conn)
            self._advance_stage(video.id, "embed")
            update("embed", 1.0)

        if not stage_done("thumbnail"):
            update("thumbnail", 0.0)
            if self._stop.is_set():
                return
            classified_chunks = self._load_chunks_from_db(video.id, conn)
            thumb_dir = cfg.paths.data_dir / "thumbnails"
            thumb_paths = thumbnails.generate_thumbnails_batch(
                classified_chunks,
                video.file_path,
                thumb_dir,
                width=cfg.thumbnails.width,
                quality=cfg.thumbnails.quality,
            )
            # Write thumbnail paths back to SQLite
            for chunk, path in zip(classified_chunks, thumb_paths):
                if path.exists():
                    conn.execute(
                        "UPDATE chunks SET thumbnail_path = ? WHERE id = ?",
                        (str(path), chunk.id),
                    )
            conn.commit()
            self._advance_stage(video.id, "thumbnail")
            update("thumbnail", 1.0)

        if not stage_done("visual_embed"):
            if clip_model is not None and visual_collection is not None:
                update("visual_embed", 0.0)
                if self._stop.is_set():
                    return
                classified_chunks = self._load_chunks_from_db(video.id, conn)
                from jitsudb.pipeline.visual_embedder import embed_chunks_visual
                embed_chunks_visual(classified_chunks, clip_model, visual_collection)
            self._advance_stage(video.id, "visual_embed")
            update("visual_embed", 1.0)

        if not stage_done("graph_edges"):
            update("graph_edges", 0.0)
            if self._stop.is_set():
                return
            classified_chunks = self._load_chunks_from_db(video.id, conn)
            from jitsudb.db.graph import build_edges_for_video
            build_edges_for_video(video.id, classified_chunks, chroma_collection, conn=conn)
            self._advance_stage(video.id, "graph_edges")
            update("graph_edges", 1.0)

        self._mark_done(video.id)
        logger.info(f"'{video.title}' fully indexed")

    def _validate_video(self, video: Video) -> None:
        """Check codec, duration > 0, and audio stream presence before processing."""
        if not video.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {video.file_path}")

        if not audio.verify_audio_stream(video.file_path):
            raise ValueError(f"No audio stream found in {video.file_path.name}")

        duration = audio.get_video_duration(video.file_path)
        if duration <= 0:
            raise ValueError(f"Invalid duration ({duration}s) for {video.file_path.name}")

        # Update the video's duration in DB (seed_catalog sets it to 0)
        conn = get_connection()
        conn.execute(
            "UPDATE videos SET duration_seconds = ? WHERE id = ?",
            (duration, video.id),
        )
        conn.commit()
        logger.debug(f"Validated '{video.title}' ({duration:.1f}s)")

    def _update_progress(self, progress: PipelineProgress) -> None:
        """Write progress to the shared state dict under a lock."""
        now = time.monotonic()
        if self._start_time is not None:
            elapsed = now - self._start_time
            progress.elapsed_seconds = elapsed
            qp = progress.overall_progress
            if qp > 0.01:
                progress.queue_eta_seconds = elapsed / qp * (1.0 - qp)
        if self._video_start_time is not None:
            video_elapsed = now - self._video_start_time
            vp = progress.video_progress
            if vp > 0.01:
                progress.video_eta_seconds = video_elapsed / vp * (1.0 - vp)
        with self._lock:
            self._progress.update(progress.model_dump())

    def get_progress(self) -> PipelineProgress:
        """Return a snapshot of the current pipeline progress."""
        with self._lock:
            return PipelineProgress(**self._progress)

    def _load_queue(self) -> list[tuple[QueueItem, Video]]:
        """Return all pending/processing queue items ordered by priority."""
        conn = get_connection()
        rows = conn.execute(
            """
            SELECT q.video_id, q.priority, q.status, q.error_message, q.stage_completed,
                   v.id, v.title, v.instructor, v.file_path, v.duration_seconds,
                   v.gi_nogi, v.primary_category, v.status as video_status,
                   v.added_at, v.processed_at
            FROM processing_queue q
            JOIN videos v ON v.id = q.video_id
            WHERE q.status IN ('pending', 'processing')
            ORDER BY q.priority ASC, v.added_at ASC
            """
        ).fetchall()

        result: list[tuple[QueueItem, Video]] = []
        for row in rows:
            qi = QueueItem(
                video_id=row["video_id"],
                priority=row["priority"],
                status=row["status"],
                error_message=row["error_message"],
                stage_completed=row["stage_completed"],
            )
            vid = Video(
                id=row["id"],
                title=row["title"],
                instructor=row["instructor"],
                file_path=Path(row["file_path"]),
                duration_seconds=row["duration_seconds"],
                gi_nogi=row["gi_nogi"],
                primary_category=row["primary_category"],
                status=row["video_status"],
                added_at=row["added_at"],
                processed_at=row["processed_at"],
            )
            result.append((qi, vid))

        return result

    def _advance_stage(self, video_id: str, stage: str) -> None:
        conn = get_connection()
        conn.execute(
            "UPDATE processing_queue SET stage_completed = ?, status = 'processing' WHERE video_id = ?",
            (stage, video_id),
        )
        conn.commit()

    def _mark_processing(self, video_id: str) -> None:
        conn = get_connection()
        conn.execute(
            "UPDATE processing_queue SET status = 'processing' WHERE video_id = ?",
            (video_id,),
        )
        conn.execute(
            "UPDATE videos SET status = 'processing' WHERE id = ?",
            (video_id,),
        )
        conn.commit()

    def _mark_done(self, video_id: str) -> None:
        conn = get_connection()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE processing_queue SET status = 'done' WHERE video_id = ?",
            (video_id,),
        )
        conn.execute(
            "UPDATE videos SET status = 'indexed', processed_at = ? WHERE id = ?",
            (now, video_id),
        )
        conn.commit()

    def _mark_failed(self, video_id: str, error: str) -> None:
        conn = get_connection()
        conn.execute(
            "UPDATE processing_queue SET status = 'failed', error_message = ? WHERE video_id = ?",
            (error[:2000], video_id),
        )
        conn.execute(
            "UPDATE videos SET status = 'failed' WHERE id = ?",
            (video_id,),
        )
        conn.commit()

    def _save_chunks_to_db(
        self,
        chunks: list[Chunk],
        raw_chunks: list[RawChunk],
        conn,
    ) -> None:
        """Persist classified chunks and their word timestamps to SQLite."""
        for chunk, raw in zip(chunks, raw_chunks):
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                    (id, video_id, start_time, end_time, transcript_text, context, gi_nogi)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.video_id,
                    chunk.start_time,
                    chunk.end_time,
                    chunk.transcript_text,
                    chunk.context,
                    chunk.gi_nogi,
                ),
            )

            # Clear and re-insert taxonomy labels (handles re-classify case)
            conn.execute("DELETE FROM taxonomy_labels WHERE chunk_id = ?", (chunk.id,))
            for label in chunk.position:
                conn.execute(
                    "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?, ?, ?)",
                    (chunk.id, "position", label),
                )
            for label in chunk.intent:
                conn.execute(
                    "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?, ?, ?)",
                    (chunk.id, "intent", label),
                )
            for label in chunk.techniques:
                conn.execute(
                    "INSERT INTO taxonomy_labels (chunk_id, label_type, label_value) VALUES (?, ?, ?)",
                    (chunk.id, "technique", label),
                )

            conn.execute("DELETE FROM word_timestamps WHERE chunk_id = ?", (chunk.id,))
            for word in raw.words:
                conn.execute(
                    "INSERT INTO word_timestamps (chunk_id, word, start_sec, end_sec) VALUES (?, ?, ?, ?)",
                    (chunk.id, word.word, word.start, word.end),
                )

        conn.commit()

    def _load_chunks_from_db(self, video_id: str, conn) -> list[Chunk]:
        """Load all chunks for a video from SQLite, including taxonomy labels."""
        chunk_rows = conn.execute(
            "SELECT * FROM chunks WHERE video_id = ? ORDER BY start_time ASC",
            (video_id,),
        ).fetchall()

        chunks: list[Chunk] = []
        for row in chunk_rows:
            labels = conn.execute(
                "SELECT label_type, label_value FROM taxonomy_labels WHERE chunk_id = ?",
                (row["id"],),
            ).fetchall()

            positions = [r["label_value"] for r in labels if r["label_type"] == "position"]
            intents = [r["label_value"] for r in labels if r["label_type"] == "intent"]
            techs = [r["label_value"] for r in labels if r["label_type"] == "technique"]

            chunks.append(
                Chunk(
                    id=row["id"],
                    video_id=row["video_id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    transcript_text=row["transcript_text"],
                    thumbnail_path=Path(row["thumbnail_path"]) if row["thumbnail_path"] else None,
                    position=positions,
                    context=row["context"],
                    intent=intents,
                    techniques=techs,
                    gi_nogi=row["gi_nogi"],
                )
            )

        return chunks


def _save_transcript(path: Path, words: list[WordTimestamp]) -> None:
    """Serialise a word-level transcript to JSON on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"word": w.word, "start": w.start, "end": w.end} for w in words]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_transcript(path: Path) -> list[WordTimestamp]:
    """Deserialise a word-level transcript from JSON on disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [WordTimestamp(word=d["word"], start=d["start"], end=d["end"]) for d in data]
