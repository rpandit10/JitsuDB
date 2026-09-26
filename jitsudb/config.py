"""Application configuration loaded from config.yaml via Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    """File system paths."""

    video_library: Path = Path("./video_library")
    data_dir: Path = Path("./data")


class TranscriptionConfig(BaseModel):
    """faster-whisper transcription settings."""

    model_size: Literal["tiny", "base", "small", "medium", "large"] = "medium"
    compute_type: str = "float16"
    language: str = "en"
    glossary_path: Path = Path("./jitsudb/taxonomy/glossary.txt")


class ChunkingConfig(BaseModel):
    """Sliding-window chunking parameters."""

    chunk_duration_seconds: float = 45.0
    overlap_seconds: float = 15.0
    sentence_snap_window_seconds: float = 5.0


class EmbeddingConfig(BaseModel):
    """Sentence-transformer embedding settings."""

    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    device: str = "cpu"
    embedding_dim: int = 768
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "
    trust_remote_code: bool = True
    collection_name: str = "jitsudb_chunks"
    encoder_path: str | None = None
    ft_collection: str = "text_ft"


class RerankerConfig(BaseModel):
    """Cross-encoder re-ranking settings."""

    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda"
    conformal_coverage: float | None = None


class VisualEmbeddingConfig(BaseModel):
    """CLIP visual embedding for multi-modal retrieval."""

    enabled: bool = True
    model_name: str = "clip-ViT-B-32"
    device: str = "cpu"
    visual_weight: float = 0.3


class RagConfig(BaseModel):
    """Local LLM RAG answer synthesis via Ollama."""

    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    max_context_chunks: int = 5
    max_tokens: int = 400
    temperature: float = 0.1
    faithfulness_check: bool = False


class SearchConfig(BaseModel):
    """Hybrid search parameters."""

    default_top_k: int = 30
    semantic_weight: float = 0.7
    beam_width: int = 2
    max_expand: int = 4
    reranker_top_k: int = 20
    candidate_k: int = 30
    hyde_enabled: bool = False
    hyde_n_hypotheticals: int = 3
    colbert_enabled: bool = False
    colbert_weight: float = 0.3


class ThumbnailsConfig(BaseModel):
    """Thumbnail generation settings."""

    width: int = 320
    format: str = "jpg"
    quality: int = 5


class ClipsConfig(BaseModel):
    """On-demand clip generation settings."""

    duration_seconds: int = 15
    padding_seconds: int = 3
    max_cache_count: int = 100
    width: int = 640


class QueueConfig(BaseModel):
    """Processing queue behaviour."""

    fallback_order: Literal["alphabetical", "instructor", "folder", "filesize"] = "folder"


class YouTubeScraperConfig(BaseModel):
    """Automatic recommendation scraper settings."""

    reputable_channels: list[str] = []
    search_tags: list[str] = [
        "BJJ instructional",
        "jiu jitsu technique breakdown",
        "grappling instructional",
    ]
    max_results_per_tag: int = 10
    max_per_channel: int = 10
    min_duration_seconds: int = 60
    cache_ttl_seconds: int = 3600


class YouTubeConfig(BaseModel):
    """yt-dlp download settings."""

    enabled: bool = False
    download_dir: Path = Path("./data/youtube_downloads")
    default_format: str = "bestvideo[height<=1080]+bestaudio/best"
    scraper: YouTubeScraperConfig = YouTubeScraperConfig()


class AppConfig(BaseModel):
    """Root configuration object. All sub-configs have defaults matching config.yaml."""

    paths: PathsConfig = PathsConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    reranker: RerankerConfig = RerankerConfig()
    visual_embedding: VisualEmbeddingConfig = VisualEmbeddingConfig()
    rag: RagConfig = RagConfig()
    search: SearchConfig = SearchConfig()
    thumbnails: ThumbnailsConfig = ThumbnailsConfig()
    clips: ClipsConfig = ClipsConfig()
    queue: QueueConfig = QueueConfig()
    youtube: YouTubeConfig = YouTubeConfig()


def load_config(config_path: Path = Path("config.yaml")) -> AppConfig:
    """Load AppConfig from a YAML file, falling back to defaults if the file is absent."""
    if config_path.exists():
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig.model_validate(raw)
    return AppConfig()


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Return the application config singleton, loading from disk on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Clear the cached config singleton (used in tests to reload from a temp path)."""
    global _config
    _config = None
