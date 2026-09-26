"""Project-wide constants. All magic numbers live here — never scatter them through the codebase."""

# Chunking
CHUNK_DURATION_SECONDS: float = 45.0
OVERLAP_SECONDS: float = 15.0
SENTENCE_SNAP_WINDOW_SECONDS: float = 5.0
MIN_SILENCE_GAP_SECONDS: float = 3.0
NATURAL_PAUSE_THRESHOLD_SECONDS: float = 1.5

# Transcription
WHISPER_MODEL_SIZE: str = "medium"
WHISPER_COMPUTE_TYPE: str = "float16"
WHISPER_LANGUAGE: str = "en"

# Embedding
EMBEDDING_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM: int = 768
CHROMADB_COLLECTION_NAME: str = "jitsudb_chunks"

# Search
DEFAULT_TOP_K: int = 30
SEMANTIC_CANDIDATE_K: int = 50
SEMANTIC_WEIGHT: float = 0.7

# Thumbnails
THUMBNAIL_WIDTH: int = 320
THUMBNAIL_FORMAT: str = "jpg"
THUMBNAIL_QUALITY: int = 5
THUMBNAIL_OFFSET_SECONDS: float = 5.0

# Clips
CLIP_DURATION_SECONDS: int = 15
CLIP_PADDING_SECONDS: int = 3
CLIP_MAX_CACHE_COUNT: int = 100
CLIP_WIDTH: int = 640

# Audio extraction
AUDIO_SAMPLE_RATE: int = 16000
AUDIO_CHANNELS: int = 1

# Database
DB_FILENAME: str = "jitsudb.db"

# Taxonomy versioning — bump when dictionary.py changes to trigger re-classification
TAXONOMY_VERSION: str = "1.0.0"
