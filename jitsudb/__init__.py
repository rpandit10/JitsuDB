"""JitsuDB — local-first BJJ instructional video search engine."""

import os

# Set to "1" after all models are cached locally to skip HF Hub update checks
# and avoid startup delay / 429 rate-limit errors. Leave as "0" on a fresh
# machine so models can download on first run.
os.environ.setdefault("HF_HUB_OFFLINE", "0")

__version__ = "0.1.0"
