"""End-to-end integration test: video file → searchable chunks in ChromaDB + SQLite.

Requires a real short video file (≤ 5 minutes) placed at tests/fixtures/sample.mp4.
The test is skipped if the fixture file is absent so CI passes without large binaries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_VIDEO = Path(__file__).parent.parent / "fixtures" / "sample.mp4"


@pytest.mark.skipif(
    not FIXTURE_VIDEO.exists(),
    reason="Sample video fixture not present — place a short .mp4 at tests/fixtures/sample.mp4",
)
class TestFullPipeline:
    def test_pipeline_produces_chunks(self, tmp_path):
        """Running the orchestrator on a real video should create rows in chunks table."""
        # TODO: implement when orchestrator is complete (Stage 3)
        pytest.skip("Orchestrator not yet implemented")

    def test_chunks_are_searchable(self, tmp_path):
        """After indexing, a relevant query should return at least one result."""
        # TODO: implement when search engine is complete (Stage 4)
        pytest.skip("Search engine not yet implemented")
