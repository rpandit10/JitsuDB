"""Integration tests for GET /api/search using FastAPI TestClient."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from jitsudb.api.app import create_app
from jitsudb.api.dependencies import get_search_engine


@pytest.fixture
def client():
    """TestClient with the search engine dependency overridden by a mock."""
    app = create_app()

    mock_engine = MagicMock()
    mock_engine.search.return_value = []

    app.dependency_overrides[get_search_engine] = lambda: mock_engine
    return TestClient(app)


class TestSearchRoute:
    def test_returns_200_on_valid_query(self, client):
        res = client.get("/api/search?q=armbar")
        assert res.status_code == 200

    def test_returns_list(self, client):
        res = client.get("/api/search?q=kimura")
        assert isinstance(res.json(), list)

    def test_missing_query_returns_422(self, client):
        res = client.get("/api/search")
        assert res.status_code == 422

    def test_accepts_filter_params(self, client):
        res = client.get("/api/search?q=escape&position=side_control&intent=defense")
        assert res.status_code == 200
