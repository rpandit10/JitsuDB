"""Integration tests for /api/queue routes using FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jitsudb.api.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient wired to a temporary database."""
    import jitsudb.config as cfg_module
    from jitsudb.config import AppConfig, PathsConfig

    cfg_module._config = AppConfig(paths=PathsConfig(data_dir=tmp_path))

    app = create_app()
    with TestClient(app) as c:
        yield c

    cfg_module._config = None


class TestQueueRoutes:
    def test_list_queue_returns_200(self, client):
        res = client.get("/api/queue")
        assert res.status_code == 200

    def test_list_queue_returns_list(self, client):
        res = client.get("/api/queue")
        assert isinstance(res.json(), list)

    def test_enqueue_missing_body_returns_422(self, client):
        res = client.post("/api/queue")
        assert res.status_code == 422
