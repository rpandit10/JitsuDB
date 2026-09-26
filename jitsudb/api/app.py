"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jitsudb.api.routes import answer, conformal, feedback, instructionals, labels, media, metrics, process, queue, search, status, videos, youtube
from jitsudb.db import init_db


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Registers all route modules, sets up CORS for local frontend dev,
    and ensures the database schema exists on startup.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import sentence_transformers  # noqa: F401 — pre-init transformers before pipeline thread starts
        init_db()
        from jitsudb.api.dependencies import start_pipeline
        start_pipeline()
        yield

    app = FastAPI(
        title="JitsuDB",
        description="Local-first BJJ instructional video search engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # Vite dev server
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(search.router, prefix="/api")
    app.include_router(answer.router, prefix="/api")
    app.include_router(videos.router, prefix="/api")
    app.include_router(queue.router, prefix="/api")
    app.include_router(process.router, prefix="/api")
    app.include_router(status.router, prefix="/api")
    app.include_router(media.router, prefix="/api")
    app.include_router(youtube.router, prefix="/api")
    app.include_router(instructionals.router, prefix="/api")
    app.include_router(labels.router, prefix="/api")
    app.include_router(feedback.router, prefix="/api")
    app.include_router(metrics.router, prefix="/api")
    app.include_router(conformal.router, prefix="/api")

    return app


app = create_app()


def main() -> None:
    """Entry point for the ``jitsudb-server`` CLI command."""
    import uvicorn
    uvicorn.run("jitsudb.api.app:app", host="127.0.0.1", port=8000, reload=True)
