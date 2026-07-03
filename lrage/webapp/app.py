from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from lrage.webapp import registry_meta
from lrage.webapp.api import compare, meta, results, runs
from lrage.webapp.jobs.manager import JobManager, WorkerFn
from lrage.webapp.settings import Settings
from lrage.webapp.store.runs_repo import RunsRepository

try:
    LRAGE_VERSION = version("lrage")
except PackageNotFoundError:
    LRAGE_VERSION = "unknown"


def create_app(
    settings: Optional[Settings] = None,
    worker_fn: Optional[WorkerFn] = None,
) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()

    app = FastAPI(title="LRAGE", version=LRAGE_VERSION, docs_url="/api/docs")
    app.state.settings = settings

    repo = RunsRepository(settings.db_path)
    repo.sweep_stale()
    app.state.repo = repo

    manager = JobManager(repo, settings, worker_fn=worker_fn)
    manager.start()
    app.state.job_manager = manager

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": LRAGE_VERSION,
            "tasks_loaded": registry_meta.tasks_loaded(),
        }

    app.include_router(meta.router)
    app.include_router(runs.router)
    app.include_router(results.router)
    app.include_router(compare.router)

    _mount_spa(app, settings)
    return app


def _mount_spa(app: FastAPI, settings: Settings) -> None:
    index_html = settings.static_dir / "index.html"

    if not index_html.is_file():
        @app.get("/", include_in_schema=False)
        def no_frontend():
            return PlainTextResponse(
                "LRAGE API is running, but the web frontend is not built.\n"
                "Build it with: cd frontend && npm run build\n"
                "API docs: /api/docs\n"
            )
        return

    assets_dir = settings.static_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        # Serve real files (favicon etc.); anything else is a client-side
        # route and gets index.html.
        candidate = settings.static_dir / path
        if path and ".." not in path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_html)
