import threading
import time

import pytest
from fastapi.testclient import TestClient

from lrage.webapp import registry_meta
from lrage.webapp.app import create_app
from lrage.webapp.settings import Settings

STUB_TASKS = [
    {"name": "abercrombie", "type": "task"},
    {"name": "barexam_qa", "type": "task"},
    {"name": "legalbench", "type": "group"},
]

SUBMISSION = {
    "name": "test run",
    "config": {
        "model": "openai-chat-completions",
        "model_args": "model=gpt-4o-mini",
        "tasks": ["abercrombie"],
    },
}


@pytest.fixture(autouse=True)
def reset_sse_app_status():
    # sse-starlette binds a module-level asyncio.Event to the first event
    # loop it sees; each TestClient uses a fresh loop, so reset it per test.
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit_event = None


@pytest.fixture
def settings(tmp_path):
    # Nonexistent static_dir keeps tests hermetic regardless of whether the
    # frontend has been built into lrage/webapp/static.
    return Settings(data_dir=tmp_path / "data", static_dir=tmp_path / "nostatic")


@pytest.fixture
def stub_tasks(monkeypatch):
    # The real list_tasks imports the full task stack and indexes ~2.5k YAMLs.
    monkeypatch.setattr(registry_meta, "list_tasks", lambda: STUB_TASKS)
    return STUB_TASKS


class FakeWorker:
    """Injectable worker: publishes a realistic event sequence and finishes.
    Set `gate` to make it block until released (for cancellation tests)."""

    def __init__(self):
        self.gate = None  # threading.Event to block on, if set
        self.started = threading.Event()
        self.calls = []
        self.fail_with = None

    def __call__(self, ctx):
        self.calls.append(ctx.job.run_id)
        self.started.set()
        ctx.publish({"type": "phase", "phase": "loading_model"})
        ctx.publish(
            {
                "type": "progress",
                "phase": "running_requests",
                "desc": "Running requests",
                "n": 1,
                "total": 4,
                "pct": 25.0,
            }
        )
        ctx.publish({"type": "log", "level": "INFO", "message": "hello from fake"})
        if self.gate is not None:
            # Poll the gate so cancellation can interrupt us.
            while not self.gate.wait(timeout=0.05):
                ctx.token.raise_if_cancelled()
        if self.fail_with is not None:
            raise self.fail_with
        ctx.repo.set_outputs(
            ctx.job.run_id,
            output_dir=str(ctx.settings.output_root / ctx.job.run_id),
            results_path=None,
            samples_paths=[],
            summary=[],
        )


@pytest.fixture
def fake_worker():
    return FakeWorker()


@pytest.fixture
def client(settings, stub_tasks, fake_worker):
    app = create_app(settings, worker_fn=fake_worker)
    with TestClient(app) as c:
        yield c


def wait_for_status(client, run_id, statuses, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(
        f"run {run_id} did not reach {statuses} within {timeout}s; last: {body}"
    )
