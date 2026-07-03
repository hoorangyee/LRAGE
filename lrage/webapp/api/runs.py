import asyncio
import json
import queue
import shutil
from functools import partial
from pathlib import Path
from typing import Optional

import anyio.to_thread
from fastapi import APIRouter, HTTPException, Request, Response
from sse_starlette.sse import EventSourceResponse

from lrage.webapp.jobs.events import STREAM_END
from lrage.webapp.jobs.manager import JobManager, QueueFull
from lrage.webapp.schemas import (
    ProgressSnapshot,
    RunDetail,
    RunListItem,
    RunStatus,
    RunSubmission,
)
from lrage.webapp.store.runs_repo import RunsRepository

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _repo(request: Request) -> RunsRepository:
    return request.app.state.repo


def _manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def _headline_metric(record: dict) -> Optional[dict]:
    summary = record.get("summary") or []
    if not summary:
        return None
    row = summary[0]
    return {"metric": row["metric"], "value": row["value"]}


def _to_list_item(record: dict) -> RunListItem:
    return RunListItem(
        run_id=record["run_id"],
        name=record["name"],
        status=RunStatus(record["status"]),
        created_at=record["created_at"],
        started_at=record["started_at"],
        finished_at=record["finished_at"],
        model=record["model"],
        tasks=record["tasks"],
        error=record["error"],
        headline_metric=_headline_metric(record),
    )


@router.post("", status_code=201)
def submit_run(submission: RunSubmission, request: Request):
    try:
        job = _manager(request).submit(submission)
    except QueueFull:
        raise HTTPException(status_code=409, detail="Run queue is full")
    return {
        "run_id": job.run_id,
        "status": job.status.value,
        "queue_position": _manager(request).queue_position(job.run_id),
    }


@router.get("")
def list_runs(
    request: Request,
    status: Optional[RunStatus] = None,
    limit: int = 100,
    offset: int = 0,
):
    records = _repo(request).list(status=status, limit=limit, offset=offset)
    return {"runs": [_to_list_item(r).model_dump() for r in records]}


@router.get("/{run_id}")
def get_run(run_id: str, request: Request):
    record = _repo(request).get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    detail = RunDetail(
        **_to_list_item(record).model_dump(),
        config=record["config"],
    )
    job = _manager(request).get(run_id)
    if job is not None and not job.status.is_terminal:
        detail.progress = job.progress
        detail.queue_position = _manager(request).queue_position(run_id)
    return detail.model_dump()


@router.post("/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str, request: Request):
    record = _repo(request).get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    result = _manager(request).cancel(run_id)
    if result is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run is already {record['status']}",
        )
    return {"run_id": run_id, "status": result.value}


@router.delete("/{run_id}", status_code=204)
def delete_run(run_id: str, request: Request, purge_files: bool = False):
    record = _repo(request).get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if _manager(request).is_active(run_id):
        raise HTTPException(status_code=409, detail="Run is still active")
    if purge_files and record.get("output_dir"):
        out = Path(record["output_dir"]).resolve()
        # Refuse to delete anything outside the server's output root.
        root = request.app.state.settings.output_root.resolve()
        if _is_within(out, root):
            shutil.rmtree(out, ignore_errors=True)
    _repo(request).delete(run_id)
    return Response(status_code=204)


@router.get("/{run_id}/events")
async def run_events(run_id: str, request: Request):
    record = _repo(request).get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")

    job = _manager(request).get(run_id)

    last_id_raw = request.headers.get("last-event-id") or request.query_params.get(
        "lastEventId"
    )
    try:
        last_event_id = int(last_id_raw) if last_id_raw else None
    except ValueError:
        last_event_id = None

    if job is None:
        # Historical run from a previous server session: emit the terminal
        # state immediately and end the stream.
        async def replay_terminal():
            yield {
                "id": "1",
                "event": "done",
                "data": json.dumps({"type": "done", "status": record["status"]}),
            }

        return EventSourceResponse(replay_terminal())

    subscriber = job.events.subscribe(last_event_id)

    async def stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await anyio.to_thread.run_sync(
                        partial(subscriber.get, timeout=2.0)
                    )
                except queue.Empty:
                    continue
                except asyncio.CancelledError:
                    break
                if event is STREAM_END:
                    break
                yield {
                    "id": str(event["id"]),
                    "event": event.get("type", "message"),
                    "data": json.dumps(event),
                }
                if event.get("type") == "done":
                    break
        finally:
            job.events.unsubscribe(subscriber)

    return EventSourceResponse(stream())
