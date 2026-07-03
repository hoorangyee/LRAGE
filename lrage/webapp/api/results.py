import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from lrage.webapp.store import samples_reader
from lrage.webapp.store.runs_repo import RunsRepository

router = APIRouter(prefix="/api/runs", tags=["results"])

LOG_TAIL_BYTES = 512 * 1024


def _repo(request: Request) -> RunsRepository:
    return request.app.state.repo


def _get_record(request: Request, run_id: str) -> dict:
    record = _repo(request).get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return record


@router.get("/{run_id}/results")
def get_results(run_id: str, request: Request):
    record = _get_record(request, run_id)
    if record["status"] != "completed":
        raise HTTPException(
            status_code=409, detail=f"Run is {record['status']}, not completed"
        )

    raw = {}
    if record.get("results_path"):
        path = Path(record["results_path"])
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)

    return {
        "run_id": run_id,
        "table": record.get("summary") or [],
        "results": raw,
    }


def _samples_paths_by_task(record: dict) -> dict:
    return {
        samples_reader.task_name_for(path): path
        for path in record.get("samples_paths") or []
    }


@router.get("/{run_id}/samples")
def list_samples(
    run_id: str,
    request: Request,
    task: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    q: Optional[str] = None,
    incorrect_only: bool = False,
    judge_min: Optional[float] = None,
    judge_max: Optional[float] = None,
):
    record = _get_record(request, run_id)
    by_task = _samples_paths_by_task(record)
    if not by_task:
        return {"tasks": [], "task": None, "total": 0, "page": 1, "items": []}

    task = task if task in by_task else sorted(by_task)[0]
    samples = samples_reader.load_samples(by_task[task])
    filtered = samples_reader.filter_samples(
        samples,
        q=q,
        incorrect_only=incorrect_only,
        judge_min=judge_min,
        judge_max=judge_max,
    )
    start = (page - 1) * page_size
    items = [samples_reader.to_list_item(s) for s in filtered[start : start + page_size]]
    return {
        "tasks": sorted(by_task),
        "task": task,
        "total": len(filtered),
        "page": page,
        "page_size": page_size,
        "items": items,
    }


@router.get("/{run_id}/samples/{task}/{doc_id}")
def get_sample(run_id: str, task: str, doc_id: int, request: Request):
    record = _get_record(request, run_id)
    by_task = _samples_paths_by_task(record)
    if task not in by_task:
        raise HTTPException(status_code=404, detail="No samples for this task")
    sample = samples_reader.find_sample(
        samples_reader.load_samples(by_task[task]), doc_id
    )
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    return {
        "run_id": run_id,
        "task": task,
        "sample": sample,
        "metrics": samples_reader.metric_values(sample),
    }


@router.get("/{run_id}/logs")
def get_logs(run_id: str, request: Request):
    record = _get_record(request, run_id)
    if not record.get("output_dir"):
        return PlainTextResponse("")
    log_path = Path(record["output_dir"]) / "run.log"
    if not log_path.is_file():
        return PlainTextResponse("")
    size = log_path.stat().st_size
    with open(log_path, "rb") as f:
        if size > LOG_TAIL_BYTES:
            f.seek(size - LOG_TAIL_BYTES)
        content = f.read().decode("utf-8", errors="replace")
    return PlainTextResponse(content)
