from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Query, Request

from lrage.webapp.store.runs_repo import RunsRepository

router = APIRouter(prefix="/api/compare", tags=["compare"])

MAX_RUNS = 4


@router.get("")
def compare(request: Request, run_ids: str = Query(min_length=1)):
    ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
    if not 2 <= len(ids) <= MAX_RUNS:
        raise HTTPException(
            status_code=422, detail=f"Pick 2 to {MAX_RUNS} runs to compare"
        )

    repo: RunsRepository = request.app.state.repo
    runs = []
    for rid in ids:
        record = repo.get(rid)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Run {rid} not found")
        if record["status"] != "completed":
            raise HTTPException(
                status_code=409, detail=f"Run {rid} is {record['status']}"
            )
        runs.append(record)

    # Union of (task, metric) rows in first-seen order, values per run.
    rows: "OrderedDict[tuple, dict]" = OrderedDict()
    for record in runs:
        for entry in record.get("summary") or []:
            key = (entry["task"], entry["metric"])
            row = rows.setdefault(
                key,
                {"task": entry["task"], "metric": entry["metric"], "values": {}},
            )
            row["values"][record["run_id"]] = {
                "value": entry["value"],
                "stderr": entry.get("stderr"),
            }

    return {
        "runs": [
            {
                "run_id": r["run_id"],
                "name": r["name"],
                "model": r["model"],
                "tasks": r["tasks"],
                "config": r["config"],
                "finished_at": r["finished_at"],
            }
            for r in runs
        ],
        "table": list(rows.values()),
    }
