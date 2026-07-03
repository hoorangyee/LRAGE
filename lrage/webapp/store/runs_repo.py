import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lrage.webapp.schemas import RunStatus
from lrage.webapp.store import db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunsRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db.init_schema(db_path)

    @contextmanager
    def _conn(self):
        conn = db.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create(
        self,
        run_id: str,
        name: Optional[str],
        config: Dict[str, Any],
        model: str,
        tasks: List[str],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, name, status, created_at, config_json,"
                " model, tasks_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    name,
                    RunStatus.queued.value,
                    _now(),
                    json.dumps(config),
                    model,
                    json.dumps(tasks),
                ),
            )

    def set_status(self, run_id: str, status: RunStatus) -> None:
        stamps = ""
        if status == RunStatus.running:
            stamps = ", started_at = ?"
        elif status.is_terminal:
            stamps = ", finished_at = ?"
        params = [status.value]
        if stamps:
            params.append(_now())
        params.append(run_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE runs SET status = ?{stamps} WHERE run_id = ?", params
            )

    def set_error(self, run_id: str, error: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET error = ? WHERE run_id = ?", (error, run_id)
            )

    def set_outputs(
        self,
        run_id: str,
        output_dir: str,
        results_path: Optional[str],
        samples_paths: List[str],
        summary: Optional[List[Dict[str, Any]]],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET output_dir = ?, results_path = ?,"
                " samples_paths_json = ?, summary_json = ? WHERE run_id = ?",
                (
                    output_dir,
                    results_path,
                    json.dumps(samples_paths),
                    json.dumps(summary) if summary is not None else None,
                    run_id,
                ),
            )

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(
        self,
        status: Optional[RunStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM runs"
        params: List[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete(self, run_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        return cur.rowcount > 0

    def sweep_stale(self) -> int:
        """Mark runs left running/cancelling/queued by a dead server as failed.
        Called once at startup, before the job manager starts."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE runs SET status = ?, error = ?, finished_at = ?"
                " WHERE status IN (?, ?, ?)",
                (
                    RunStatus.failed.value,
                    "interrupted by server shutdown",
                    _now(),
                    RunStatus.running.value,
                    RunStatus.cancelling.value,
                    RunStatus.queued.value,
                ),
            )
        return cur.rowcount

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json"))
        d["tasks"] = json.loads(d.pop("tasks_json"))
        d["samples_paths"] = (
            json.loads(d["samples_paths_json"]) if d.get("samples_paths_json") else []
        )
        d.pop("samples_paths_json", None)
        d["summary"] = (
            json.loads(d["summary_json"]) if d.get("summary_json") else None
        )
        d.pop("summary_json", None)
        return d
