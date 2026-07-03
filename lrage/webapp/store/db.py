import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  name TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  config_json TEXT NOT NULL,
  model TEXT NOT NULL,
  tasks_json TEXT NOT NULL,
  output_dir TEXT,
  results_path TEXT,
  samples_paths_json TEXT,
  summary_json TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """One short-lived connection per operation keeps stdlib sqlite3 safe
    across the API event loop and the worker thread."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
