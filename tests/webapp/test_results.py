import pytest

from tests.webapp.conftest import SUBMISSION, wait_for_status
from tests.webapp.test_worker import (  # noqa: F401 (fixtures)
    fake_evaluate,
    real_worker_client,
    submission_with_keys,
)


def test_results_endpoint(real_worker_client):
    client = real_worker_client
    run_id = client.post("/api/runs", json=submission_with_keys()).json()["run_id"]
    wait_for_status(client, run_id, {"completed"}, timeout=15)

    body = client.get(f"/api/runs/{run_id}/results").json()
    assert body["run_id"] == run_id
    assert body["table"] == [
        {
            "task": "abercrombie_tiny",
            "n_shot": "0",
            "metric": "acc",
            "value": 0.75,
            "stderr": 0.05,
        }
    ]
    # The raw results JSON written by EvaluationTracker is included.
    assert body["results"]["results"]["abercrombie_tiny"]["acc,none"] == 0.75
    assert body["results"]["model_source"] == "openai-chat-completions"


def test_results_conflict_while_not_completed(client, fake_worker):
    import threading

    fake_worker.gate = threading.Event()
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    assert fake_worker.started.wait(timeout=5)
    assert client.get(f"/api/runs/{run_id}/results").status_code == 409
    fake_worker.gate.set()
    wait_for_status(client, run_id, {"completed"})


def test_results_unknown_run(client):
    assert client.get("/api/runs/nope/results").status_code == 404


def test_logs_endpoint(real_worker_client):
    client = real_worker_client
    run_id = client.post("/api/runs", json=submission_with_keys()).json()["run_id"]
    wait_for_status(client, run_id, {"completed"}, timeout=15)

    res = client.get(f"/api/runs/{run_id}/logs")
    assert res.status_code == 200
    assert "Running generate_until requests" in res.text


def test_logs_empty_for_queued_style_run(client):
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    wait_for_status(client, run_id, {"completed"})
    # FakeWorker doesn't write a log file; endpoint degrades to empty.
    assert client.get(f"/api/runs/{run_id}/logs").text == ""
