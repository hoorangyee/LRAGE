import threading

from tests.webapp.conftest import SUBMISSION, wait_for_status


def test_cancel_running_job(client, fake_worker):
    fake_worker.gate = threading.Event()
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    assert fake_worker.started.wait(timeout=5)

    res = client.post(f"/api/runs/{run_id}/cancel")
    assert res.status_code == 202
    assert res.json()["status"] in ("cancelling", "cancelled")

    body = wait_for_status(client, run_id, {"cancelled"})
    assert body["status"] == "cancelled"


def test_cancel_queued_job(client, fake_worker):
    fake_worker.gate = threading.Event()
    first = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    assert fake_worker.started.wait(timeout=5)
    second = client.post("/api/runs", json=SUBMISSION).json()["run_id"]

    res = client.post(f"/api/runs/{second}/cancel")
    assert res.status_code == 202
    assert res.json()["status"] == "cancelled"

    fake_worker.gate.set()
    wait_for_status(client, first, {"completed"})
    # The cancelled-queued job must never have run.
    assert fake_worker.calls == [first]


def test_cancel_completed_run_conflicts(client):
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    wait_for_status(client, run_id, {"completed"})
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409


def test_delete_running_run_conflicts(client, fake_worker):
    fake_worker.gate = threading.Event()
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    assert fake_worker.started.wait(timeout=5)
    assert client.delete(f"/api/runs/{run_id}").status_code == 409
    fake_worker.gate.set()
    wait_for_status(client, run_id, {"completed"})
