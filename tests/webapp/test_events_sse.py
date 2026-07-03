import json
import threading

from tests.webapp.conftest import SUBMISSION, wait_for_status


def parse_sse(text):
    """Parse an SSE body into a list of {id, event, data} dicts."""
    events = []
    text = text.replace("\r\n", "\n")
    for block in text.strip().split("\n\n"):
        event = {}
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            key, _, value = line.partition(": ")
            event[key] = value
        if event:
            events.append(event)
    return events


def test_event_stream_sequence(client):
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    with client.stream("GET", f"/api/runs/{run_id}/events") as res:
        assert res.status_code == 200
        body = "".join(res.iter_text())
    events = parse_sse(body)
    types = [e["event"] for e in events if "event" in e]
    assert types[0] == "status"  # queued
    assert "phase" in types
    assert "progress" in types
    assert "log" in types
    assert types[-1] == "done"
    done = json.loads(events[-1]["data"])
    assert done["status"] == "completed"
    # ids monotonically increase
    ids = [int(e["id"]) for e in events if "id" in e]
    assert ids == sorted(ids)


def test_event_stream_replay_with_last_event_id(client, fake_worker):
    fake_worker.gate = threading.Event()
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    assert fake_worker.started.wait(timeout=5)
    fake_worker.gate.set()
    wait_for_status(client, run_id, {"completed"})

    # Reconnect claiming we saw everything through event 2: replay must
    # resume after it.
    with client.stream(
        "GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "2"}
    ) as res:
        body = "".join(res.iter_text())
    ids = [int(e["id"]) for e in parse_sse(body) if "id" in e]
    assert ids and min(ids) == 3


def test_event_stream_for_historical_run(client, settings, stub_tasks):
    # A run that finished under a previous server session has no live job.
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    wait_for_status(client, run_id, {"completed"})

    from lrage.webapp.app import create_app
    from fastapi.testclient import TestClient

    fresh = create_app(settings, worker_fn=lambda ctx: None)
    with TestClient(fresh) as fresh_client:
        with fresh_client.stream("GET", f"/api/runs/{run_id}/events") as res:
            body = "".join(res.iter_text())
        events = parse_sse(body)
        assert events[-1]["event"] == "done"


def test_event_stream_unknown_run_404(client):
    assert client.get("/api/runs/nope/events").status_code == 404
