import copy

from tests.webapp.conftest import SUBMISSION, wait_for_status


def test_submit_and_complete(client, fake_worker):
    res = client.post("/api/runs", json=SUBMISSION)
    assert res.status_code == 201
    run_id = res.json()["run_id"]

    body = wait_for_status(client, run_id, {"completed"})
    assert body["name"] == "test run"
    assert body["model"] == "openai-chat-completions"
    assert body["tasks"] == ["abercrombie"]
    assert body["started_at"] and body["finished_at"]
    assert fake_worker.calls == [run_id]

    listed = client.get("/api/runs").json()["runs"]
    assert [r["run_id"] for r in listed] == [run_id]


def test_submit_validation(client):
    bad = copy.deepcopy(SUBMISSION)
    bad["config"]["tasks"] = []
    assert client.post("/api/runs", json=bad).status_code == 422

    bad = copy.deepcopy(SUBMISSION)
    bad["config"]["retrieval"] = {"retrieve_docs": True}
    assert client.post("/api/runs", json=bad).status_code == 422

    bad = copy.deepcopy(SUBMISSION)
    bad["config"]["retrieval"] = {
        "retrieve_docs": False,
        "rerank": True,
        "reranker": "rerankers",
        "reranker_args": "reranker_type=colbert",
    }
    assert client.post("/api/runs", json=bad).status_code == 422


def test_failed_run_records_error(client, fake_worker):
    fake_worker.fail_with = RuntimeError("boom")
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    body = wait_for_status(client, run_id, {"failed"})
    assert "boom" in body["error"]


def test_get_unknown_run_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_delete_completed_run(client):
    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    wait_for_status(client, run_id, {"completed"})
    assert client.delete(f"/api/runs/{run_id}").status_code == 204
    assert client.get(f"/api/runs/{run_id}").status_code == 404


def test_runs_ordered_newest_first(client):
    first = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    wait_for_status(client, first, {"completed"})
    second = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    wait_for_status(client, second, {"completed"})
    listed = client.get("/api/runs").json()["runs"]
    assert [r["run_id"] for r in listed] == [second, first]
