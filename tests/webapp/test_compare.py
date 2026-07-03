import json

from tests.webapp.conftest import SUBMISSION, wait_for_status


def submit_completed(client, name):
    sub = json.loads(json.dumps(SUBMISSION))
    sub["name"] = name
    run_id = client.post("/api/runs", json=sub).json()["run_id"]
    wait_for_status(client, run_id, {"completed"})
    return run_id


def test_compare(client, settings):
    # FakeWorker doesn't write summaries; inject them via the repo.
    from lrage.webapp.store.runs_repo import RunsRepository

    repo = RunsRepository(settings.db_path)
    a = submit_completed(client, "run a")
    b = submit_completed(client, "run b")
    repo.set_outputs(a, "x", None, [], [
        {"task": "t1", "n_shot": "0", "metric": "acc", "value": 0.8, "stderr": 0.01}
    ])
    repo.set_outputs(b, "y", None, [], [
        {"task": "t1", "n_shot": "0", "metric": "acc", "value": 0.6, "stderr": 0.02},
        {"task": "t2", "n_shot": "0", "metric": "f1", "value": 0.5, "stderr": None},
    ])

    body = client.get(f"/api/compare?run_ids={a},{b}").json()
    assert [r["run_id"] for r in body["runs"]] == [a, b]
    assert body["runs"][0]["config"]["model"] == "openai-chat-completions"

    table = {(row["task"], row["metric"]): row["values"] for row in body["table"]}
    assert table[("t1", "acc")][a]["value"] == 0.8
    assert table[("t1", "acc")][b]["value"] == 0.6
    assert a not in table[("t2", "f1")]
    assert table[("t2", "f1")][b]["value"] == 0.5


def test_compare_validations(client):
    a = submit_completed(client, "solo")
    assert client.get(f"/api/compare?run_ids={a}").status_code == 422
    assert client.get(f"/api/compare?run_ids={a},nope").status_code == 404
