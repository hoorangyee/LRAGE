import json

import pytest

from lrage.webapp.store import samples_reader
from tests.webapp.conftest import wait_for_status
from tests.webapp.test_worker import (  # noqa: F401 (fixtures)
    fake_evaluate,
    real_worker_client,
    submission_with_keys,
)


def test_task_name_for():
    assert (
        samples_reader.task_name_for(
            "/x/samples_abercrombie_tiny_2026-07-03T14-50-12.123456.jsonl"
        )
        == "abercrombie_tiny"
    )


def test_metric_values_excludes_structural_keys():
    sample = {"doc_id": 3, "acc": 1.0, "Rating": 4, "target": "x", "f1": 0.5}
    assert samples_reader.metric_values(sample) == {"acc": 1.0, "f1": 0.5}


def test_filter_samples():
    samples = [
        {"doc_id": 0, "doc": {"text": "alpha"}, "target": "a", "filtered_resps": ["a"], "acc": 1.0, "Rating": 5},
        {"doc_id": 1, "doc": {"text": "beta"}, "target": "b", "filtered_resps": ["c"], "acc": 0.0, "Rating": 2},
    ]
    assert len(samples_reader.filter_samples(samples, q="beta")) == 1
    assert len(samples_reader.filter_samples(samples, incorrect_only=True)) == 1
    assert samples_reader.filter_samples(samples, incorrect_only=True)[0]["doc_id"] == 1
    assert len(samples_reader.filter_samples(samples, judge_min=4)) == 1
    assert len(samples_reader.filter_samples(samples, judge_max=3)) == 1


@pytest.fixture
def completed_run(real_worker_client):
    client = real_worker_client
    run_id = client.post("/api/runs", json=submission_with_keys()).json()["run_id"]
    wait_for_status(client, run_id, {"completed"}, timeout=15)
    return client, run_id


def test_samples_endpoint(completed_run):
    client, run_id = completed_run
    body = client.get(f"/api/runs/{run_id}/samples").json()
    assert body["tasks"] == ["abercrombie_tiny"]
    assert body["task"] == "abercrombie_tiny"
    assert body["total"] == 1
    item = body["items"][0]
    assert item["doc_id"] == 0
    assert item["target"] == "suggestive"
    assert item["resp"] == "suggestive"
    assert item["metrics"] == {"acc": 1.0}


def test_samples_pagination_and_filters(completed_run):
    client, run_id = completed_run
    body = client.get(
        f"/api/runs/{run_id}/samples", params={"page": 2, "page_size": 1}
    ).json()
    assert body["total"] == 1
    assert body["items"] == []

    body = client.get(
        f"/api/runs/{run_id}/samples", params={"q": "no-such-text"}
    ).json()
    assert body["total"] == 0

    body = client.get(
        f"/api/runs/{run_id}/samples", params={"incorrect_only": True}
    ).json()
    assert body["total"] == 0  # the canned sample is correct


def test_sample_detail(completed_run):
    client, run_id = completed_run
    body = client.get(f"/api/runs/{run_id}/samples/abercrombie_tiny/0").json()
    assert body["sample"]["doc_id"] == 0
    assert body["sample"]["target"] == "suggestive"
    assert body["metrics"] == {"acc": 1.0}

    assert (
        client.get(f"/api/runs/{run_id}/samples/abercrombie_tiny/99").status_code
        == 404
    )
    assert client.get(f"/api/runs/{run_id}/samples/nope/0").status_code == 404


def test_samples_empty_for_run_without_files(client):
    from tests.webapp.conftest import SUBMISSION

    run_id = client.post("/api/runs", json=SUBMISSION).json()["run_id"]
    wait_for_status(client, run_id, {"completed"})
    body = client.get(f"/api/runs/{run_id}/samples").json()
    assert body == {"tasks": [], "task": None, "total": 0, "page": 1, "items": []}
