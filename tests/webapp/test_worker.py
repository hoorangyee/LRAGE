"""End-to-end worker tests: the default worker path with a fake
simple_evaluate, exercising the real EvaluationTracker persistence, progress
capture, env scoping, and output discovery."""
import json
import logging
import os

import pytest
from fastapi.testclient import TestClient
from tqdm import tqdm

from lrage.webapp.app import create_app
from lrage.webapp.jobs import worker
from tests.webapp.conftest import SUBMISSION, wait_for_status

CANNED_SAMPLE = {
    "doc_id": 0,
    "doc": {"text": "Is 'Apple' a generic mark for computers?"},
    "target": "suggestive",
    "arguments": [["Q: Is 'Apple' generic?\nA:"]],
    "resps": [["suggestive"]],
    "filtered_resps": ["suggestive"],
    "doc_hash": "d" * 64,
    "prompt_hash": "p" * 64,
    "target_hash": "t" * 64,
    "acc": 1.0,
}


class FakeSimpleEvaluate:
    """Mimics lrage.evaluator.simple_evaluate closely enough to exercise the
    worker: logs real phase messages, runs a real tqdm loop, records
    experiment args on the tracker, and returns a results dict."""

    def __init__(self):
        self.kwargs = None
        self.env_seen = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.env_seen = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
        }
        logger = logging.getLogger("lrage")
        logger.info(
            "Initializing %s model, with arguments: {}", kwargs["model"]
        )
        logger.info("Building contexts for abercrombie_tiny on rank 0...")
        logger.info("Running generate_until requests")
        for _ in tqdm(range(4), total=4, desc="Running requests"):
            pass
        tracker = kwargs["evaluation_tracker"]
        tracker.general_config_tracker.log_experiment_args(
            model_source=kwargs["model"],
            model_args=kwargs["model_args"],
            system_instruction=kwargs.get("system_instruction"),
            chat_template=None,
            fewshot_as_multiturn=kwargs.get("fewshot_as_multiturn", False),
        )
        return {
            "results": {
                "abercrombie_tiny": {
                    "alias": "abercrombie_tiny",
                    "acc,none": 0.75,
                    "acc_stderr,none": 0.05,
                }
            },
            "configs": {"abercrombie_tiny": {"task": "abercrombie_tiny"}},
            "versions": {"abercrombie_tiny": 1.0},
            "n-shot": {"abercrombie_tiny": 0},
            "samples": {"abercrombie_tiny": [dict(CANNED_SAMPLE)]},
        }


@pytest.fixture
def fake_evaluate(monkeypatch):
    fake = FakeSimpleEvaluate()
    monkeypatch.setattr(worker, "simple_evaluate", fake)
    return fake


@pytest.fixture
def real_worker_client(settings, stub_tasks, fake_evaluate):
    app = create_app(settings)  # default worker -> jobs.worker.run_evaluation
    with TestClient(app) as c:
        yield c


def submission_with_keys():
    sub = json.loads(json.dumps(SUBMISSION))
    sub["config"]["tasks"] = ["abercrombie_tiny"]
    sub["api_keys"] = {"openai_api_key": "sk-test-123", "hf_token": "hf-test"}
    return sub


def test_worker_end_to_end(real_worker_client, fake_evaluate, settings):
    client = real_worker_client
    run_id = client.post("/api/runs", json=submission_with_keys()).json()["run_id"]
    body = wait_for_status(client, run_id, {"completed", "failed"}, timeout=15)
    assert body["status"] == "completed", body.get("error")

    # Kwargs faithfully translated from RunConfig.
    kw = fake_evaluate.kwargs
    assert kw["model"] == "openai-chat-completions"
    assert kw["model_args"] == "model=gpt-4o-mini"
    assert kw["tasks"] == ["abercrombie_tiny"]
    assert kw["batch_size"] == "auto"
    assert kw["retrieve_docs"] is False
    assert kw["gen_kwargs"] is None or isinstance(kw["gen_kwargs"], str)

    # API keys visible during the run, restored afterwards.
    assert fake_evaluate.env_seen["OPENAI_API_KEY"] == "sk-test-123"
    assert fake_evaluate.env_seen["HF_TOKEN"] == "hf-test"
    assert os.environ.get("OPENAI_API_KEY") != "sk-test-123"

    # Tracker wrote results under the per-run directory and the repo indexed them.
    run_dir = settings.output_root / run_id
    results_files = list(run_dir.glob("**/results_*.json"))
    samples_files = list(run_dir.glob("**/samples_*.jsonl"))
    assert len(results_files) == 1
    assert len(samples_files) == 1

    from lrage.webapp.store.runs_repo import RunsRepository

    record = RunsRepository(settings.db_path).get(run_id)
    assert record["results_path"] == str(results_files[0])
    assert record["samples_paths"] == [str(samples_files[0])]
    assert record["summary"] == [
        {
            "task": "abercrombie_tiny",
            "n_shot": "0",
            "metric": "acc",
            "value": 0.75,
            "stderr": 0.05,
        }
    ]

    # The saved samples file round-trips.
    with open(samples_files[0]) as f:
        saved_sample = json.loads(f.readline())
    assert saved_sample["target"] == "suggestive"


def test_worker_streams_phases_and_progress(real_worker_client, fake_evaluate):
    client = real_worker_client
    run_id = client.post("/api/runs", json=submission_with_keys()).json()["run_id"]
    wait_for_status(client, run_id, {"completed"}, timeout=15)

    with client.stream("GET", f"/api/runs/{run_id}/events") as res:
        text = "".join(res.iter_text())
    assert '"phase": "loading_model"' in text.replace('", "', '", "') or "loading_model" in text
    assert "running_requests" in text
    assert '"type": "progress"' in text or '"type":"progress"' in text
    assert "done" in text


def test_gen_kwargs_serialization(real_worker_client, fake_evaluate):
    client = real_worker_client
    sub = submission_with_keys()
    sub["config"]["gen_kwargs"] = {
        "max_gen_toks": 256,
        "temperature": 0.2,
        "do_sample": True,
    }
    run_id = client.post("/api/runs", json=sub).json()["run_id"]
    wait_for_status(client, run_id, {"completed"}, timeout=15)
    assert (
        fake_evaluate.kwargs["gen_kwargs"]
        == "max_gen_toks=256,temperature=0.2,do_sample=True"
    )
