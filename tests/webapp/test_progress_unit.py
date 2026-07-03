import logging

import pytest
from tqdm import tqdm

from lrage.webapp.jobs.cancellation import CancelToken, JobCancelled
from lrage.webapp.jobs.progress import (
    capture_lrage_logs,
    match_phase,
    tqdm_progress_hook,
)


@pytest.mark.parametrize(
    "message,phase",
    [
        ("Initializing openai-chat-completions model, with arguments: {}", "loading_model"),
        ("Initializing pyserini retriever, with arguments: {...}", "loading_retriever"),
        ("Retrieving docs...", "retrieving"),
        ("Reranking docs for abercrombie", "reranking"),
        ("Building contexts for abercrombie on rank 0...", "building_requests"),
        ("Running generate_until requests", "running_requests"),
        ("Saving results aggregated", "saving"),
        ("Setting random seed to 0", None),
    ],
)
def test_match_phase(message, phase):
    assert match_phase(message) == phase


def test_log_capture_publishes_phase_and_log():
    events = []
    with capture_lrage_logs(events.append):
        logging.getLogger("lrage").info("Running generate_until requests")
        logging.getLogger("lrage").warning("something odd")
    types = [(e["type"], e.get("phase") or e.get("message")) for e in events]
    assert ("phase", "running_requests") in types
    assert ("log", "something odd") in types
    # Handler removed after the context exits.
    events.clear()
    logging.getLogger("lrage").info("Running generate_until requests")
    assert events == []


def test_tqdm_hook_publishes_and_restores():
    events = []
    token = CancelToken()
    orig_update, orig_close = tqdm.update, tqdm.close
    with tqdm_progress_hook(events.append, token, min_interval=0):
        for _ in tqdm(range(5), total=5, desc="Running requests"):
            pass
    progress = [e for e in events if e["type"] == "progress"]
    assert progress, "no progress events published"
    assert progress[-1]["n"] == 5
    assert progress[-1]["total"] == 5
    assert progress[-1]["pct"] == 100.0
    assert tqdm.update is orig_update and tqdm.close is orig_close


def test_tqdm_hook_raises_on_cancel():
    # The contract: cancellation is checked at every tqdm.update() call —
    # model backends call update() per generation batch.
    token = CancelToken()
    token.cancel()
    with tqdm_progress_hook(lambda e: None, token):
        bar = tqdm(total=3, disable=True)
        with pytest.raises(JobCancelled):
            bar.update(1)
    # Hook restored even after the exception.
    bar = tqdm(total=1, disable=True)
    bar.update(1)
    bar.close()
