"""Progress capture: phases from the "lrage" logger, percentages from tqdm.

The tqdm hook patches methods on the shared tqdm class object, which is safe
only because JobManager runs one evaluation at a time in this process.

Note: the log handler cannot be a cancellation checkpoint — logging swallows
exceptions raised in Handler.emit(). Cancellation is checked in the tqdm hook,
which runs in the worker's own call stack.
"""
import logging
import time
from contextlib import contextmanager
from typing import Callable, Dict, Optional

from lrage.webapp.jobs.cancellation import CancelToken, JobCancelled

Publish = Callable[[Dict], None]

# Ordered substring rules mapping eval_logger messages to phases.
_PHASE_RULES = [
    ("retriever", "loading_retriever"),
    ("reranker", "loading_reranker"),
    ("Initializing", "loading_model"),  # "Initializing {model} model"
    ("Retrieving docs", "retrieving"),
    ("Reranking docs", "reranking"),
    ("Building contexts", "building_requests"),
    ("Running", "running_requests"),  # "Running {generate_until} requests"
    ("Saving results", "saving"),
]

PHASES = [
    "loading_model",
    "loading_retriever",
    "loading_reranker",
    "retrieving",
    "reranking",
    "building_requests",
    "running_requests",
    "judging",
    "saving",
]


def match_phase(message: str) -> Optional[str]:
    for needle, phase in _PHASE_RULES:
        if needle in message:
            if phase in ("loading_retriever", "loading_reranker") and (
                "Initializing" not in message
            ):
                continue
            return phase
    return None


class LrageLogHandler(logging.Handler):
    def __init__(self, publish: Publish) -> None:
        super().__init__(level=logging.INFO)
        self._publish = publish

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        phase = match_phase(message)
        if phase:
            self._publish({"type": "phase", "phase": phase})
        self._publish(
            {"type": "log", "level": record.levelname, "message": message}
        )


@contextmanager
def capture_lrage_logs(publish: Publish):
    logger = logging.getLogger("lrage")
    handler = LrageLogHandler(publish)
    logger.addHandler(handler)
    # Phase detection needs INFO records; don't rely on simple_evaluate
    # setting the level before the interesting messages fire.
    old_level = logger.level
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


@contextmanager
def tqdm_progress_hook(publish: Publish, token: CancelToken, min_interval: float = 0.4):
    """Patch tqdm.update/close to publish throttled progress events and check
    for cancellation between iterations."""
    from tqdm import tqdm

    orig_update = tqdm.update
    orig_close = tqdm.close
    last_emit = [0.0]

    def publish_bar(bar, force=False):
        now = time.monotonic()
        finished = bar.total is not None and bar.n >= bar.total
        if not force and not finished and now - last_emit[0] < min_interval:
            return
        last_emit[0] = now
        total = int(bar.total) if bar.total else None
        publish(
            {
                "type": "progress",
                "desc": bar.desc or None,
                "n": int(bar.n),
                "total": total,
                "pct": round(100.0 * bar.n / total, 1) if total else None,
            }
        )

    def update(self, n=1):
        result = orig_update(self, n)
        if token.is_cancelled():
            raise JobCancelled()
        publish_bar(self)
        return result

    def close(self):
        orig_close(self)
        # Emit the final count so the bar doesn't appear stuck mid-way.
        if getattr(self, "n", None):
            publish_bar(self, force=True)

    tqdm.update = update
    tqdm.close = close
    try:
        yield
    finally:
        tqdm.update = orig_update
        tqdm.close = orig_close
