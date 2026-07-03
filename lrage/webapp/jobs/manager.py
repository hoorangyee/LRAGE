import logging
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from lrage.webapp.jobs.cancellation import CancelToken, JobCancelled
from lrage.webapp.jobs.events import EventBuffer
from lrage.webapp.schemas import ProgressSnapshot, RunStatus, RunSubmission
from lrage.webapp.settings import Settings
from lrage.webapp.store.runs_repo import RunsRepository

logger = logging.getLogger("lrage.webapp")

QUEUE_CAP = 20


class QueueFull(Exception):
    pass


@dataclass
class Job:
    run_id: str
    submission: RunSubmission  # api_keys live only here, never in the DB
    token: CancelToken = field(default_factory=CancelToken)
    events: EventBuffer = field(default_factory=EventBuffer)
    status: RunStatus = RunStatus.queued
    progress: ProgressSnapshot = field(default_factory=ProgressSnapshot)


class JobContext:
    """What a worker function gets to talk back with."""

    def __init__(self, job: Job, repo: RunsRepository, settings: Settings) -> None:
        self.job = job
        self.repo = repo
        self.settings = settings
        self.token = job.token

    def publish(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "phase":
            self.job.progress.phase = event.get("phase")
            self.job.progress.desc = None
            self.job.progress.n = None
            self.job.progress.total = None
            self.job.progress.pct = None
        elif etype == "progress":
            self.job.progress.phase = event.get("phase", self.job.progress.phase)
            self.job.progress.desc = event.get("desc")
            self.job.progress.n = event.get("n")
            self.job.progress.total = event.get("total")
            self.job.progress.pct = event.get("pct")
        self.job.events.publish(event)


# A worker takes a JobContext, runs the evaluation, and returns when done.
WorkerFn = Callable[[JobContext], None]


def _default_worker(ctx: JobContext) -> None:
    # Imported lazily: pulls in the full evaluation stack (torch etc.).
    from lrage.webapp.jobs.worker import run_evaluation

    run_evaluation(ctx)


class JobManager:
    """Single daemon worker thread consuming a FIFO queue: one GPU, one run
    at a time. The tqdm/log hooks in jobs.progress patch process-global state
    and rely on this single-flight guarantee."""

    def __init__(
        self,
        repo: RunsRepository,
        settings: Settings,
        worker_fn: Optional[WorkerFn] = None,
    ) -> None:
        self._repo = repo
        self._settings = settings
        self._worker_fn = worker_fn or _default_worker
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._jobs: Dict[str, Job] = {}
        self._pending: List[str] = []  # run_ids waiting, in order
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._worker_loop, name="lrage-eval-worker", daemon=True
            )
            self._thread.start()

    def submit(self, submission: RunSubmission) -> Job:
        with self._lock:
            if len(self._pending) >= QUEUE_CAP:
                raise QueueFull()
            run_id = uuid.uuid4().hex[:12]
            job = Job(run_id=run_id, submission=submission)
            self._jobs[run_id] = job
            self._pending.append(run_id)
        config = submission.config.model_dump(mode="json")
        self._repo.create(
            run_id,
            submission.name,
            config,
            submission.config.model,
            submission.config.tasks,
        )
        job.events.publish({"type": "status", "status": RunStatus.queued.value})
        self._queue.put(job)
        return job

    def get(self, run_id: str) -> Optional[Job]:
        return self._jobs.get(run_id)

    def queue_position(self, run_id: str) -> Optional[int]:
        with self._lock:
            if run_id in self._pending:
                return self._pending.index(run_id) + 1
        return None

    def is_active(self, run_id: str) -> bool:
        job = self._jobs.get(run_id)
        return job is not None and not job.status.is_terminal

    def cancel(self, run_id: str) -> Optional[RunStatus]:
        """Returns the resulting status, or None if the run is not live."""
        job = self._jobs.get(run_id)
        if job is None or job.status.is_terminal:
            return None
        with self._lock:
            queued = run_id in self._pending
            if queued:
                self._pending.remove(run_id)
        job.token.cancel()
        if queued:
            # The worker loop will skip it; finalize now for instant feedback.
            self._finalize(job, RunStatus.cancelled)
            return RunStatus.cancelled
        job.status = RunStatus.cancelling
        self._repo.set_status(run_id, RunStatus.cancelling)
        job.events.publish(
            {"type": "status", "status": RunStatus.cancelling.value}
        )
        return RunStatus.cancelling

    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job.status.is_terminal:  # cancelled while queued
                continue
            with self._lock:
                if job.run_id in self._pending:
                    self._pending.remove(job.run_id)
            if job.token.is_cancelled():
                self._finalize(job, RunStatus.cancelled)
                continue

            job.status = RunStatus.running
            self._repo.set_status(job.run_id, RunStatus.running)
            job.events.publish({"type": "status", "status": RunStatus.running.value})

            ctx = JobContext(job, self._repo, self._settings)
            try:
                self._worker_fn(ctx)
            except JobCancelled:
                self._finalize(job, RunStatus.cancelled)
            except Exception:
                tb = traceback.format_exc()
                logger.error("run %s failed:\n%s", job.run_id, tb)
                self._repo.set_error(job.run_id, tb[-4000:])
                self._finalize(job, RunStatus.failed)
            else:
                self._finalize(job, RunStatus.completed)

    def _finalize(self, job: Job, status: RunStatus) -> None:
        job.status = status
        self._repo.set_status(job.run_id, status)
        job.events.publish({"type": "status", "status": status.value})
        job.events.publish({"type": "done", "status": status.value})
        job.events.close()
