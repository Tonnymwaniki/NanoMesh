"""Long-running work (downloads, benchmarks) started from the MCP server.

A tool call has to answer quickly, so these run in a background thread: the
tool returns a job id at once and the agent checks on it with job_status.
Jobs live as long as the MCP server process; downloads resume if restarted.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable

from pydantic import BaseModel

from nanomesh.results import now


class Job(BaseModel):
    id: str
    kind: str
    description: str
    status: str = "running"  # running | done | failed | cancelled
    progress: float | None = None  # 0-1 when known
    message: str | None = None
    eta_s: int | None = None  # seconds left, when known
    result: dict | None = None
    error: str | None = None
    started: str
    finished: str | None = None


class Handle:
    """What a job's function uses to report progress and notice cancellation."""

    def __init__(self, job: Job):
        self.job = job
        self.cancel = threading.Event()
        self.done = threading.Event()

    def update(self, progress: float | None = None, message: str | None = None, eta_s: float | None = None) -> None:
        if progress is not None:
            self.job.progress = round(min(max(progress, 0.0), 1.0), 3)
        if message is not None:
            self.job.message = message
        if eta_s is not None:
            self.job.eta_s = int(eta_s)


_jobs: dict[str, Handle] = {}
_ids = itertools.count(1)
_lock = threading.Lock()


def start(kind: str, description: str, fn: Callable[[Handle], dict]) -> Job:
    with _lock:
        job = Job(id=f"{kind}-{next(_ids)}", kind=kind, description=description, started=now())
        handle = _jobs[job.id] = Handle(job)

    def run():
        try:
            job.result = fn(handle)
            job.status = "cancelled" if handle.cancel.is_set() else "done"
            if job.status == "done":
                job.progress = 1.0
        except Exception as e:  # noqa: BLE001 - reported to the agent, not raised
            job.status = "cancelled" if handle.cancel.is_set() else "failed"
            job.error = str(e)
        job.eta_s = None
        job.finished = now()
        handle.done.set()

    threading.Thread(target=run, name=job.id, daemon=True).start()
    return job


def get(job_id: str, wait_s: float = 0) -> Job | None:
    """The job; with wait_s, first wait up to that long for it to finish."""
    h = _jobs.get(job_id)
    if h and wait_s > 0:
        h.done.wait(wait_s)
    return h.job if h else None


def all_jobs() -> list[Job]:
    return [h.job for h in _jobs.values()]


def cancel(job_id: str) -> Job | None:
    h = _jobs.get(job_id)
    if h and h.job.status == "running":
        h.cancel.set()
    return h.job if h else None
