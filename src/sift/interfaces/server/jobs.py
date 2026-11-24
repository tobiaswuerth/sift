import itertools
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from sift.api.errors import JobStoppedError, NotFoundError
from sift.api.events import Advanced, CollectionStarted, Event, Reporter
from sift.api.jobs import ServedJob

STOPPED = "the job was stopped: what it wrote so far is kept"
STOPPED_REMEDY = "run it again to go on from there"
JOBS_REMEDY = "see the jobs that run or wait with sift jobs"


def _numbers() -> Iterator[int]:
    return itertools.count(1)


@dataclass(slots=True, eq=False)
class HeldJob:
    number: int
    command: str
    started: float
    stop: threading.Event = field(default_factory=threading.Event)
    running: bool = False
    watched: bool = True
    collection: str | None = None
    done: int | None = None


@dataclass(slots=True, eq=False)
class Jobs:
    clock: Callable[[], float] = time.perf_counter
    lock: threading.Lock = field(default_factory=threading.Lock)
    held: dict[int, HeldJob] = field(default_factory=dict[int, HeldJob])
    numbers: Iterator[int] = field(default_factory=_numbers)

    def queued(self, command: str) -> HeldJob:
        with self.lock:
            job = HeldJob(next(self.numbers), command, self.clock())
            self.held[job.number] = job
            return job

    def ahead(self, job: HeldJob) -> tuple[HeldJob, ...]:
        with self.lock:
            return tuple(other for number, other in sorted(self.held.items()) if number < job.number)

    def began(self, job: HeldJob) -> None:
        if job.stop.is_set():
            raise JobStoppedError(STOPPED, STOPPED_REMEDY)
        job.running = True

    def ended(self, job: HeldJob) -> None:
        with self.lock:
            _ = self.held.pop(job.number, None)

    def reporting(self, job: HeldJob, report: Reporter) -> Reporter:
        def said(event: Event) -> None:
            if job.stop.is_set():
                raise JobStoppedError(STOPPED, STOPPED_REMEDY)
            if isinstance(event, CollectionStarted):
                job.collection, job.done = event.name, None
            elif isinstance(event, Advanced):
                job.done = event.done
            report(event)

        return said

    def _shown(self, job: HeldJob) -> ServedJob:
        seconds = self.clock() - job.started
        return ServedJob(job.number, job.command, job.running, seconds, job.watched, job.collection, job.done)

    def listed(self) -> tuple[ServedJob, ...]:
        with self.lock:
            held = [job for _, job in sorted(self.held.items())]
        return tuple(map(self._shown, held))

    def stopped(self, number: int) -> ServedJob:
        with self.lock:
            job = self.held.get(number)
        if job is None:
            raise NotFoundError(f"there is no job {number}", JOBS_REMEDY)
        job.stop.set()
        return self._shown(job)

    def stop_all(self) -> tuple[HeldJob, ...]:
        with self.lock:
            held = tuple(job for job in self.held.values() if not job.stop.is_set())
        for job in held:
            job.stop.set()
        return held
