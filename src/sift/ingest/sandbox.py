import multiprocessing
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from typing import Protocol, cast

from sift.domain.text.documents import Outcome, Refused
from sift.ingest.readers import ToRead, read_file

type ReadOne = Callable[[ToRead], Outcome]
type FileReading = Callable[[Sequence[ToRead]], Iterator[tuple[int, Outcome]]]
type OpenReading = Callable[[], AbstractContextManager[FileReading]]
type Task = tuple[int, ToRead, float]
type Heard = tuple[int, object]

ENDED = "its reader ended the process it ran in"
RAISED = 1
READY = "ready"
MOST_WORKERS = 30
STOP_SECONDS = 10.0
HEARD_SECONDS = 1.0


class ReaderFailedError(Exception):
    pass


class _Channel(Protocol):
    def send(self, obj: object) -> None: ...

    def recv(self) -> object: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Patience:
    base_seconds: float = 600.0
    bytes_a_second: float = 100_000.0
    start_seconds: float = 120.0

    def seconds(self, size: int) -> float:
        return self.base_seconds + size / self.bytes_a_second


PATIENCE = Patience()


def came_to_no_end(seconds: float) -> Refused:
    return Refused(f"its reader came to no end in {seconds:,.0f} seconds", read_again=False)


def _message(channel: _Channel) -> object:
    try:
        return channel.recv()
    except EOFError:
        return None


def _serve(asked: _Channel, answers: _Channel, read: ReadOne) -> None:
    _ = signal.signal(signal.SIGINT, signal.SIG_IGN)
    answers.send(READY)
    while (each := _message(asked)) is not None:
        answers.send(read(cast("ToRead", each)))


@dataclass(frozen=True, slots=True)
class _Greeting:
    said: threading.Event = field(default_factory=threading.Event)
    settled: threading.Event = field(default_factory=threading.Event)


def _listened(number: int, channel: _Channel, greeting: _Greeting, heard: "queue.Queue[Heard]") -> None:
    while (message := _message(channel)) is not None:
        if message == READY:
            greeting.said.set()
            greeting.settled.set()
        else:
            heard.put((number, message))
    greeting.settled.set()
    heard.put((number, None))


@dataclass(slots=True)
class _Worker:
    number: int
    process: BaseProcess
    asked: _Channel
    answers: _Channel
    greeting: _Greeting
    listener: threading.Thread
    task: Task | None = None


def _stopped(worker: _Worker) -> None:
    worker.asked.close()
    if worker.task is not None or not worker.greeting.said.is_set():
        worker.process.kill()
    worker.process.join(STOP_SECONDS)
    worker.listener.join(STOP_SECONDS)
    worker.answers.close()


def _failed(asked: ToRead, exit_code: int | None) -> Outcome:
    if exit_code == RAISED:
        raise ReaderFailedError(f"reading {asked.path} raised an error in its process; its traceback is above")
    return Refused(ENDED, read_again=False)


@dataclass(slots=True)
class _Pool:
    read: ReadOne
    patience: Patience
    most: int
    clock: Callable[[], float]
    workers: dict[int, _Worker] = field(default_factory=dict[int, _Worker])
    heard: "queue.Queue[Heard]" = field(default_factory=queue.Queue[Heard])
    made: int = 0

    def _started(self, count: int) -> None:
        context = multiprocessing.get_context("spawn")
        started: list[_Worker] = []
        for _ in range(count):
            asked_there, asked = context.Pipe(duplex=False)
            answers, answers_there = context.Pipe(duplex=False)
            process = context.Process(target=_serve, args=(asked_there, answers_there, self.read), daemon=True)
            process.start()
            asked_there.close()
            answers_there.close()
            self.made += 1
            greeting = _Greeting()
            listener = threading.Thread(target=_listened, args=(self.made, answers, greeting, self.heard), daemon=True)
            listener.start()
            self.workers[self.made] = worker = _Worker(self.made, process, asked, answers, greeting, listener)
            started.append(worker)
        deadline = self.clock() + self.patience.start_seconds
        for worker in started:
            if not worker.greeting.settled.wait(max(0.0, deadline - self.clock())):
                raise ReaderFailedError(f"a reader process did not start in {self.patience.start_seconds:,.0f} seconds")
            if not worker.greeting.said.is_set():
                worker.process.join(STOP_SECONDS)
                raise ReaderFailedError(f"a reader process ended as it started, exit code {worker.process.exitcode}")

    def _dropped(self, worker: _Worker) -> None:
        _ = self.workers.pop(worker.number)
        _stopped(worker)

    def _given(self, todo: deque[tuple[int, ToRead]]) -> None:
        for worker in [each for each in self.workers.values() if each.task is None][: len(todo)]:
            at, asked = todo.popleft()
            worker.asked.send(asked)
            worker.task = at, asked, self.clock() + self.patience.seconds(asked.size)

    def _overdue(self) -> Iterator[tuple[int, Outcome]]:
        now = self.clock()
        for worker in list(self.workers.values()):
            if worker.task is not None and now >= worker.task[2]:
                at, asked, _ = worker.task
                self._dropped(worker)
                yield at, came_to_no_end(self.patience.seconds(asked.size))

    def _answer(self, number: int, message: object) -> tuple[int, Outcome] | None:
        worker = self.workers.get(number)
        if worker is None or worker.task is None:
            return None
        at, asked, _ = worker.task
        if message is not None:
            worker.task = None
            return at, cast("Outcome", message)
        worker.task = None
        worker.process.join(STOP_SECONDS)
        exit_code = worker.process.exitcode
        self._dropped(worker)
        return at, _failed(asked, exit_code)

    def _waited(self) -> float:
        nearest = min(task[2] for worker in self.workers.values() if (task := worker.task) is not None)
        return min(HEARD_SECONDS, max(0.0, nearest - self.clock()))

    def files(self, asked: Sequence[ToRead]) -> Iterator[tuple[int, Outcome]]:
        todo = deque(enumerate(asked))
        while todo or any(worker.task is not None for worker in self.workers.values()):
            busy = sum(worker.task is not None for worker in self.workers.values())
            if (wanted := min(self.most, busy + len(todo)) - len(self.workers)) > 0:
                self._started(wanted)
            self._given(todo)
            try:
                number, message = self.heard.get(timeout=self._waited())
            except queue.Empty:
                yield from self._overdue()
                continue
            if (answer := self._answer(number, message)) is not None:
                yield answer

    def ended(self) -> None:
        for worker in list(self.workers.values()):
            self._dropped(worker)


@contextmanager
def sandboxed(
    read: ReadOne = read_file,
    patience: Patience = PATIENCE,
    workers: int = MOST_WORKERS,
    clock: Callable[[], float] = time.perf_counter,
) -> Generator[FileReading]:
    if workers < 1:
        raise ValueError(f"a sandbox of {workers} reader processes reads nothing: give it one at least")
    pool = _Pool(read, patience, min(workers, max(1, (os.cpu_count() or 2) - 1)), clock)
    try:
        yield pool.files
    finally:
        pool.ended()
