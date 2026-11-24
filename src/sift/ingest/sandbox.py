import multiprocessing
import multiprocessing.connection
import os
import pickle
import queue
import signal
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import NoReturn, Protocol, cast

from sift.domain.text.documents import Outcome, Refused, stopped
from sift.ingest.pictures import Pictured, ToPicture, picture
from sift.ingest.readers import ToRead, read_file


class Request(Protocol):
    @property
    def path(self) -> Path: ...

    @property
    def size(self) -> int: ...


type Answering[A, B] = Callable[[Sequence[A]], Iterator[tuple[int, B | Refused]]]
type ReadOne = Callable[[ToRead], Outcome]
type FileReading = Answering[ToRead, Outcome]
type OpenReading = Callable[[], AbstractContextManager[FileReading]]
type Picturing = Answering[ToPicture, Pictured]
type OpenPicturing = Callable[[], AbstractContextManager[Picturing]]
type Heard = tuple[int, object]

ENDED = "its reader ended the process it ran in"
READY = "ready"
BEGAN = "began"
ORPHANED = 4
MOST_WORKERS = 30
MOST_ANSWER_BYTES = 2 << 30
HEARD_SECONDS = 1.0


class ReaderFailedError(Exception):
    pass


class _Channel(Protocol):
    def send(self, obj: object) -> None: ...

    def send_bytes(self, buf: bytes) -> None: ...

    def recv(self) -> object: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Patience:
    base_seconds: float = 600.0
    bytes_a_second: float = 100_000.0
    start_seconds: float = 120.0
    stop_seconds: float = 10.0

    def seconds(self, size: int) -> float:
        return self.base_seconds + size / self.bytes_a_second


PATIENCE = Patience()


@dataclass(frozen=True, slots=True)
class _Raised:
    error: str


@dataclass(frozen=True, slots=True)
class _Task[A]:
    at: int
    asked: A
    deadline: float


def came_to_no_end(seconds: float) -> Refused:
    return stopped(f"its reader came to no end in {seconds:,.0f} seconds")


def too_large_an_answer(size: int, most: int) -> Refused:
    return Refused(
        f"read, it comes to {size:,} bytes, more than the {most:,} sift takes from one file", read_again=False
    )


def usable_cpus() -> int:
    found = (
        os.process_cpu_count()
        if sys.version_info >= (3, 13)
        else len(os.sched_getaffinity(0))
        if sys.platform == "linux"
        else os.cpu_count()
    )
    return found or 1


def reader_processes(most: int, cpus: int | None) -> int:
    return min(most, max(1, (cpus or 2) - 1))


def _message(channel: _Channel) -> object:
    try:
        return channel.recv()
    except (EOFError, OSError):
        return None


def _ends_with(parent: BaseProcess) -> NoReturn:
    os._exit(ORPHANED * len(multiprocessing.connection.wait([parent.sentinel])))


def _pickled(outcome: object, most: int) -> bytes:
    held = pickle.dumps(outcome, pickle.HIGHEST_PROTOCOL)
    return held if len(held) <= most else pickle.dumps(too_large_an_answer(len(held), most), pickle.HIGHEST_PROTOCOL)


def _serve[A, B](asked: _Channel, answers: _Channel, read: Callable[[A], B], most: int) -> None:
    _ = signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = cast("BaseProcess", multiprocessing.parent_process())
    threading.Thread(target=_ends_with, args=(parent,), daemon=True).start()
    answers.send(READY)
    while (each := _message(asked)) is not None:
        answers.send(BEGAN)
        try:
            outcome = read(cast("A", each))
        except Exception as error:
            answers.send(_Raised("".join(traceback.format_exception_only(error)).strip()))
            raise
        answers.send_bytes(_pickled(outcome, most))


@dataclass(frozen=True, slots=True)
class _Greeting:
    said: threading.Event = field(default_factory=threading.Event)
    settled: threading.Event = field(default_factory=threading.Event)


def _listened(number: int, channel: _Channel, greeting: _Greeting, heard: "queue.Queue[Heard]") -> None:
    try:
        while (message := _message(channel)) is not None:
            if message == READY:
                greeting.said.set()
                greeting.settled.set()
            else:
                heard.put((number, message))
    finally:
        greeting.settled.set()
        heard.put((number, None))


@dataclass(slots=True)
class _Worker[A]:
    number: int
    process: BaseProcess
    asked: _Channel
    answers: _Channel
    greeting: _Greeting
    listener: threading.Thread
    task: _Task[A] | None = None
    began: bool = False


def _stopped[A](worker: _Worker[A], seconds: float) -> None:
    worker.asked.close()
    if worker.task is not None or not worker.greeting.said.is_set():
        worker.process.kill()
    worker.process.join(seconds)
    if worker.process.is_alive():
        worker.process.kill()
        worker.process.join(seconds)
    worker.listener.join(seconds)
    worker.answers.close()


def _settled(greeting: _Greeting, deadline: float, clock: Callable[[], float]) -> bool:
    while not greeting.settled.wait(min(HEARD_SECONDS, max(0.0, deadline - clock()))):
        if clock() >= deadline:
            return False
    return True


@dataclass(slots=True)
class _Reading[A: Request]:
    todo: deque[tuple[int, A]]
    again: set[int] = field(default_factory=set[int])

    def back(self, task: _Task[A]) -> None:
        if task.at in self.again:
            raise ReaderFailedError(f"reader processes ended twice before they began reading {task.asked.path}")
        self.again.add(task.at)
        self.todo.appendleft((task.at, task.asked))


@dataclass(slots=True)
class _Pool[A: Request, B]:
    read: Callable[[A], B]
    patience: Patience
    most: int
    most_answer_bytes: int
    clock: Callable[[], float]
    workers: dict[int, _Worker[A]] = field(default_factory=dict[int, _Worker[A]])
    heard: "queue.Queue[Heard]" = field(default_factory=queue.Queue[Heard])
    made: int = 0

    def _started(self, count: int) -> None:
        context = multiprocessing.get_context("spawn")
        started: list[_Worker[A]] = []
        for _ in range(count):
            asked_there, asked = context.Pipe(duplex=False)
            answers, answers_there = context.Pipe(duplex=False)
            arguments = (asked_there, answers_there, self.read, self.most_answer_bytes)
            process = context.Process(target=_serve, args=arguments, daemon=True)
            process.start()
            asked_there.close()
            answers_there.close()
            self.made += 1
            greeting = _Greeting()
            listener = threading.Thread(target=_listened, args=(self.made, answers, greeting, self.heard), daemon=True)
            listener.start()
            worker = _Worker[A](self.made, process, asked, answers, greeting, listener)
            self.workers[self.made] = worker
            started.append(worker)
        deadline = self.clock() + self.patience.start_seconds
        for worker in started:
            if not _settled(worker.greeting, deadline, self.clock):
                raise ReaderFailedError(f"a reader process did not start in {self.patience.start_seconds:,.0f} seconds")
            if not worker.greeting.said.is_set():
                worker.process.join(self.patience.stop_seconds)
                raise ReaderFailedError(f"a reader process ended as it started, exit code {worker.process.exitcode}")

    def _busy(self) -> int:
        return sum(worker.task is not None for worker in self.workers.values())

    def _dropped(self, worker: _Worker[A]) -> None:
        _ = self.workers.pop(worker.number)
        _stopped(worker, self.patience.stop_seconds)

    def _given(self, todo: deque[tuple[int, A]]) -> None:
        for worker in [each for each in self.workers.values() if each.task is None][: len(todo)]:
            at, asked = todo.popleft()
            worker.task, worker.began = _Task(at, asked, self.clock() + self.patience.seconds(asked.size)), False
            with suppress(OSError):
                worker.asked.send(asked)

    def _overdue(self) -> Iterator[tuple[int, B | Refused]]:
        now = self.clock()
        for worker in list(self.workers.values()):
            if (task := worker.task) is not None and now >= task.deadline:
                self._dropped(worker)
                yield task.at, came_to_no_end(self.patience.seconds(task.asked.size))

    def _ended(self, worker: _Worker[A], reading: _Reading[A]) -> tuple[int, B | Refused] | None:
        task, began = worker.task, worker.began
        self._dropped(worker)
        if task is None:
            return None
        if not began:
            reading.back(task)
            return None
        return task.at, stopped(ENDED)

    def _answer(self, number: int, message: object, reading: _Reading[A]) -> tuple[int, B | Refused] | None:
        worker = self.workers.get(number)
        if worker is None:
            return None
        if message is None:
            return self._ended(worker, reading)
        task = cast("_Task[A]", worker.task)
        if message == BEGAN:
            worker.began = True
            return None
        if isinstance(message, _Raised):
            worker.task = None
            raise ReaderFailedError(
                f"reading {task.asked.path} raised an error in its process, {message.error}; its traceback is above"
            )
        worker.task = None
        return task.at, cast("B", message)

    def _waited(self) -> float:
        nearest = min(task.deadline for worker in self.workers.values() if (task := worker.task) is not None)
        return min(HEARD_SECONDS, max(0.0, nearest - self.clock()))

    def files(self, asked: Sequence[A]) -> Iterator[tuple[int, B | Refused]]:
        reading = _Reading(deque(enumerate(asked)))
        while reading.todo or self._busy():
            if (wanted := min(self.most, self._busy() + len(reading.todo)) - len(self.workers)) > 0:
                self._started(wanted)
            self._given(reading.todo)
            try:
                number, message = self.heard.get(timeout=self._waited())
            except queue.Empty:
                yield from self._overdue()
                continue
            if (answer := self._answer(number, message, reading)) is not None:
                yield answer

    def ended(self) -> None:
        for worker in list(self.workers.values()):
            self._dropped(worker)


@contextmanager
def _sandboxed[A: Request, B](
    read: Callable[[A], B], patience: Patience, workers: int, clock: Callable[[], float], most_answer_bytes: int
) -> Generator[Answering[A, B]]:
    if workers < 1:
        raise ValueError(f"a sandbox of {workers} reader processes reads nothing: give it one at least")
    pool = _Pool(read, patience, reader_processes(workers, usable_cpus()), most_answer_bytes, clock)
    try:
        yield pool.files
    finally:
        pool.ended()


def sandboxed(
    read: ReadOne = read_file,
    patience: Patience = PATIENCE,
    workers: int = MOST_WORKERS,
    clock: Callable[[], float] = time.perf_counter,
    most_answer_bytes: int = MOST_ANSWER_BYTES,
) -> AbstractContextManager[FileReading]:
    return _sandboxed(read, patience, workers, clock, most_answer_bytes)


def picturing(
    patience: Patience = PATIENCE, clock: Callable[[], float] = time.perf_counter
) -> AbstractContextManager[Picturing]:
    return _sandboxed(picture, patience, 1, clock, MOST_ANSWER_BYTES)
