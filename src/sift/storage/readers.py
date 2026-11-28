import json
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass, field
from pathlib import Path

from sift.api.errors import StoreUnavailableError
from sift.storage.database import Connection, read_only

PARTS = 4
FEWEST = 16
WAITED_SECONDS = 600.0
STUCK_AHEAD = "rows were not read within {seconds:.0f} s: reading the store is stuck"
STUCK_AHEAD_REMEDY = "search again; if it sticks again, check the drive of the store"


@dataclass(eq=False, slots=True)
class ReadPool:
    every: list[Connection]
    idle: list[Connection]
    lock: threading.Lock = field(default_factory=threading.Lock)
    left: bool = False

    def taken(self) -> Connection:
        with self.lock:
            return self.idle.pop()

    def given_back(self, connection: Connection) -> None:
        with self.lock:
            if self.left:
                connection.close()
            else:
                self.idle.append(connection)

    def leave(self) -> None:
        with self.lock:
            self.left = True
            for connection in self.every:
                connection.interrupt()
            for connection in self.idle:
                connection.close()


def readers_of(path: Path, count: int) -> ReadPool:
    with ExitStack() as opening:
        every = [opening.enter_context(closing(read_only(path))) for _ in range(count)]
        opening.pop_all()
    return ReadPool(every, list(every))


def ready[T](future: "Future[T]", waited: float, said: str, remedy: str) -> T:
    try:
        return future.result(timeout=waited)
    except TimeoutError:
        raise StoreUnavailableError(said.format(seconds=waited), remedy) from None


def _touched(readers: ReadPool, sql: str, keys: Sequence[object]) -> None:
    connection = readers.taken()
    try:
        connection.execute(sql, (json.dumps(list(keys)),)).fetchall()
    finally:
        readers.given_back(connection)


@dataclass(eq=False, slots=True)
class Ahead:
    path: Path
    parts: int = PARTS
    fewest: int = FEWEST
    opened: Callable[[Path, int], ReadPool] = readers_of
    waited: float = WAITED_SECONDS
    held: tuple[ReadPool, ThreadPoolExecutor] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def touched(self, sql: str, keys: Sequence[object]) -> None:
        if len(keys) < self.fewest:
            return
        readers, pool = self._held()
        size = -(-len(keys) // min(self.parts, len(keys)))
        pending = [pool.submit(_touched, readers, sql, keys[at : at + size]) for at in range(0, len(keys), size)]
        for each in pending:
            ready(each, self.waited, STUCK_AHEAD, STUCK_AHEAD_REMEDY)

    def _held(self) -> tuple[ReadPool, ThreadPoolExecutor]:
        with self.lock:
            if self.held is None:
                pool = ThreadPoolExecutor(self.parts, thread_name_prefix="sift-ahead")
                self.held = (self.opened(self.path, self.parts), pool)
            return self.held

    def close(self) -> None:
        with self.lock:
            held, self.held = self.held, None
        if held is not None:
            readers, pool = held
            readers.leave()
            pool.shutdown(wait=True, cancel_futures=True)
