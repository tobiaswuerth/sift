import contextlib
import glob
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from sift.api.events import Event
from sift.interfaces.console.live import Live
from sift.interfaces.console.progress import Looks
from sift.interfaces.console.style import PLAIN, unpainted

LOGGED = frozenset({"add", "rm", "move", "activate", "deactivate", "index", "embed"})
FOLDER = "logs"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
SAID = "sift"
PARTS = 3
COLUMNS = 120


@dataclass(frozen=True, slots=True)
class LogLimits:
    kept: int = 20
    most_bytes: int = 16 << 20


LOG_LIMITS = LogLimits()


@dataclass(frozen=True, slots=True)
class Log:
    path: Path
    said: logging.Logger

    def write(self, text: str, /) -> int:
        for line in text.split("\n"):
            if line:
                self.said.info("%s", unpainted(line))
        return len(text)

    def flush(self) -> None:
        for handler in self.said.handlers:
            handler.flush()

    def ended(self, code: int) -> None:
        self.said.log(logging.ERROR if code else logging.INFO, "ended with exit code %d", code)

    def stopped(self) -> None:
        self.said.error("stopped")

    def failed(self) -> None:
        self.said.exception("failed")


@dataclass(frozen=True, slots=True)
class HeardToo:
    screen: Live
    log: Live

    def __call__(self, event: Event) -> None:
        self.screen(event)
        self.log(event)

    def tick(self) -> None:
        self.screen.tick()
        self.log.tick()


def heard_too(screen: Live, log: Log, clock: Callable[[], float]) -> HeardToo:
    looks = Looks(PLAIN, screen.looks.platform)
    return HeardToo(screen, Live(log, looks, redraw=False, columns=lambda: COLUMNS, clock=clock))


def _claimed(folder: Path, stem: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    taken = 1
    while True:
        path = folder / (f"{stem}-{taken}.log" if taken > 1 else f"{stem}.log")
        try:
            path.touch(exist_ok=False)
        except FileExistsError:
            taken += 1
        else:
            return path


def _deleted_unless_another_sift_writes_it(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError, PermissionError):
        path.unlink()


def _pruned(folder: Path, command: str, kept: int) -> None:
    logs = sorted(folder.glob(f"{glob.escape(command)}-*.log"))
    for old in logs[: max(len(logs) - kept, 0)]:
        for path in (old, *folder.glob(f"{glob.escape(old.name)}.*")):
            _deleted_unless_another_sift_writes_it(path)


@contextmanager
def kept(folder: Path, command: str, now: datetime, heading: str, limits: LogLimits = LOG_LIMITS) -> Generator[Log]:
    path = _claimed(folder, f"{command}-{now:%Y-%m-%d_%H%M%S}")
    _pruned(folder, command, limits.kept)
    handler = RotatingFileHandler(path, maxBytes=limits.most_bytes, backupCount=PARTS, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    root, said = logging.getLogger(), logging.getLogger(SAID)
    levels, propagates = (root.level, said.level), said.propagate
    said.propagate = False
    said.setLevel(logging.INFO)
    root.setLevel(min(root.level, logging.INFO) if root.level else root.level)
    for logger in (root, said):
        logger.addHandler(handler)
    try:
        said.info("%s", heading)
        yield Log(path, said)
    finally:
        for logger in (root, said):
            logger.removeHandler(handler)
        root.setLevel(levels[0])
        said.setLevel(levels[1])
        said.propagate = propagates
        handler.close()
