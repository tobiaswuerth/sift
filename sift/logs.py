"""The commands that change the index or serve it (LOGGED) keep what sift said in a log of their own, every line with
its time: SIFT_HOME/logs/serve-2026-10-23_210509.log, with the records of the libraries it runs, and how the command
ended if it failed. The latest KEPT logs of each command are kept, and of a long one, as a server's that logs every
request, its last parts (MOST_BYTES, PARTS)."""

import contextlib
import glob
import logging
import logging.handlers
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

from . import __version__, settings

LOGGED = ("add", "rm", "index", "embed", "serve", "worker")
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
KEPT = 20  # logs of each command kept, the latest: the others are deleted as a command starts another
MOST_BYTES = 16 << 20  # a log past this goes on in a new part, its earlier ones NAME.log.1 (the latest), .2, ...
PARTS = 3  # earlier parts of a log kept: a server running for weeks keeps its last 64 MB, not a file without end
said = logging.getLogger("sift.console")  # what console says, which it prints itself: never passed on, so that no
said.propagate = False  # handler of the root logger prints it twice
_handler: logging.FileHandler | None = None  # the log file of this command, while there is one


def _claim(folder: Path, stem: str) -> Path:
    """A new log file in `folder`, created at once: `stem`.log, or `stem`-2.log and so on if that is taken, so that two
    commands started in the same second, in two processes too, never write into one."""
    folder.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        path = folder / (stem + (f"-{n}" if n > 1 else "") + ".log")
        try:
            path.open("x").close()
            return path
        except FileExistsError:
            n += 1


def _prune(folder: Path, name: str) -> None:
    """Delete the logs of the command `name` in `folder` but the latest KEPT, each with its parts. One a running sift
    still writes, which Windows keeps from being deleted, is left for the next to delete."""
    by_start = sorted(folder.glob(f"{glob.escape(name)}-*.log"))  # their names begin with the time they began
    for log in by_start[:-KEPT]:
        for file in [log, *folder.glob(f"{glob.escape(log.name)}.*")]:
            with contextlib.suppress(OSError):
                file.unlink(missing_ok=True)


@contextlib.contextmanager
def kept(name: str, now: datetime) -> Generator[Path]:
    """Log this command, `name`, to a new file while the block runs; yields its path."""
    global _handler
    folder = settings.current().home / "logs"
    path = _claim(folder, f"{name}-{now:%Y-%m-%d_%H%M%S}")
    _prune(folder, name)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=MOST_BYTES, backupCount=PARTS, encoding="utf-8")
    outer = _handler
    handler.setFormatter(logging.Formatter(FORMAT))
    root, level = logging.getLogger(), logging.getLogger().level
    for logger in (root, said):
        logger.addHandler(handler)
    said.setLevel(logging.INFO)
    root.setLevel(min(level, logging.INFO) if level else level)
    _handler = handler
    said.info("sift %s, %s", __version__, name)  # the log's alone: it says which sift wrote it
    try:
        yield path
    except KeyboardInterrupt:
        said.error("stopped")
        raise
    except SystemExit as e:
        if e.code not in (None, 0):
            said.error("%s", e.code)
        raise
    except Exception as e:
        said.error("%s: %s", type(e).__name__, e)
        raise
    finally:
        for logger in [root, *map(logging.getLogger, list(logging.root.manager.loggerDict))]:
            if handler in logger.handlers:
                logger.removeHandler(handler)
        root.setLevel(level)
        handler.close()
        _handler = outer


def attach(*names: str) -> None:
    """Log these loggers too, which don't pass their records on (uvicorn's, once it has set them up)."""
    for name in names:
        if _handler and _handler not in logging.getLogger(name).handlers:
            logging.getLogger(name).addHandler(_handler)
