"""The commands that change the index or serve it (LOGGED) keep what sift said in a log of their own, every line with
its time: SIFT_HOME/logs/serve-2026-10-23_210509.log, with the records of the libraries it runs, and how the command
ended if it failed. Every log is kept."""

import contextlib
import logging

from . import __version__, settings

LOGGED = ("add", "rm", "index", "embed", "serve", "worker")
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
said = logging.getLogger("sift.console")  # what console says, which it prints itself: never passed on, so that no
said.propagate = False  # handler of the root logger prints it twice
_handler = None  # the log file of this command, while there is one


def _claim(folder, stem):
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


@contextlib.contextmanager
def kept(name, now):
    """Log this command, `name`, to a new file while the block runs; yields its path."""
    global _handler
    path = _claim(settings.current().home / "logs", f"{name}-{now:%Y-%m-%d_%H%M%S}")
    handler, outer = logging.FileHandler(path, encoding="utf-8"), _handler
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


def attach(*names):
    """Log these loggers too, which don't pass their records on (uvicorn's, once it has set them up)."""
    for name in names:
        if _handler and _handler not in logging.getLogger(name).handlers:
            logging.getLogger(name).addHandler(_handler)
