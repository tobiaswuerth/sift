import signal
import subprocess
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from sift.interfaces.server.endpoint import own_endpoint
from sift.settings import OWN_FOLDER, Settings
from tests.environments import clean_environment

INTERRUPT = cast("int", getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
OWN_GROUP = cast("int", getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def settings_of(home: Path) -> Settings:
    return Settings(home=home, user=None, cwd=home.parent, state=home.parent / "state" / OWN_FOLDER)


def environment(home: Path, model: str = "", reranker: str = "") -> dict[str, str]:
    return clean_environment(
        SIFT_HOME=str(home),
        LOCALAPPDATA=str(home.parent / "state"),
        XDG_STATE_HOME=str(home.parent / "state"),
        PYTHONIOENCODING="utf-8",
        NO_COLOR="1",
        SIFT_EMBED_MODEL=model,
        SIFT_RERANK_MODEL=reranker,
    )


def sift(
    home: Path, *args: str, model: str = "", reranker: str = "", cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sift", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=180,
        env=environment(home, model, reranker),
    )


@contextmanager
def served(home: Path) -> Generator[subprocess.Popen[str]]:
    running = subprocess.Popen(
        [sys.executable, "-m", "sift", "serve", "--port", "0", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment(home),
        creationflags=OWN_GROUP,
    )
    try:
        waited, listed = threading.Event(), own_endpoint(settings_of(home))
        for _ in range(6000):
            if listed.exists() or running.poll() is not None:
                break
            waited.wait(0.01)
        yield running
    finally:
        if running.poll() is None:
            running.send_signal(INTERRUPT)
        running.communicate(timeout=60)
