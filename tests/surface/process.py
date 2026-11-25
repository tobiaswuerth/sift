import os
import signal
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from sift.interfaces.server.endpoint import found, own_endpoint
from sift.settings import OWN_FOLDER, Settings
from tests.environments import clean_environment
from tests.waits import until

INTERRUPT = cast("int", getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
OWN_GROUP = cast("int", getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
SERVE = ("serve", "--port", "0", "--json")


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
        [sys.executable, "-m", "sift", *SERVE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment(home),
        creationflags=OWN_GROUP,
    )
    try:
        listed = own_endpoint(settings_of(home))
        assert until(lambda: listed.exists() or running.poll() is not None, 60), (
            "the home process neither listed its endpoint nor ended within 60 s"
        )
        yield running
    finally:
        if running.poll() is None:
            running.send_signal(INTERRUPT)
        running.communicate(timeout=60)


def through_stdio(home: Path, args: tuple[str, ...], sent: str) -> subprocess.CompletedProcess[str]:
    settings = settings_of(home)
    try:
        return subprocess.run(
            [sys.executable, "-m", "sift", *args],
            input=sent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=180,
            env=environment(home),
        )
    finally:
        left = found(settings)
        if left is not None:
            os.kill(left.pid, signal.SIGTERM)
            assert until(lambda: found(settings) is None, 60), "the started home process did not stop within 60 s"
