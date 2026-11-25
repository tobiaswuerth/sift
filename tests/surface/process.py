import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sift.interfaces.server.endpoint import found, own_endpoint
from sift.settings import OWN_FOLDER, Settings
from tests.environments import clean_environment
from tests.waits import joined, until

INTERRUPT = cast("int", getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
OWN_GROUP = cast("int", getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
SERVE = ("serve", "--port", "0", "--json")
LOOPBACK = "127.0.0.1"
WORKER = ("worker", "--port", "0", "--host", LOOPBACK, "--json")
REMOTE_SHOWN = re.compile(r"SIFT_EMBED_REMOTE=(https://\S+@127\.0\.0\.1:\d+)")


def settings_of(home: Path) -> Settings:
    return Settings(home=home, user=None, cwd=home.parent, state=home.parent / "state" / OWN_FOLDER)


def environment(home: Path, model: str = "", reranker: str = "", remote: str = "") -> dict[str, str]:
    return clean_environment(
        SIFT_HOME=str(home),
        LOCALAPPDATA=str(home.parent / "state"),
        XDG_STATE_HOME=str(home.parent / "state"),
        PYTHONIOENCODING="utf-8",
        NO_COLOR="1",
        SIFT_EMBED_MODEL=model,
        SIFT_RERANK_MODEL=reranker,
        SIFT_EMBED_REMOTE=remote,
    )


def sift(
    home: Path, *args: str, model: str = "", reranker: str = "", remote: str = "", cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sift", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=180,
        env=environment(home, model, reranker, remote),
    )


@dataclass(frozen=True, slots=True)
class Lent:
    remote: str
    said: list[str]
    running: subprocess.Popen[str]
    answered: list[str]


def _heard(running: subprocess.Popen[str], said: list[str]) -> None:
    if running.stderr is not None:
        said.extend(running.stderr)


@contextmanager
def lending(home: Path, model: str, reranker: str) -> Generator[Lent]:
    running = subprocess.Popen(
        [sys.executable, "-m", "sift", *WORKER],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment(home, model, reranker),
        creationflags=OWN_GROUP,
    )
    said, answered = list[str](), list[str]()
    listening = threading.Thread(target=_heard, args=(running, said), daemon=True)
    listening.start()
    try:
        assert until(lambda: any(REMOTE_SHOWN.search(line) for line in said) or running.poll() is not None, 120), (
            "the worker neither said its key nor ended within 120 s"
        )
        shown = next((found for line in said if (found := REMOTE_SHOWN.search(line))), None)
        assert shown is not None, said
        yield Lent(shown[1], said, running, answered)
    finally:
        if running.poll() is None:
            running.send_signal(INTERRUPT)
        assert running.wait(60) is not None
        joined(listening, 60)
        for pipe in (running.stdout, running.stderr):
            if pipe is not None:
                answered.append(cast("str", pipe.read()))
                pipe.close()


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
