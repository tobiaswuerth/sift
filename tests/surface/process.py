import os
import signal
import subprocess
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

ENDPOINT = "server.json"
INTERRUPT = signal.SIGINT


def environment(home: Path, model: str = "", reranker: str = "") -> dict[str, str]:
    return {
        **os.environ,
        "SIFT_HOME": str(home),
        "PYTHONIOENCODING": "utf-8",
        "NO_COLOR": "1",
        "SIFT_EMBED_MODEL": model,
        "SIFT_RERANK_MODEL": reranker,
    }


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
    )
    try:
        waited = threading.Event()
        for _ in range(6000):
            if (home / ENDPOINT).exists() or running.poll() is not None:
                break
            waited.wait(0.01)
        yield running
    finally:
        if running.poll() is None:
            stop = running.kill if sys.platform == "win32" else partial(running.send_signal, INTERRUPT)
            stop()
        running.communicate(timeout=60)
