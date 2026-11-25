import shutil
import subprocess
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tests.environments import clean_environment
from tests.tiny_model import tiny_model, tiny_reranker
from tests.waits import joined

COMMIT = "0123456789abcdef0123456789abcdef01234567"
EMBEDDER, RERANKER, MISSING = "example-org/tiny-embedder", "example-org/tiny-reranker", "example-org/never-fetched"
LOADS = """
import json, sys
from sift.api.errors import ModelUnavailableError
from sift.inference.local import Local
embedder, reranker, missing = sys.argv[1:]
models = Local(embedder, reranker)
found = [len(models.encode(["dark matter"], query=True)), len(models.rerank("dark matter", ["halo"]))]
try:
    Local(missing).encode(["dark matter"], query=True)
except ModelUnavailableError:
    found.append("refused")
print(json.dumps(found))
"""


@contextmanager
def hub(asked: list[str]) -> Generator[str]:
    class Refusing(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            asked.append(self.path)
            self.send_response(404)
            self.end_headers()

        def do_HEAD(self) -> None:
            self.do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Refusing)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        joined(thread, 5)


def cached(cache: Path, repository: str, folder: Path) -> None:
    shutil.copytree(folder, cache / f"models--{repository.replace('/', '--')}" / "snapshots" / COMMIT)


def test_a_model_in_the_cache_loads_without_asking_the_hub_and_one_missing_is_asked_for(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cached(cache, EMBEDDER, tiny_model(tmp_path / "made"))
    cached(cache, RERANKER, tiny_reranker(tmp_path / "made"))
    asked = list[str]()
    with hub(asked) as endpoint:
        done = subprocess.run(
            [sys.executable, "-c", LOADS, *(f"{name}@{COMMIT}" for name in (EMBEDDER, RERANKER, MISSING))],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=300,
            env=clean_environment(HF_HUB_CACHE=str(cache), HF_ENDPOINT=endpoint, HF_HUB_DISABLE_TELEMETRY="1"),
        )
    assert (done.returncode, done.stdout) == (0, '[1, 1, "refused"]\n'), done.stderr[-2000:]
    assert ({MISSING in path for path in asked}, len(asked) > 0) == ({True}, True)
