import dataclasses
import io
import json
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import cast

import pytest

from sift.api.serving import WorkerAddress
from sift.interfaces.cli.main import run
from sift.settings import Settings
from tests.environments import clean_environment
from tests.models import Hashed
from tests.surface.commands import indexed_notes, ran
from tests.tiny_model import tiny_models
from tests.waits import until
from tests.workers import LOOPBACK, nobody_listens, working

KEY_SHOWN = re.compile(r"SIFT_EMBED_REMOTE=https://([\w-]+)\.([\w-]+)@127\.0\.0\.1:(\d+)")
THROUGH_A_WORKER = """
import sys
from sift.interfaces.cli.main import run
from sift.settings import from_environment

settings = from_environment()
for argv in (["add", sys.argv[1]], ["index"], ["embed"], ["query", "brush strokes"], ["vsearch", "brush"]):
    if run(argv, settings, None, None) != 0:
        raise SystemExit(f"sift {' '.join(argv)} failed")
print("torch" in sys.modules)
"""


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    tiny = tiny_models(tmp_path_factory.mktemp("models"))
    return str(tiny), str(tiny.parent / "reranker")


def shown_address(err: io.StringIO) -> WorkerAddress | None:
    found = KEY_SHOWN.search(err.getvalue())
    return None if found is None else WorkerAddress(LOOPBACK, int(found[3]), found[1], found[2])


def test_sift_worker_lends_this_machine_s_models_to_another_sift_and_keeps_its_key_out_of_its_log(
    settings: Settings, notes: Path, tmp_path: Path, models: tuple[str, str]
) -> None:
    elsewhere = WorkerAddress("elsewhere.test", 8766, "t", "p")
    lending = dataclasses.replace(settings, embed_model=models[0], rerank_model=models[1], embed_remote=elsewhere)
    out, err, asked = io.StringIO(), io.StringIO(), list[tuple[int, str, str]]()

    def sent_work_then_stopped() -> None:
        try:
            assert until(lambda: shown_address(err) is not None), "the worker did not say its key"
            client = Settings(
                tmp_path / "client", None, tmp_path, state=tmp_path / "state", embed_remote=shown_address(err)
            )
            indexed_notes(client, notes)
            asked.extend(ran(client, *argv) for argv in (["index"], ["embed"], ["query", "reward shaping", "--json"]))
        finally:
            signal.raise_signal(signal.SIGINT)

    threading.Thread(target=sent_work_then_stopped, daemon=True).start()
    code = run(["worker", "--port", "0", "--host", LOOPBACK], lending, out, err)
    address = shown_address(err)
    assert address is not None
    assert (code, [each[0] for each in asked]) == (0, [0, 0, 0])
    hits = cast("dict[str, list[dict[str, str]]]", json.loads(asked[2][1]))["hits"]
    assert {hit["id"] for hit in hits} == {"notes/rl.md", "notes/robots/brush.md"}
    assert err.getvalue().endswith(f" done  stopped serving https://{LOOPBACK}:{address.port}\n")
    (log,) = (settings.home / "logs").glob("worker-*.log")
    logged = log.read_text(encoding="utf-8")
    assert (address.token in logged, f"SIFT_EMBED_REMOTE=https://<key>@{LOOPBACK}:{address.port}" in logged) == (
        False,
        True,
    )


def test_a_sift_sending_its_work_to_a_worker_never_loads_the_model_library(
    settings: Settings, notes: Path, tmp_path: Path
) -> None:
    indexed_notes(settings, notes)
    with working(settings, Hashed()) as worker:
        address = worker.address
        remote = f"https://{address.token}.{address.pin}@{LOOPBACK}:{address.port}"
        environment = clean_environment(
            SIFT_HOME=str(tmp_path / "client"), SIFT_EMBED_REMOTE=remote, PYTHONIOENCODING="utf-8"
        )
        done = subprocess.run(
            [sys.executable, "-c", THROUGH_A_WORKER, str(notes)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=environment,
        )
    assert (done.returncode, done.stdout, " load " in done.stderr) == (0, "False\n", False), done.stderr


def test_the_cli_says_a_worker_cannot_be_reached_without_its_key(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    assert ran(settings, "index")[0] == 0
    port = nobody_listens()
    unreachable = dataclasses.replace(settings, embed_remote=WorkerAddress(LOOPBACK, port, "s3cret", "p1n"))
    code, out, err = ran(unreachable, "embed")
    assert (code, out, "s3cret" in err) == (1, "", False)
    _, said, fix, _ = err.split("\n")
    assert said.startswith(f"error  embedding remote https://{LOOPBACK}:{port} is unreachable or failed: ")
    assert fix == "  fix  start the worker on that machine and let it through its firewall, or unset SIFT_EMBED_REMOTE"
