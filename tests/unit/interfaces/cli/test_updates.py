import importlib.util
import json
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sift.interfaces.cli.updates import DAY, Checking, Known, fetched, from_source, known, newer, told, upgrade_command

NOW = 1_800_000_000.0
LATEST = json.dumps({"info": {"version": "2.1.0"}}).encode()
NEWER = (
    ("2.1.0", "2.0.0"),
    ("2.0.0", "2.0.0a1"),
    ("2.0.0a2", "2.0.0a1"),
    ("2.0.0rc1", "2.0.0b3"),
    ("2.0.1", "2.0"),
)
NOT_NEWER = (
    ("2.0", "2.0.0"),
    ("1.15.2", "2.0.0a0"),
    ("2.0.0", "2.0.0"),
    ("2.0.0.dev1", "1.0.0"),
    ("2.\N{SUPERSCRIPT ONE}.0", "1.0.0"),
    ("3.0.0", "a local build"),
)


@contextmanager
def pypi(answer: bytes, status: int = 200) -> Generator[str]:
    class Answering(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(answer)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Answering)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/pypi/sift-rag/json"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.mark.parametrize("latest, installed", NEWER)
def test_a_newer_sift_is_one_of_a_later_release_or_pre_release(latest: str, installed: str) -> None:
    assert newer(latest, installed)


@pytest.mark.parametrize("latest, installed", NOT_NEWER)
def test_the_same_an_earlier_or_an_unreadable_version_is_no_newer_sift(latest: str, installed: str) -> None:
    assert not newer(latest, installed)


def test_a_sift_run_from_its_source_is_not_installed_in_its_environment(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    assert from_source(environment, environment / "Lib" / "site-packages" / "sift") is False
    assert from_source(environment, tmp_path / "src" / "sift") is True


def test_each_installation_is_upgraded_as_it_was_made(tmp_path: Path) -> None:
    tool = tmp_path / "uv" / "tools" / "sift-rag"
    tool.mkdir(parents=True)
    (tool / "uv-receipt.toml").write_bytes(b"")
    made_by_uv = tmp_path / "project" / ".venv"
    made_by_uv.mkdir(parents=True)
    (made_by_uv / "pyvenv.cfg").write_bytes(b"home = /usr/bin\nuv = 0.9.2\n")
    cached = tmp_path / "cache" / "archive-v0" / "abc"
    cached.mkdir(parents=True)
    (cached / "pyvenv.cfg").write_bytes(b"uv = 0.9.2\n")
    commands = [
        upgrade_command(tool, pip=False, python="python"),
        upgrade_command(tmp_path / "pipx" / "venvs" / "sift-rag", pip=True, python="python"),
        upgrade_command(tmp_path / "venv", pip=True, python="/venv/bin/python"),
        upgrade_command(made_by_uv, pip=False, python="python"),
        upgrade_command(cached, pip=False, python="python"),
        upgrade_command(tmp_path / "conda", pip=False, python="python"),
    ]
    assert commands == [
        "uv tool upgrade sift-rag",
        "pipx upgrade sift-rag",
        "/venv/bin/python -m pip install --upgrade sift-rag",
        "uv pip install --upgrade sift-rag",
        "uvx sift-rag@latest",
        None,
    ]


@pytest.mark.parametrize(
    "written",
    [b"", b"{not json", b"[1, 2]", b'{"checked": "today", "latest": 2, "failed": []}', b'{"checked": true}'],
)
def test_what_the_state_holds_in_another_shape_counts_as_nothing(tmp_path: Path, written: bytes) -> None:
    (tmp_path / "update.json").write_bytes(written)
    assert known(tmp_path / "update.json") == Known()
    assert known(tmp_path / "missing.json") == Known()


def recording(asked: list[str], latest: str) -> Callable[[str, str], str]:
    def fetch(url: str, installed: str) -> str:
        asked.append(f"{url} for {installed}")
        return latest

    return fetch


def checking(tmp_path: Path, url: str = "pypi", fetch: Callable[[str, str], str] = fetched) -> Checking:
    return Checking(
        now=lambda: NOW,
        fetch=fetch,
        url=url,
        prefix=tmp_path / "venv",
        package=tmp_path / "venv" / "sift",
        pip=lambda: True,
        python="python",
    )


def test_once_a_day_pypi_is_asked_and_a_newer_sift_told_of_with_its_upgrade_command(tmp_path: Path) -> None:
    with pypi(LATEST) as url:
        assert told(tmp_path, "2.0.0", checking(tmp_path, url)) == (
            "sift 2.1.0 is out, this is 2.0.0: python -m pip install --upgrade sift-rag"
        )
    assert known(tmp_path / "update.json") == Known(NOW, "2.1.0", None)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["update.json"]
    asked = list[str]()
    again = checking(tmp_path, fetch=recording(asked, "2.2.0"))
    (tmp_path / "update.json").write_bytes(json.dumps({"checked": NOW - DAY + 1, "latest": "2.1.0"}).encode())
    assert (told(tmp_path, "2.0.0", again), asked) == (None, [])


@pytest.mark.parametrize(
    "answer, status, url, why",
    [
        (LATEST, 503, None, "ValueError: PyPI answered 503 Service Unavailable"),
        (b"<html>", 200, None, "JSONDecodeError: "),
        (b'{"info": {"version": 2}}', 200, None, "TypeError: PyPI named no version"),
        (LATEST, 200, "file:///etc/passwd", "ValueError: file:///etc/passwd is no web address"),
    ],
)
def test_a_check_that_fails_keeps_when_and_why_and_what_it_knew(
    tmp_path: Path, answer: bytes, status: int, url: str | None, why: str
) -> None:
    state = tmp_path / "update.json"
    state.write_bytes(json.dumps({"checked": 1.0, "latest": "2.0.5"}).encode())
    with pypi(answer, status) as served:
        assert told(tmp_path, "2.0.0", checking(tmp_path, url or served)) == (
            "sift 2.0.5 is out, this is 2.0.0: python -m pip install --upgrade sift-rag"
        )
    found = known(state)
    assert (found.checked, found.latest) == (NOW, "2.0.5")
    assert found.failed is not None
    assert found.failed.startswith(why)


def test_what_pypi_said_that_cannot_be_kept_is_said_and_leaves_nothing_behind(tmp_path: Path) -> None:
    (tmp_path / "update.json").mkdir()
    line = told(tmp_path, "2.0.0", checking(tmp_path, fetch=recording([], "2.1.0")))
    assert line is not None
    assert line.startswith(f"what PyPI said could not be kept in {tmp_path / 'update.json'} (")
    assert line.endswith("); the next command asks again")
    assert [path.name for path in tmp_path.iterdir()] == ["update.json"]


def test_nothing_newer_is_not_told_of_and_without_a_home_pypi_is_not_asked(tmp_path: Path) -> None:
    asked = list[str]()
    latest = checking(tmp_path, fetch=recording(asked, "2.1.0"))
    assert told(tmp_path, "2.1.0", latest) is None
    assert told(tmp_path / "no home yet", "2.0.0", latest) is None
    assert asked == ["pypi for 2.1.0"]


def test_a_sift_run_from_its_source_is_told_nothing_and_asks_nothing(tmp_path: Path) -> None:
    asked = list[str]()
    source = Checking(now=lambda: NOW, fetch=recording(asked, "9.0.0"), package=tmp_path)
    assert (told(tmp_path, "2.0.0", source), asked) == (None, [])
    assert told(tmp_path / "no home yet", "2.0.0") is None


def test_whether_pip_upgrades_it_is_looked_up_in_this_environment(tmp_path: Path) -> None:
    looked_up = Checking(
        now=lambda: NOW, fetch=recording([], "2.1.0"), prefix=tmp_path, package=tmp_path / "sift", python="python"
    )
    by_pip = ": python -m pip install --upgrade sift-rag" if importlib.util.find_spec("pip") is not None else ""
    assert told(tmp_path, "2.0.0", looked_up) == f"sift 2.1.0 is out, this is 2.0.0{by_pip}"


def test_without_a_command_known_a_newer_sift_is_told_of_alone(tmp_path: Path) -> None:
    unknown = Checking(
        now=lambda: NOW, fetch=recording([], "2.1.0"), prefix=tmp_path, package=tmp_path / "sift", pip=lambda: False
    )
    assert told(tmp_path, "2.0.0", unknown) == "sift 2.1.0 is out, this is 2.0.0"
