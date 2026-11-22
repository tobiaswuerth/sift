import errno
import io
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TextIO, final, override

import pytest

from sift.interfaces.cli.main import main, run
from sift.interfaces.console.text import table
from sift.interfaces.console.units import size
from sift.settings import Settings


@final
class Refusing(io.TextIOWrapper):
    def __init__(self, error: BaseException) -> None:
        super().__init__(io.BytesIO(), encoding="utf-8")
        self.error = error

    @override
    def write(self, s: str, /) -> int:
        raise self.error


def sift(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sift", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
        env={**os.environ, "SIFT_HOME": str(home), "PYTHONIOENCODING": "utf-8"},
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    for name in ("user", "work"):
        (tmp_path / name).mkdir()
    return Settings(home=tmp_path / "home", user=tmp_path / "user", cwd=tmp_path / "work")


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    path = tmp_path / "notes"
    path.mkdir()
    return path


def ran(settings: Settings, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run(argv, settings, out, err)
    return code, out.getvalue(), err.getvalue()


def test_version_names_the_installed_release(tmp_path: Path) -> None:
    result = sift(tmp_path, "--version")
    assert (result.returncode, result.stdout, result.stderr) == (0, f"sift {version('sift-rag')}\n", "")


def test_without_a_command_the_usage_is_shown(settings: Settings) -> None:
    code, out, err = ran(settings)
    assert (code, err) == (0, "")
    assert out.startswith("usage: sift")


def test_an_unknown_option_is_a_usage_error(settings: Settings) -> None:
    code, out, err = ran(settings, "--frobnicate")
    assert (code, out) == (2, "")
    assert "unrecognized arguments: --frobnicate" in err


def test_main_runs_a_command_in_this_process_and_ends_with_its_code(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ended:
        main(["--version"])
    assert (ended.value.code, capsys.readouterr().out) == (0, f"sift {version('sift-rag')}\n")


def test_a_collection_added_is_shown_by_status(settings: Settings, notes: Path) -> None:
    store = (settings.home / "collections" / "notes.sqlite").resolve()
    assert ran(settings, "add", str(notes)) == (
        0,
        "",
        f"added  notes  {notes.resolve()}\nstore  {store}\n next  sift index notes\n",
    )
    assert ran(settings, "status") == (
        0,
        "\n".join(
            table(
                (
                    ("collection", "folder", "documents", "chunks", "size"),
                    ("notes", str(notes.resolve()), "0", "0", size(store.stat().st_size)),
                ),
                frozenset({2, 3, 4}),
            )
        )
        + "\n",
        "",
    )


def test_status_and_add_answer_in_json_when_asked(settings: Settings, notes: Path) -> None:
    store = (settings.home / "collections" / "notes.sqlite").resolve()
    code, out, err = ran(settings, "add", str(notes), "--title-from", "name", "--json")
    assert (code, err) == (0, "")
    assert json.loads(out) == {
        "name": "notes",
        "folder": str(notes.resolve()),
        "store": str(store),
        "title_from": "name",
        "vault": False,
    }
    code, out, err = ran(settings, "status", "--json")
    assert (code, err) == (0, "")
    assert json.loads(out) == {
        "home": str(settings.home),
        "collections": [
            {
                "name": "notes",
                "folder": str(notes.resolve()),
                "store": str(store),
                "contents": {"documents": 0, "chunks": 0, "stored_bytes": store.stat().st_size, "state": "counted"},
            }
        ],
    }


def test_a_path_is_taken_from_the_user_s_folder_or_the_current_one(settings: Settings) -> None:
    for place in (settings.user / "notes", settings.cwd / "papers"):
        place.mkdir()
    assert ran(settings, "add", "~/notes")[0] == 0
    assert ran(settings, "add", "papers")[0] == 0
    shown = ran(settings, "status")[1]
    assert str((settings.user / "notes").resolve()) in shown
    assert str((settings.cwd / "papers").resolve()) in shown


def test_an_error_is_a_message_with_its_fix_and_no_traceback(settings: Settings, tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    assert ran(settings, "add", str(gone)) == (
        1,
        "",
        f"error  {gone} is no folder\n  fix  give the folder whose files the collection holds\n",
    )


def test_an_error_of_the_system_is_a_message(settings: Settings, notes: Path) -> None:
    settings.home.write_bytes(b"a file, not a folder")
    code, out, err = ran(settings, "add", str(notes))
    assert (code, out) == (1, "")
    assert err.startswith("error  [")
    assert err.endswith("home'\n")
    assert err.count("\n") == 1


def test_a_bug_keeps_its_traceback(settings: Settings, notes: Path) -> None:
    ran(settings, "add", str(notes))
    with pytest.raises(KeyError, match="broken"):
        run(["status"], settings, Refusing(KeyError("broken")), io.StringIO())


def test_ctrl_c_ends_a_command_quietly(settings: Settings, notes: Path) -> None:
    ran(settings, "add", str(notes))
    err = io.StringIO()
    assert run(["status"], settings, Refusing(KeyboardInterrupt()), err) == 130
    assert err.getvalue() == ""


@pytest.mark.parametrize("gone", [BrokenPipeError(errno.EPIPE, "Broken pipe"), OSError(errno.EINVAL, "Invalid")])
def test_a_reader_that_left_ends_the_command_quietly(settings: Settings, notes: Path, gone: OSError) -> None:
    ran(settings, "add", str(notes))
    err = io.StringIO()
    assert run(["status"], settings, Refusing(gone), err) == 1
    assert err.getvalue() == ""


@final
class GoneFile(io.TextIOWrapper):
    @override
    def write(self, s: str, /) -> int:
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")


def test_results_nobody_reads_are_let_go_so_that_python_ends_without_a_word(
    settings: Settings, notes: Path, tmp_path: Path
) -> None:
    ran(settings, "add", str(notes))
    with GoneFile((tmp_path / "out").open("wb"), encoding="utf-8") as gone:
        assert run(["status"], settings, gone, io.StringIO()) == 1
        gone.buffer.write(b"left in the buffer")
        gone.buffer.flush()
    assert (tmp_path / "out").read_bytes() == b""


def test_results_that_cannot_be_written_for_another_reason_are_an_error(settings: Settings, notes: Path) -> None:
    ran(settings, "add", str(notes))
    err = io.StringIO()
    assert run(["status"], settings, Refusing(OSError(errno.ENOSPC, "No space left on device")), err) == 1
    assert err.getvalue() == f"error  [Errno {errno.ENOSPC}] No space left on device\n"


def closed() -> TextIO:
    stream = io.StringIO()
    stream.close()
    return stream


@pytest.mark.parametrize("err", [Refusing(BrokenPipeError(errno.EPIPE, "Broken pipe")), closed(), None])
def test_a_stderr_no_one_reads_stops_no_work(settings: Settings, notes: Path, err: TextIO | None) -> None:
    assert run(["add", str(notes)], settings, io.StringIO(), err) == 0
    assert ran(settings, "status")[1].split("\n")[1].startswith("notes")


def test_without_streams_a_command_runs(settings: Settings, notes: Path) -> None:
    assert run(["add", str(notes)], settings, None, None) == 0
    assert run(["status"], settings, None, None) == 0
    assert run([], settings, None, None) == 0


@pytest.mark.parametrize("stream", ["out", "err"])
def test_a_stream_of_another_encoding_is_switched_to_utf8(settings: Settings, tmp_path: Path, stream: str) -> None:
    folder = tmp_path / "Notizen-ä"
    folder.mkdir()
    legacy = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    other = io.StringIO()
    out, err = (legacy, other) if stream == "out" else (other, legacy)
    assert run(["add", str(folder)], settings, other, err) == 0
    assert run(["status"], settings, out, other) == 0
    legacy.flush()
    assert legacy.encoding == "utf-8"
    assert "Notizen-ä" in legacy.buffer.getvalue().decode()


def test_a_reader_leaving_is_found_before_python_ends(tmp_path: Path, notes: Path) -> None:
    assert sift(tmp_path / "home", "add", str(notes)).returncode == 0
    environment = {**os.environ, "SIFT_HOME": str(tmp_path / "home")}
    unread, results = os.pipe()
    os.close(unread)
    try:
        ended = subprocess.run(
            [sys.executable, "-m", "sift", "status"],
            stdout=results,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
            timeout=60,
        )
    finally:
        os.close(results)
    assert (ended.returncode, ended.stderr) == (1, b"")


def test_the_command_line_needs_the_standard_library_alone() -> None:
    code = (
        "import sys; before = set(sys.modules); import sift.interfaces.cli.main; "
        "print(sorted({name.split('.')[0] for name in set(sys.modules) - before} - set(sys.stdlib_module_names)))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60)
    assert result.stdout == "['sift']\n"
