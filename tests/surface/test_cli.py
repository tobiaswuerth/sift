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
from tests.surface.commands import Terminal, indexed_notes, ran
from tests.surface.process import sift


@final
class Refusing(io.TextIOWrapper):
    def __init__(self, error: BaseException) -> None:
        super().__init__(io.BytesIO(), encoding="utf-8")
        self.error = error

    @override
    def write(self, s: str, /) -> int:
        raise self.error


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
        "follow_links": False,
        "agents": True,
        "state": "added",
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
                "contents": {
                    "documents": 0,
                    "chunks": 0,
                    "stored_bytes": store.stat().st_size,
                    "unreadable": 0,
                    "unreadable_files": [],
                    "links": 0,
                    "unresolved": 0,
                    "embedded": 0,
                    "model": None,
                    "fields": [],
                    "state": "counted",
                },
                "active": True,
                "agents": True,
            }
        ],
    }


def test_a_path_is_taken_from_the_user_s_folder_or_the_current_one(settings: Settings) -> None:
    user = settings.user
    assert user is not None
    for place in (user / "notes", settings.cwd / "papers"):
        place.mkdir()
    assert ran(settings, "add", "~/notes")[0] == 0
    assert ran(settings, "add", "papers")[0] == 0
    shown = ran(settings, "status")[1]
    assert str((user / "notes").resolve()) in shown
    assert str((settings.cwd / "papers").resolve()) in shown


def test_an_error_is_a_message_with_its_fix_and_no_traceback(settings: Settings, tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    assert ran(settings, "add", str(gone)) == (
        2,
        "",
        f"error  {gone} is no folder\n  fix  give the folder whose files the collection holds\n",
    )


def test_an_error_of_the_system_is_a_message(settings: Settings, notes: Path) -> None:
    settings.home.write_bytes(b"a file, not a folder")
    code, out, err = ran(settings, "add", str(notes))
    as_the_system_names_it = repr(str(settings.home))[:-1]
    assert (code, out) == (1, "")
    assert (err.startswith("error  ["), as_the_system_names_it in err) == (True, True)
    assert err.endswith("'\n  fix  check the path and its permissions\n")
    assert err.count("\n") == 2


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
    assert err.getvalue() == f"error  [Errno {errno.ENOSPC}] No space left on device\n  fix  free space on the disk\n"


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
    assert run(["--help"], settings, None, None) == 0
    assert run(["--version"], settings, None, None) == 0
    assert run(["--frobnicate"], settings, None, None) == 2


@pytest.mark.parametrize(
    "argv", [("get", "notes/name\udce9.md"), ("add", "bad\udce9", "--name", "ok"), ("status", "n\udce9")]
)
def test_an_argument_that_is_no_text_is_refused_with_what_to_do(settings: Settings, argv: tuple[str, ...]) -> None:
    code, out, err = ran(settings, *argv)
    assert (code, out) == (2, "")
    assert err.startswith("error  an argument is no text: '")
    assert "\\udce9" in err
    assert err.endswith("  fix  name files and folders in UTF-8, or rename this one\n")


@pytest.mark.parametrize("argv", [("add", ""), ("add", "notes", "--store", "")])
def test_an_empty_path_is_a_usage_error_not_the_current_folder(settings: Settings, argv: tuple[str, ...]) -> None:
    code, out, err = ran(settings, *argv)
    assert (code, out) == (2, "")
    assert "an empty path names nothing" in err
    assert not settings.home.exists()


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


def test_index_says_what_it_read_phase_by_phase(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    code, out, err = ran(settings, "index")
    lines = err.split("\n")
    assert (code, out) == (0, "")
    assert lines[0] == f"notes  {notes.resolve()}"
    assert lines[1].split()[:3] == ["scan", "2", "files"]
    assert lines[2].split()[:3] == ["read", "2", "files"]
    assert lines[3].startswith("  done  +2  ~0  -0  \N{MIDDLE DOT}  ")
    assert lines[4:] == [""]


def test_index_answers_in_json_when_asked(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    code, out, _ = ran(settings, "index", "notes", "--json")
    outcome = {"files": 2, "added": 2, "changed": 0, "removed": 0, "unreadable": 0, "state": "indexed"}
    assert (code, json.loads(out)) == (0, {"collections": [{"name": "notes", "outcome": outcome, "named": []}]})


def test_index_without_collections_says_how_to_add_one(settings: Settings) -> None:
    assert ran(settings, "index") == (0, "", "empty  no collections yet\n next  sift add FOLDER\n")


def test_get_prints_what_was_indexed(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    ran(settings, "index")
    assert ran(settings, "get", "notes/brush.md") == (0, "# Brush\n", "")
    assert ran(settings, "get", "notes/rl.md", "--lines", "3") == (0, "Reward shaping speeds up learning.\n", "")
    assert ran(settings, "get", "notes/rl.md", "notes/brush.md", "--lines", "1") == (
        0,
        "==> notes/rl.md <==\n# Reward shaping\n\n==> notes/robots/brush.md <==\n# Brush\n",
        "",
    )


def test_get_of_a_document_there_is_not_is_an_error(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    ran(settings, "index")
    code, out, err = ran(settings, "get", "notes/gone.md")
    assert (code, out) == (1, "")
    assert err.startswith("error  notes/gone.md: unknown document 'notes/gone.md'\n  fix  ")
    code, out, err = ran(settings, "get", "notes/gone.md", "--json")
    assert (code, json.loads(out)["unfound"][0]["id"], err) == (1, "notes/gone.md", "")


def test_colour_is_given_when_asked_and_kept_out_of_results(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    code, out, err = ran(settings, "index", "--color", "always")
    assert (code, out) == (0, "")
    assert err.startswith("\x1b[1mnotes\x1b[0m")
    assert "\x1b[" not in ran(settings, "status", "--color", "always")[1]


def test_index_fails_if_a_collection_was_skipped_however_it_answers(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    (settings.home / "collections" / "notes.sqlite").unlink()
    code, out, err = ran(settings, "index")
    assert (code, out) == (1, "")
    assert "  note  the store is missing: " in err
    code, out, _ = ran(settings, "index", "--json")
    assert (code, json.loads(out)["collections"][0]["outcome"]["state"]) == (1, "skipped")


def test_in_a_terminal_a_document_cannot_steer_it_and_its_json_never_can(
    settings: Settings, notes: Path, tmp_path: Path
) -> None:
    text = "# Steer\n\nclip\x1b]52;c;aGk=\x07 \N{RIGHT-TO-LEFT OVERRIDE}x\tend\n"
    (notes / "steer.md").write_text(text, encoding="utf-8")
    indexed_notes(settings, notes)
    assert ran(settings, "index")[0] == 0
    with Terminal((tmp_path / "out").open("w+b"), encoding="utf-8", newline="\n") as out:
        assert run(["get", "notes/steer.md"], settings, out, io.StringIO()) == 0
        out.flush()
        shown = (tmp_path / "out").read_bytes().decode()
    assert shown == "# Steer\n\nclip\\x1b]52;c;aGk=\\x07 \\u202ex\tend\n"
    piped = io.StringIO()
    assert run(["get", "notes/steer.md"], settings, piped, io.StringIO()) == 0
    assert piped.getvalue() == text
    as_json = io.StringIO()
    assert run(["get", "notes/steer.md", "--json"], settings, as_json, io.StringIO()) == 0
    assert "\N{RIGHT-TO-LEFT OVERRIDE}" not in as_json.getvalue()
    assert json.loads(as_json.getvalue())["found"][0]["text"] == text


def test_in_a_terminal_the_running_phase_is_drawn_in_place_and_cleared(
    settings: Settings, notes: Path, tmp_path: Path
) -> None:
    indexed_notes(settings, notes)
    with Terminal((tmp_path / "err").open("w+b"), encoding="utf-8", newline="\n") as err:
        assert run(["index", "--color", "never"], settings, io.StringIO(), err) == 0
        err.flush()
        drawn = (tmp_path / "err").read_bytes().decode()
    lines = drawn.split("\n")
    assert lines[0] == f"notes  {notes.resolve()}"
    assert "\r  scan" in lines[1]
    assert "\r  read           0 files   " in lines[2]
    shown = [line.rsplit("\r", 1)[-1] for line in lines]
    assert [line.split()[:3] for line in shown[1:3]] == [["scan", "2", "files"], ["read", "2", "files"]]
    assert shown[3].startswith("  done  +2  ~0  -0  ")
    assert shown[4:] == [""]
    assert "\x1b[" not in drawn
