import dataclasses
import io
from pathlib import Path

from sift.interfaces.cli.main import STOPPED_WATCHING, Host, run
from sift.settings import Settings
from tests.serving import VERSION, counted, running
from tests.surface.commands import indexed_notes

NEWLINE = "\n"
ASKED = (
    ("status",),
    ("status", "--json"),
    ("get", "notes/rl.md"),
    ("get", "notes/none.md"),
    ("search", "reward"),
    ("search", "reward", "--json"),
    ("search", "reward", "-k", "0"),
    ("links", "rl"),
    ("unresolved",),
    ("deactivate", "notes"),
    ("activate", "notes"),
)


def _still() -> float:
    return 0.0


def _stopped() -> float:
    raise KeyboardInterrupt


def ran(settings: Settings, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run(argv, settings, out, err, Host(clock=_still))
    return code, out.getvalue(), err.getvalue()


def test_a_command_sent_to_the_home_process_is_answered_as_it_is_alone(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    assert ran(settings, "index")[0] == 0
    alone = [ran(settings, *argv) for argv in ASKED]
    asked = list[str]()
    with running(settings, counted(asked)):
        served = [ran(settings, *argv) for argv in ASKED]
    assert served == alone
    assert asked == ["status", "status", "get", "get", "search", "links", "unresolved", "deactivate", "activate"]
    assert [code for code, _, _ in alone] == [0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]


def test_a_failure_of_the_system_is_said_alike_served_and_alone(
    settings: Settings, notes: Path, tmp_path: Path
) -> None:
    indexed_notes(settings, notes)
    (tmp_path / "a.txt").write_text("not a folder\n", encoding="utf-8")
    (tmp_path / "other").mkdir()
    store = str(tmp_path / "a.txt" / "other.sqlite")
    argvs = [
        ("add", str(tmp_path / "other"), "--store", store),
        ("add", str(tmp_path / "other"), "--store", store, "--json"),
    ]
    alone = [ran(settings, *argv) for argv in argvs]
    with running(settings):
        served = [ran(settings, *argv) for argv in argvs]
    assert served == alone
    assert ([code for code, _, _ in alone], '"kind": "os"' in alone[1][1]) == ([1, 1], True)


def test_an_index_sent_to_the_home_process_says_its_progress_as_it_does_alone(
    settings: Settings, notes: Path, tmp_path: Path
) -> None:
    served = dataclasses.replace(settings, home=tmp_path / "served")
    indexed_notes(settings, notes)
    assert ran(served, "add", str(notes))[0] == 0
    alone, asked = ran(settings, "index"), list[str]()
    with running(served, counted(asked)):
        assert ran(served, "index") == alone
    assert (asked, "done  +2  ~0  -0" in alone[2]) == (["index"], True)


def test_a_home_process_of_another_sift_is_refused_saying_what_to_do(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    with running(settings, of="1.0.0"):
        refused = ran(settings, "status")
    said = f"error  the home process runs sift 1.0.0, this is sift {VERSION}\n"
    assert refused == (1, "", said + "  fix  stop sift serve where it runs, and start it again with this sift\n")


def test_an_interrupt_while_watching_a_served_move_says_it_goes_on(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    with running(settings):
        out, err = io.StringIO(), io.StringIO()
        assert run(("move", "notes", str(settings.cwd / "elsewhere")), settings, out, err, Host(clock=_stopped)) == 130
    assert (out.getvalue(), err.getvalue()) == ("", STOPPED_WATCHING + NEWLINE)


def test_an_interrupt_while_watching_the_home_process_leaves_the_command_to_it(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    with running(settings):
        out, err = io.StringIO(), io.StringIO()
        assert run(("index",), settings, out, err, Host(clock=_stopped)) == 130
        assert (out.getvalue(), err.getvalue()) == ("", "stopped watching: the home process goes on with it\n")
        again = ran(settings, "index")
    assert (again[0], "done  +0  ~0  -0" in again[2]) == (0, True)
