import dataclasses
import io
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from sift.api.serving import WorkerAddress
from sift.domain.text.scans import OcrSetting
from sift.interfaces.cli.entries import ENTRIES
from sift.interfaces.cli.main import STOPPED_WATCHING, Host, run
from sift.interfaces.server.endpoint import own_endpoint, written
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
    ("mentions", "brush"),
    ("similar", "notes/rl.md"),
    ("similar", "notes/rl.md", "--json"),
    ("get", "notes/none.md", "--json"),
)
WRITES = (
    ("add", "../more", "--name", "more"),
    ("index", "more"),
    ("index", "--file", "../more/m.md"),
    ("move", "more", "kept", "--json"),
    ("rm", "more", "--json"),
    ("rm", "none", "--json"),
    ("add", "../more/m.md", "--json"),
)
NOT_COMPARED = MappingProxyType(
    {
        "serve": "it starts the home process",
        "ui": "it opens the home process's pages in the browser",
        "jobs": "it asks the home process alone",
        "worker": "it lends this machine's models to others, beside any home process",
        "bench": "it measures searches in its own process",
        "import": "it reads a home of sift 1.x in its own process",
        "embed": "it needs models",
        "vsearch": "it needs models",
        "query": "it needs models",
    }
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
    (notes / "arm.md").write_text("# Arm\n\nThe arm holds a brush.\n", encoding="utf-8")
    indexed_notes(settings, notes)
    assert ran(settings, "index")[0] == 0
    alone = [ran(settings, *argv) for argv in ASKED]
    asked = list[str]()
    with running(settings, counted(asked)):
        served = [ran(settings, *argv) for argv in ASKED]
    assert served == alone
    assert asked == [
        *("status", "status", "get", "get", "search", "links", "unresolved", "deactivate", "activate"),
        *("mentions", "similar", "similar", "get"),
    ]
    assert [code for code, _, _ in alone] == [0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1]


def _homeless(home: Path, said: str) -> str:
    whole = str(home.resolve())
    return said.replace(json.dumps(whole)[1:-1], "<home>").replace(whole, "<home>")


def _seen_from(home: Path, done: tuple[int, str, str]) -> tuple[int, str, str]:
    code, out, err = done
    return code, _homeless(home, out), _homeless(home, err)


def test_writes_from_another_folder_are_answered_served_as_alone(settings: Settings, tmp_path: Path) -> None:
    (tmp_path / "more").mkdir()
    (tmp_path / "more" / "m.md").write_text("# More\n\nMore notes.\n", encoding="utf-8")
    alone, served = (dataclasses.replace(settings, home=tmp_path / name) for name in ("alone", "served"))
    said_alone = [_seen_from(alone.home, ran(alone, *argv)) for argv in WRITES]
    asked = list[str]()
    with running(served, counted(asked)):
        said_served = [_seen_from(served.home, ran(served, *argv)) for argv in WRITES]
    assert said_served == said_alone
    assert asked == ["add", "index", "index", "move", "rm", "rm", "add"]
    assert [code for code, _, _ in said_alone] == [0, 0, 0, 0, 0, 1, 2]


def test_every_command_is_compared_served_and_alone_or_says_why_not() -> None:
    compared = {argv[0] for argv in (*ASKED, *WRITES)}
    assert (compared | NOT_COMPARED.keys(), compared & NOT_COMPARED.keys()) == (set(ENTRIES), set())


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


@pytest.mark.parametrize("contract, spoken", [((1, 0), "1.0"), ((3, 0), "3.0"), ((), "none"), ((2, 9), None)])
def test_a_home_process_whose_answers_this_sift_cannot_read_is_refused_saying_what_to_do(
    settings: Settings, notes: Path, contract: tuple[int, ...], spoken: str | None
) -> None:
    indexed_notes(settings, notes)
    alone = ran(settings, "status")
    with running(settings, of="1.0.0") as server:
        written(own_endpoint(settings), dataclasses.replace(server.endpoint, contract=contract))
        found = ran(settings, "status")
    said = f"error  the home process runs sift 1.0.0, whose contract {spoken} this sift {VERSION} (2.0) cannot read\n"
    fix = "  fix  stop sift serve where it runs, and start it again with this sift\n"
    assert found == (alone if spoken is None else (1, "", said + fix))


def test_a_setting_the_home_process_does_not_share_is_noted(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    off = dataclasses.replace(settings, ocr=OcrSetting.OFF)
    alone = ran(settings, "status")
    with running(off):
        noted = [ran(each, "status") for each in (settings, dataclasses.replace(off, gpus="0"), off)]
    fix = "  fix  stop sift serve to use yours, or start it again with them" + NEWLINE
    assert noted == [
        (alone[0], alone[1], " note  the home process uses SIFT_OCR=off" + NEWLINE + fix + alone[2]),
        (alone[0], alone[1], " note  the home process uses CUDA_VISIBLE_DEVICES unset" + NEWLINE + fix + alone[2]),
        alone,
    ]


def test_a_worker_the_home_process_sends_work_to_is_noted_without_its_key(settings: Settings, notes: Path) -> None:
    indexed_notes(settings, notes)
    elsewhere = dataclasses.replace(settings, embed_remote=WorkerAddress("workstation", 8766, "s3cret", "p1n"))
    alone = ran(settings, "status")
    with running(elsewhere):
        noted = ran(settings, "status")
    fix = "  fix  stop sift serve to use yours, or start it again with them" + NEWLINE
    said = " note  the home process uses SIFT_EMBED_REMOTE=https://workstation:8766" + NEWLINE
    assert noted == (alone[0], alone[1], said + fix + alone[2])


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
        assert (out.getvalue(), err.getvalue()) == ("", STOPPED_WATCHING + NEWLINE)
        again = ran(settings, "index")
    assert (again[0], "done  +0  ~0  -0" in again[2]) == (0, True)
