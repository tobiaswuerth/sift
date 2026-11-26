import io
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from sift.interfaces.cli.main import Host, run
from sift.settings import Settings
from tests.surface.commands import ran


def logged(settings: Settings, command: str) -> str:
    (log,) = (settings.home / "logs").glob(f"{command}-*.log")
    return log.read_text(encoding="utf-8")


def test_commands_that_change_collections_keep_logs_and_the_others_do_not(settings: Settings, notes: Path) -> None:
    (notes / "rl.md").write_bytes(b"# Reward shaping\n\nShaping speeds up learning.\n")
    for argv in (
        ("add", str(notes)),
        ("index", "--json"),
        ("search", "shaping"),
        ("status",),
        ("get", "notes/rl.md"),
        ("deactivate", "notes"),
        ("activate", "notes"),
    ):
        assert ran(settings, *argv)[0] == 0
    commands = sorted(path.name.split("-20", 1)[0] for path in (settings.home / "logs").iterdir())
    assert commands == ["activate", "add", "deactivate", "index"]


def test_a_log_says_which_sift_ran_what_it_said_plainly_and_how_it_ended(settings: Settings, notes: Path) -> None:
    (notes / "data.jsonl").write_bytes(b'{"id": 1, "text": "a record"}\nnot json\n')
    ran(settings, "add", str(notes))
    assert ran(settings, "index", "--color", "always")[0] == 0
    said = logged(settings, "index")
    assert f" INFO sift: sift {version('sift-rag')}, index\n" in said
    assert "data.jsonl: 1 line skipped (line 2: not JSON)\n" in said
    assert said.endswith(" INFO sift: ended with exit code 0\n")
    assert "\x1b[" not in said


def test_without_stderr_a_command_s_log_still_holds_what_it_said(settings: Settings, notes: Path) -> None:
    assert run(("add", str(notes)), settings, io.StringIO(), None) == 0
    assert f"added  notes  {notes.resolve()}\n" in logged(settings, "add")


def _interrupted() -> float:
    raise KeyboardInterrupt


def _broken() -> float:
    raise ValueError("a clock that broke")


def test_a_command_stopped_or_broken_logs_how_it_ended(settings: Settings, notes: Path) -> None:
    ran(settings, "add", str(notes))
    stopped = Host(clock=_interrupted, now=lambda: datetime(2026, 11, 24, 12, 0, 1, tzinfo=UTC))
    assert run(("index",), settings, io.StringIO(), io.StringIO(), stopped) == 130
    broken = Host(clock=_broken, now=lambda: datetime(2026, 11, 24, 12, 0, 2, tzinfo=UTC))
    for argv in (("index",), ("search", "anything"), ("search", "anything", "--json")):
        out, err = io.StringIO(), io.StringIO()
        assert run(argv, settings, out, err, broken) == 1
        traced, _, said = err.getvalue().rpartition("error  ")
        assert traced.startswith("Traceback (most recent call last):")
        assert said == "sift failed: ValueError: a clock that broke\n  fix  report it, with the lines above it\n"
        assert ('"code": "internal"' in out.getvalue(), out.getvalue() == "") == (
            "--json" in argv,
            "--json" not in argv,
        )
    logs = settings.home / "logs"
    assert not list(logs.glob("search-*"))
    assert (logs / "index-2026-11-24_120001.log").read_text(encoding="utf-8").endswith(" ERROR sift: stopped\n")
    said = (logs / "index-2026-11-24_120002.log").read_text(encoding="utf-8")
    assert (" ERROR sift: failed\n" in said, said.endswith("ValueError: a clock that broke\n")) == (True, True)


def test_a_command_refused_logs_its_error_and_how_it_ended(settings: Settings) -> None:
    code, _, err = ran(settings, "rm", "nope")
    said = logged(settings, "rm")
    lines = err.split("\n")[:-1]
    assert (f" ERROR sift: {lines[0]}\n" in said, lines[0].startswith("error  ")) == (True, True)
    assert all(f" INFO sift: {line}\n" in said for line in lines[1:])
    assert said.endswith(f" ERROR sift: ended with exit code {code}\n")


def test_a_log_that_cannot_be_made_is_a_json_error_when_json_is_asked(settings: Settings, notes: Path) -> None:
    settings.home.mkdir(parents=True, exist_ok=True)
    (settings.home / "logs").write_bytes(b"")
    code, out, err = ran(settings, "add", str(notes), "--json")
    assert (code, json.loads(out)["error"]["kind"]) == (1, "os")
    assert err.startswith("error  ")
