"""`update`: telling of a newer sift, and updating to it on a yes. PyPI, the terminal, the processes started and the
waits are fakes, but for one real run of the process that updates."""

import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
from types import SimpleNamespace

import pytest

from sift import __version__, cli, console, update

MAJOR, MINOR, PATCH = update._version(__version__)
NEWER = f"{MAJOR}.{MINOR}.{PATCH + 1}"
NEXT_MAJOR = f"{MAJOR + 1}.0.0"
NO_PROCESS = 4_194_305  # above Linux's highest process id and macOS's, and odd, which none on Windows is
UPDATER = 4321  # the process `installed` starts to update


def waiting(pid=None, **more):
    """An update that waits for the sifts of its installation to stop, its process `pid`: by default this one, which
    runs."""
    pending = {"version": NEWER, "since": time.time(), "pid": os.getpid() if pid is None else pid, "upgrading": False}
    return pending | more


class Terminal:
    """A stream of a terminal; read, it answers `answer`."""

    def __init__(self, answer=None):
        self.answer = answer

    def isatty(self):
        return True

    def readline(self):
        assert self.answer is not None, "asked, though nothing should be"
        return self.answer


class Pipe(Terminal):
    def isatty(self):
        return False


TERMINAL = (Terminal(), Terminal(), Terminal())


@pytest.fixture
def installed(monkeypatch, home):
    """This sift as uv installed it as a tool, uv on the PATH; the processes it would start, recorded."""
    started = []
    monkeypatch.setattr(update, "upgrade_command", lambda: [str(home / "uv"), "tool", "upgrade", "sift-rag"])

    def start(args):
        started.append(args)
        return SimpleNamespace(pid=UPDATER)

    monkeypatch.setattr(update, "start", start)
    return started


def kept(home):
    return json.loads((home / "update.json").read_text(encoding="utf-8"))


def keep(home, **state):
    update.change(home / "update.json", **state)


def answer(monkeypatch, text):
    monkeypatch.setattr(sys, "stdin", Terminal(text))


# --- offering -------------------------------------------------------------------


def test_a_newer_sift_is_offered_once(installed, home, monkeypatch, capsys):
    keep(home, latest=NEWER, checked=time.time())
    answer(monkeypatch, "n\n")
    update.offer("status", TERMINAL)
    assert f"sift {NEWER} is out (this is {__version__}). Update now? [y/N] " in capsys.readouterr().err
    assert kept(home)["asked"] == NEWER
    update.offer("status", TERMINAL)  # asked once: no question now, which Terminal() would fail
    assert installed == []


def test_yes_starts_the_update_and_stops_the_command(installed, home, monkeypatch, capsys):
    keep(home, latest=NEWER, checked=time.time())
    answer(monkeypatch, "Y\n")
    with pytest.raises(SystemExit) as stopped:
        update.offer("status", TERMINAL)
    assert stopped.value.code == 0
    assert "stop any `sift serve` and `sift worker` too" in capsys.readouterr().err
    [(what, spec)] = installed
    spec = json.loads(spec)
    assert what == "finish"
    assert spec["version"] == NEWER
    assert spec["command"] == [str(home / "uv"), "tool", "upgrade", "sift-rag"]
    assert spec["log"].startswith(str(home / "logs" / "update-"))
    pending = kept(home)["pending"]
    assert (pending["version"], pending["pid"], pending["upgrading"]) == (NEWER, UPDATER, False)
    assert spec["since"] == pending["since"]


def test_an_update_that_can_t_be_started_leaves_none_waiting(installed, home, monkeypatch):
    def start(args):
        raise OSError("no Python to run it")

    monkeypatch.setattr(update, "start", start)
    keep(home, latest=NEWER, checked=time.time())
    answer(monkeypatch, "y\n")
    with pytest.raises(OSError, match="no Python"):
        update.offer("status", TERMINAL)
    assert "pending" not in kept(home)  # else every command of the next week said that it waits


def test_while_an_update_waits_it_is_said(installed, home, capsys):
    keep(home, latest=NEWER, pending=waiting(), checked=0)
    update.offer("status", TERMINAL)
    assert f"sift updates to {NEWER} once no sift of this installation runs" in capsys.readouterr().err
    assert installed == []  # nor checks meanwhile


@pytest.mark.parametrize(
    "command, streams, check",
    [
        ("status", TERMINAL, True),
        ("serve", TERMINAL, True),
        ("search", (Pipe(), Pipe(), Pipe()), True),
        ("index", TERMINAL, False),
    ],
)
def test_a_command_started_while_sift_is_upgraded_stops(installed, home, configure, command, streams, check):
    # The installer replaces sift's files just then: a command running on them fails halfway, or makes the upgrade
    # fail. And `y` stopped the command typed, so it is typed again at once.
    configure(update_check=check)
    keep(home, pending=waiting(upgrading=True))
    with pytest.raises(SystemExit, match=f"sift is being updated to {NEWER} right now: run this again in a moment"):
        update.offer(command, streams)
    assert kept(home)["pending"]["upgrading"]


def test_an_upgrade_whose_process_is_gone_stops_no_command(installed, home, monkeypatch, capsys):
    keep(home, latest=NEWER, checked=time.time(), asked=NEWER, pending=waiting(NO_PROCESS, upgrading=True))
    update.offer("serve", TERMINAL)  # nor says anything: a server asks nothing
    assert capsys.readouterr().err == ""
    answer(monkeypatch, "n\n")
    update.offer("status", TERMINAL)
    assert f"the update to {NEWER} stopped before it was done" in capsys.readouterr().err


def test_an_update_whose_process_is_gone_is_offered_again(installed, home, monkeypatch, capsys):
    # A restart of the machine ends the process that waits: nothing updated, and for a week nothing said so.
    keep(home, latest=NEWER, checked=time.time(), asked=NEWER, pending=waiting(NO_PROCESS))
    answer(monkeypatch, "n\n")
    update.offer("status", TERMINAL)
    said = capsys.readouterr().err
    assert f"the update to {NEWER} stopped before it was done" in said
    assert "Update now?" in said
    assert "pending" not in kept(home)


def test_an_update_that_ends_as_it_is_looked_at_says_how_it_went(installed, home, monkeypatch, capsys):
    def ended(pid):  # between the reading of the state and the look at its process
        keep(home, pending=None, result={"version": __version__, "ok": True, "log": "update.log"})
        return False

    monkeypatch.setattr(update, "_runs", ended)
    keep(home, checked=time.time(), asked=__version__, pending=waiting() | {"version": __version__})
    update.offer("status", TERMINAL)
    said = capsys.readouterr().err
    assert f"sift updated to {__version__}" in said
    assert "stopped before" not in said


def test_an_update_waiting_far_too_long_is_forgotten(installed, home, monkeypatch, capsys):
    # Its process is long gone, and another has its id by now.
    keep(home, latest=NEWER, checked=time.time(), pending=waiting(since=time.time() - 9 * update.DAY))
    answer(monkeypatch, "\n")  # offered again
    update.offer("status", TERMINAL)
    assert "Update now?" in capsys.readouterr().err
    assert "pending" not in kept(home)


def test_a_new_major_version_is_only_told_of(installed, home, capsys):
    keep(home, latest=NEXT_MAJOR, checked=time.time())
    update.offer("status", TERMINAL)
    assert f"sift {NEXT_MAJOR} is out, a new major version: read {update.CHANGELOG}" in capsys.readouterr().err
    assert kept(home)["asked"] == NEXT_MAJOR


def test_without_its_installer_on_the_path_the_command_is_said(installed, home, monkeypatch, capsys):
    monkeypatch.setattr(update, "upgrade_command", lambda: ["uv", "tool", "upgrade", "sift-rag"])
    keep(home, latest=NEWER, checked=time.time())
    update.offer("status", TERMINAL)
    assert "run `uv tool upgrade sift-rag` to update" in capsys.readouterr().err


@pytest.mark.parametrize("latest", [__version__, f"{MAJOR}.{MINOR}.{PATCH + 1}rc1", None])
def test_nothing_newer_is_not_offered(installed, home, latest):
    keep(home, latest=latest, checked=time.time())
    update.offer("status", TERMINAL)  # Terminal() fails if asked
    assert "asked" not in kept(home)


def test_pypi_is_asked_once_a_day(installed, home):
    keep(home, checked=time.time() - update.DAY)
    update.offer("status", TERMINAL)
    assert installed == [["check", str(home / "update.json")]]
    assert kept(home)["checked"] > time.time() - 60
    update.offer("status", TERMINAL)
    assert len(installed) == 1


def test_a_first_command_asks_pypi(installed, home):
    update.offer("status", TERMINAL)
    assert installed == [["check", str(home / "update.json")]]


@pytest.mark.parametrize(
    "command, streams, check",
    [
        ("serve", TERMINAL, True),
        ("worker", TERMINAL, True),
        ("status", (Pipe(), Terminal(), Terminal()), True),
        ("status", (Terminal(), Pipe(), Terminal()), True),
        ("status", TERMINAL, False),
    ],
)
def test_nothing_happens_but_in_a_terminal_with_checks_on(installed, home, configure, command, streams, check):
    configure(update_check=check)
    keep(home, latest=NEWER, checked=0)
    update.offer(command, streams)
    assert installed == []
    assert "asked" not in kept(home)


def test_sift_run_from_its_source_offers_nothing(home, monkeypatch):
    monkeypatch.setattr(update, "upgrade_command", lambda: None)
    monkeypatch.setattr(update, "start", lambda args: pytest.fail("started"))
    update.offer("status", TERMINAL)
    assert not (home / "update.json").exists()


@pytest.mark.parametrize(
    "version, ok, said",
    [
        (__version__, True, f"sift updated to {__version__}; restart any `sift serve` and `sift worker`"),
        (NEWER, True, f"the update to {NEWER} ran, but this is sift {__version__} still: see "),
        (NEWER, False, f"sift couldn't update to {NEWER}: see "),
    ],
)
def test_how_an_update_went_is_said_once(installed, home, capsys, version, ok, said):
    keep(home, checked=time.time(), asked=NEWER, result={"version": version, "ok": ok, "log": "update.log"})
    update.offer("status", TERMINAL)
    assert said in capsys.readouterr().err
    update.offer("status", TERMINAL)
    assert capsys.readouterr().err == ""


def test_the_command_line_offers_before_the_command(home, monkeypatch):
    offered = []
    monkeypatch.setattr(update, "offer", offered.append)
    cli.main(["status"])
    assert offered == ["status"]


def test_trouble_with_the_update_s_state_is_said_and_the_command_runs(home, monkeypatch, capsys):
    # An update is an offer, the command what was asked: a state that can't be read or written stopped every command.
    def offer(command):
        raise PermissionError(13, "Access is denied", str(home / "update.json"))

    monkeypatch.setattr(update, "offer", offer)
    cli.main(["status"])
    told = capsys.readouterr()
    assert "sift couldn't look for an update: [Errno 13] Access is denied" in told.err
    assert json.loads(told.out)["collections"] == []


def test_a_command_stopped_for_an_upgrade_stays_stopped(home, monkeypatch):
    def offer(command):
        raise SystemExit("sift is being updated")

    monkeypatch.setattr(update, "offer", offer)
    with pytest.raises(SystemExit, match="sift is being updated"):
        cli.main(["status"])


# --- asking PyPI ----------------------------------------------------------------


def test_the_check_keeps_the_latest_version(home, monkeypatch):
    def urlopen(request, timeout):
        assert request.full_url == update.PYPI
        assert request.get_header("User-agent") == f"sift/{__version__}"  # nothing of the user
        return io.BytesIO(json.dumps({"info": {"version": NEWER}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    keep(home, failed="earlier")
    update.check(home / "update.json")
    state = kept(home)
    assert state["latest"] == NEWER
    assert state["checked"] > time.time() - 60
    assert "failed" not in state


@pytest.mark.parametrize("trouble", [urllib.error.URLError("offline"), TimeoutError("timed out"), ValueError("html")])
def test_a_check_that_fails_waits_a_day_and_says_why(home, monkeypatch, trouble):
    def urlopen(request, timeout):
        raise trouble

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    keep(home, latest=NEWER)
    update.check(home / "update.json")
    state = kept(home)
    assert state["latest"] == NEWER
    assert state["checked"] > time.time() - 60
    assert state["failed"].startswith(type(trouble).__name__)


def test_entries_of_another_shape_count_as_none(home):
    # Edited by hand, or by another version of sift: an entry sift can't use would stop every command.
    result = {"version": NEWER, "ok": True, "log": "update.log"}
    keep(home, checked="today", latest=12, pending={"version": NEWER}, result=result, other=1)
    assert update.read(home / "update.json") == {"result": result}
    keep(home, result={"version": NEWER, "ok": "yes", "log": "update.log"})
    assert update.read(home / "update.json") == {}


def test_a_state_cut_short_counts_as_none(home):
    (home / "update.json").parent.mkdir(parents=True, exist_ok=True)
    (home / "update.json").write_text('{"latest": "1.', encoding="utf-8")
    assert update.read(home / "update.json") == {}
    (home / "update.json").write_text("[]", encoding="utf-8")
    assert update.read(home / "update.json") == {}


def refusing(real, times):
    """`real`, refused `times` times first, as Windows refuses a file another process holds; and the refusals."""
    refused = []

    def refuse(*args, **kwargs):
        if len(refused) < times:
            refused.append(args)
            raise PermissionError(13, "The process cannot access the file")
        return real(*args, **kwargs)

    return refuse, refused


def test_a_state_another_process_reads_just_then_is_written_once_it_has(home, monkeypatch):
    # Windows refuses to replace a file that another process has open, as a command reading the state has, for
    # microseconds.
    replace, refused = refusing(os.replace, times=2)
    monkeypatch.setattr(update.os, "replace", replace)
    update.change(home / "update.json", latest=NEWER)
    assert len(refused) == 2
    assert kept(home) == {"latest": NEWER}


def test_a_state_another_process_replaces_just_then_is_read_once_it_has(home, monkeypatch):
    keep(home, latest=NEWER)
    read_text, refused = refusing(update.Path.read_text, times=2)
    monkeypatch.setattr(update.Path, "read_text", read_text)
    assert update.read(home / "update.json") == {"latest": NEWER}
    assert len(refused) == 2


def test_a_state_that_stays_refused_is_an_error_and_leaves_nothing_behind(home, monkeypatch):
    # A folder that may not be written: tried for a moment, not for ever, and the file written beside the state goes.
    keep(home, latest=NEWER)
    replace, refused = refusing(os.replace, times=10**9)
    monkeypatch.setattr(update.os, "replace", replace)
    monkeypatch.setattr(update, "SHARING", 0.05)
    with pytest.raises(PermissionError):
        update.change(home / "update.json", asked=NEWER)
    assert len(refused) > 1
    assert [file.name for file in home.iterdir()] == ["update.json"]
    assert kept(home) == {"latest": NEWER}


def test_commands_at_once_read_and_change_the_state_without_error(home):
    # On Windows a read and a replace at the same moment refuse each other: of 3,000 changes beside two readers,
    # 2,591 failed, and 1,295 reads.
    path, errors, done = home / "update.json", [], threading.Event()
    update.change(path, checked=0.0)

    def reading():
        while not done.is_set():
            try:
                assert "checked" in update.read(path)
            except Exception as error:  # handed to the test
                errors.append(error)

    readers = [threading.Thread(target=reading) for _ in range(2)]
    for reader in readers:
        reader.start()
    try:
        for n in range(100):
            update.change(path, checked=float(n))
    finally:
        done.set()
        for reader in readers:
            reader.join()
    assert errors == []
    assert kept(home) == {"checked": 99.0}


# --- how sift was installed ------------------------------------------------------


@pytest.fixture
def environment(tmp_path):
    """A virtual environment holding sift."""
    prefix = tmp_path / "env"
    (prefix / "Lib" / "site-packages" / "sift").mkdir(parents=True)
    return prefix, prefix / "Lib" / "site-packages" / "sift" / "update.py"


def test_a_uv_tool_is_upgraded_by_uv(environment):
    prefix, module = environment
    (prefix / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
    command = update.upgrade_command(prefix, module, which=lambda name: f"/bin/{name}")
    assert command == ["/bin/uv", "tool", "upgrade", "sift-rag"]
    assert update.upgrade_command(prefix, module, which=lambda name: None)[0] == "uv"


def test_a_pipx_installation_is_upgraded_by_pipx(tmp_path):
    prefix = tmp_path / "pipx" / "venvs" / "sift-rag"
    (prefix / "lib" / "sift").mkdir(parents=True)
    command = update.upgrade_command(prefix, prefix / "lib" / "sift" / "update.py", which=lambda name: f"/bin/{name}")
    assert command == ["/bin/pipx", "upgrade", "sift-rag"]


def test_another_environment_is_upgraded_by_pip(environment):
    command = update.upgrade_command(*environment)
    assert command == [sys.executable, "-m", "pip", "install", "--upgrade", "sift-rag"]


def test_sift_run_from_its_source_has_no_upgrade(environment, tmp_path):
    prefix, _ = environment
    assert update.upgrade_command(prefix, tmp_path / "checkout" / "sift" / "update.py") is None


def test_a_running_sift_holds_its_environment_s_python_and_launcher(tmp_path, monkeypatch):
    launcher = tmp_path / "sift.exe"
    launcher.write_bytes(b"")
    # As a launcher of pip's or uv's leaves sys.argv[0]: it cuts ".exe" off before sift starts, so the name as it is
    # never ended in it, and a sift outside a virtual environment held nothing the update waited for.
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "sift"), "status"])
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "python"))
    assert update.held_files() == [update.Path(sys.executable), launcher]
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)  # no virtual environment
    assert update.held_files() == [launcher]
    monkeypatch.setattr(sys, "argv", [str(launcher), "status"])  # a launcher that leaves its name whole
    assert update.held_files() == [launcher]
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "sift" / "__main__.py"), "status"])  # python -m sift: none
    assert update.held_files() == []


# --- the processes it starts ------------------------------------------------------


@pytest.mark.parametrize(
    "platform, apart",
    [("win32", {"creationflags": 0x08000200}), ("linux", {"start_new_session": True})],  # Windows: a hidden console
)
def test_a_process_started_outlives_this_one_without_its_terminal(monkeypatch, platform, apart):
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kwargs: started.append((args, kwargs)))
    update.start(["check", "update.json"], platform)
    [(args, kwargs)] = started
    assert args[1:4] == ["-I", "-B", "-c"]
    assert args[-2:] == ["check", "update.json"]
    streams = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    assert kwargs == streams | apart | {"cwd": update.Path(args[0]).anchor}


def test_main_runs_a_check_or_an_update(monkeypatch):
    done = []
    monkeypatch.setattr(update, "check", lambda path: done.append(("check", path)))
    monkeypatch.setattr(update, "finish", lambda spec: done.append(("finish", spec)))
    update.main(["check", "update.json"])
    update.main(["finish", '{"version": "9.9.9"}'])
    assert done == [("check", update.Path("update.json")), ("finish", {"version": "9.9.9"})]


@pytest.fixture
def spec(home, tmp_path):
    """An update of this sift, its upgrade a Python that prints."""
    return {
        "version": NEWER,
        "command": [sys.executable, "-c", "print('upgraded')"],
        "since": 1_700_000_000.0,
        "pid": 1234,
        "held": [],
        "log": str(home / "logs" / "update-1.log"),
    }


class Kernel:
    """Windows' kernel32, as far as waiting for a process goes: it runs for `looks` more looks."""

    def __init__(self, looks):
        self.looks, self.opened, self.closed = looks, 0, 0

    def OpenProcess(self, access, inherit, pid):
        assert access == update.SYNCHRONIZE
        self.opened += self.looks is not None
        return 0 if self.looks is None else 7

    def WaitForSingleObject(self, handle, milliseconds):
        self.looks -= 1
        return update.WAIT_TIMEOUT if self.looks >= 0 else 0

    def CloseHandle(self, handle):
        self.closed += 1


def test_the_update_waits_for_the_sift_that_asked_then_upgrades(spec, home, monkeypatch):
    looks = iter([None, None, ProcessLookupError(), ProcessLookupError()])  # the last: once more before it upgrades

    def kill(pid, signal):
        assert (pid, signal) == (1234, 0)  # only asks
        if isinstance(error := next(looks), Exception):
            raise error

    monkeypatch.setattr(update.os, "kill", kill)
    monkeypatch.setattr(update, "POLL", 0)
    update.finish(spec, platform="linux")
    log = (home / "logs" / "update-1.log").read_text(encoding="utf-8")
    command = console.command(*spec["command"])  # quoted for the shell of the OS it runs on
    assert log.splitlines() == [command, "upgraded", "exit code 0"]
    assert kept(home)["result"] == {"version": NEWER, "ok": True, "log": spec["log"]}


def test_while_it_upgrades_the_state_says_so_with_its_process(spec, home, monkeypatch):
    monkeypatch.setattr(update, "_running", lambda *args: False)
    shown = "import json, sys; print(json.dumps(json.load(open(sys.argv[1], encoding='utf-8'))['pending']))"
    spec["command"] = [sys.executable, "-c", shown, str(home / "update.json")]
    update.finish(spec, platform="linux")
    pending = json.loads((home / "logs" / "update-1.log").read_text(encoding="utf-8").splitlines()[1])
    assert pending.pop("since") > time.time() - 60  # since it upgrades, not since it was agreed to
    assert pending == {"version": NEWER, "pid": os.getpid(), "upgrading": True}
    assert "pending" not in kept(home)


def test_an_upgrade_past_its_time_stops_no_command(installed, home, capsys):
    # The installer is stopped when its time is up: a process running after it has the id of the one that upgraded.
    keep(home, pending=waiting(upgrading=True, since=time.time() - update.RUN - 1))
    update.offer("serve", TERMINAL)
    update.offer("status", TERMINAL)
    assert f"sift updates to {NEWER} once no sift" in capsys.readouterr().err


def test_a_sift_started_just_before_the_upgrade_is_waited_for_too(spec, home, monkeypatch):
    # It started before the state said that sift upgrades, so it runs: the upgrade waits again, and says no more
    # that it upgrades, or every command of the installation would stop meanwhile.
    looks, states = iter([False, True, True, False, False]), []

    def running(*args):
        states.append(update.read(home / "update.json").get("pending", {}).get("upgrading"))
        return next(looks)

    monkeypatch.setattr(update, "_running", running)
    monkeypatch.setattr(update, "POLL", 0)
    update.finish(spec, platform="linux")
    assert states == [None, True, False, False, True]
    assert kept(home)["result"]["ok"]


def test_a_process_of_another_user_counts_as_running(monkeypatch):
    def kill(pid, signal):
        raise PermissionError

    monkeypatch.setattr(update.os, "kill", kill)
    assert update._alive(1234)


def test_on_windows_the_update_waits_for_every_sift_of_the_installation(spec, home, tmp_path, monkeypatch):
    serve = tmp_path / "python.exe"
    serve.write_bytes(b"MZ")
    spec["held"] = [str(serve), str(tmp_path / "gone.exe")]
    opened, real = [], open

    def held_twice(file, mode="r", *args, **kwargs):  # a `sift serve` holds the environment's Python twice more
        if file == str(serve) and mode == "r+b" and opened.append(file) is None and len(opened) <= 2:
            raise PermissionError(13, "in use")
        return real(file, mode, *args, **kwargs)

    monkeypatch.setattr(update, "open", held_twice, raising=False)
    monkeypatch.setattr(update, "POLL", 0)
    kernel = Kernel(looks=1)  # the sift that asked stops after one look
    update.finish(spec, platform="win32", kernel32=kernel)
    assert len(opened) == 4  # in use, in use, free, and free still once the state says that sift upgrades
    assert kernel.closed == kernel.opened == 5  # one look while it ran, four after
    assert serve.read_bytes() == b"MZ"  # looked at, never written
    assert kept(home)["result"]["ok"]


def test_on_windows_a_process_gone_can_t_be_opened():
    assert not update._alive_on_windows(1234, Kernel(looks=None))


def test_on_windows_its_kernel_is_asked(monkeypatch):
    import ctypes

    kernel = Kernel(looks=0)  # stopped
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel), raising=False)  # only Windows has it
    assert not update._alive_on_windows(1234)
    assert kernel.closed == kernel.opened == 1


def test_an_update_waits_a_week_at_most(spec, home, monkeypatch):
    monkeypatch.setattr(update, "_running", lambda *args: True)
    monkeypatch.setattr(update, "WAIT", 0)
    monkeypatch.setattr(update.time, "perf_counter", lambda: 1000.0)  # a clock that reads the same twice
    monkeypatch.setattr(update.time, "sleep", lambda seconds: pytest.fail("waited, though the week was up"))
    update.finish(spec, platform="linux")
    assert kept(home)["result"] == {"version": NEWER, "ok": False, "log": spec["log"]}
    assert "kept running for a week" in (home / "logs" / "update-1.log").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "command, how",
    [
        ([sys.executable, "-c", "raise SystemExit(3)"], "exit code 3"),
        (["no-such-program-of-sift"], "FileNotFoundError"),
    ],
)
def test_an_upgrade_that_fails_is_said(spec, home, monkeypatch, command, how):
    monkeypatch.setattr(update, "_running", lambda *args: False)
    spec["command"] = command
    update.finish(spec, platform="linux")
    assert not kept(home)["result"]["ok"]
    assert how in (home / "logs" / "update-1.log").read_text(encoding="utf-8")


def test_the_process_that_updates_runs_on_its_own(spec, home, monkeypatch):
    # Started as sift starts it: on the base Python, isolated, reading sift's modules from where this one does.
    monkeypatch.setenv("SIFT_HOME", str(home))
    spec["pid"] = NO_PROCESS  # the sift that asked, stopped: not the id of one that ended, which may be given again
    updating = update.start(["finish", json.dumps(spec)])
    assert updating.wait(timeout=60) == 0
    assert kept(home)["result"]["ok"]
    assert (home / "logs" / "update-1.log").read_text(encoding="utf-8").endswith("upgraded\nexit code 0\n")


def test_the_process_that_updates_leaves_the_folder_the_command_was_typed_in(spec, home, tmp_path, monkeypatch):
    # It waits up to a week: on Windows the folder it runs in can't be renamed or deleted meanwhile (WinError 32),
    # and an installer run there reads the settings of whatever project the folder holds.
    typed = tmp_path / "typed"
    typed.mkdir()
    monkeypatch.chdir(typed)
    monkeypatch.setenv("SIFT_HOME", str(home))
    spec["pid"] = NO_PROCESS
    spec["command"] = [sys.executable, "-c", "import os; print(os.getcwd())"]
    updating = update.start(["finish", json.dumps(spec)])
    assert updating.wait(timeout=60) == 0
    ran_in = update.Path((home / "logs" / "update-1.log").read_text(encoding="utf-8").splitlines()[1])
    assert ran_in == update.Path(ran_in.anchor)  # the root of a drive, which nobody renames
