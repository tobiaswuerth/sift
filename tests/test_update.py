"""`update`: telling of a newer sift, and updating to it on a yes. PyPI, the terminal, the processes started and the
waits are fakes, but for one real run of the process that updates."""

import io
import json
import subprocess
import sys
import time
import urllib.error
from types import SimpleNamespace

import pytest

from sift import __version__, cli, console, update

MAJOR, MINOR, PATCH = update._version(__version__)
NEWER = f"{MAJOR}.{MINOR}.{PATCH + 1}"
NEXT_MAJOR = f"{MAJOR + 1}.0.0"
NO_PROCESS = 4_194_305  # above Linux's highest process id and macOS's, and odd, which none on Windows is


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
    monkeypatch.setattr(update, "start", started.append)
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
    assert kept(home)["pending"]["version"] == NEWER


def test_while_an_update_waits_it_is_said(installed, home, capsys):
    keep(home, latest=NEWER, pending={"version": NEWER, "since": time.time()}, checked=0)
    update.offer("status", TERMINAL)
    assert f"sift updates to {NEWER} once no sift of this installation runs" in capsys.readouterr().err
    assert installed == []  # nor checks meanwhile


def test_an_update_waiting_far_too_long_is_forgotten(installed, home, monkeypatch, capsys):
    keep(home, latest=NEWER, checked=time.time(), pending={"version": NEWER, "since": time.time() - 9 * update.DAY})
    answer(monkeypatch, "\n")  # its process is long gone: offered again
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
    monkeypatch.setattr(sys, "argv", [str(launcher), "status"])
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "python"))
    assert update.held_files() == [update.Path(sys.executable), launcher]
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)  # no virtual environment
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "sift"), "status"])
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
    assert kwargs == {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} | apart


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
    looks = iter([None, None, ProcessLookupError()])

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
    assert len(opened) == 3  # in use, in use, free
    assert kernel.closed == kernel.opened == 4  # one look while it ran, three after
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
