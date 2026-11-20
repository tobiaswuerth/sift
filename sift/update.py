"""Telling of a newer sift, and updating to it on a yes.

At most once a day, an interactive command starts a process of its own (`check`) that asks PyPI for the latest
version and keeps it in SIFT_HOME/update.json; no command waits for the network. The next interactive command (`offer`:
in a terminal, never `serve` or `worker`, nor with SIFT_UPDATE_CHECK=off or offline) asks once per version whether to
update. On yes it stops, and a process of its own (`finish`), on the base Python, waits until no sift of this
installation runs any more, upgrades it as it was installed (uv tool, pipx or pip), logged in
SIFT_HOME/logs/update-DATE.log, and leaves how it went for the next command to say. Waiting matters on Windows, where a
running sift holds its installation's files: uv, replacing them under a running `sift serve`, failed halfway. An
update whose process is gone before it was done, as a restart of the machine ends it, is said and offered again. A
new major version is only told of, with its changelog, as it may need care.

update.json is this module's alone, written whole; deleting it only makes the next command check again."""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import __version__, console, settings

PYPI = "https://pypi.org/pypi/sift-rag/json"
CHANGELOG = "https://github.com/tobiaswuerth/sift/blob/main/CHANGELOG.md"
DAY = 24 * 3600  # seconds between checks
TIMEOUT = 10  # seconds a check waits for PyPI
WAIT = 7 * DAY  # seconds an update waits, at most, for the sifts of its installation to stop
POLL = 1.0  # seconds between looks whether they stopped
RUN = 30 * 60  # seconds the upgrade may take
SHARING = 1.0  # seconds the state's file is tried again while another sift reads or replaces it (see _shared)
QUIET = ("serve", "worker")  # servers, often started by scripts and services: they never ask
# Windows' CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP: a console of its own, hidden, which the programs it runs share,
# and no Ctrl+C. Detached from any console (DETACHED_PROCESS), each program it ran opened a window of its own.
HIDDEN = 0x08000000 | 0x00000200
SYNCHRONIZE, WAIT_TIMEOUT = 0x00100000, 0x00000102  # Windows: the right to wait for a process; "still running"


def offer(command, streams=None):
    """Before a command: say how an update went, offer a newer sift once per version, and start the day's check. Only
    in a terminal, never for QUIET commands, with SIFT_UPDATE_CHECK=off or offline, nor for a sift run from its
    source, which git updates. But any command of an installed sift, wherever it runs, stops while the installer
    replaces sift's files (see _upgrading)."""
    upgrade = upgrade_command()
    if upgrade is None:
        return
    path = state_path()
    state = read(path)
    pending = state.get("pending")
    if pending and _upgrading(pending):
        raise SystemExit(f"sift is being updated to {pending['version']} right now: run this again in a moment")
    streams = streams or (sys.stdin, sys.stdout, sys.stderr)
    if command in QUIET or not settings.current().update_check:
        return
    if not all(stream.isatty() for stream in streams):
        return
    if pending and not _runs(pending["pid"]):
        state = _stopped(path, pending)
    if result := state.get("result"):
        console.say(_outcome(result))
        change(path, result=None)
    if pending := state.get("pending"):
        if time.time() - pending["since"] < WAIT + DAY:
            console.say(f"sift updates to {pending['version']} once no sift of this installation runs any more")
            return
        change(path, pending=None)  # its process is long gone, and another has its id
    if _newer(state.get("latest")) and state.get("asked") != state["latest"]:
        _ask(path, state["latest"], upgrade)
    if time.time() - state.get("checked", 0) >= DAY:
        change(path, checked=time.time())  # first: commands started meanwhile start no check of their own
        start(["check", str(path)])


def _upgrading(pending):
    """Whether the installer replaces sift's files right now: the pending update says so (since when), its process
    runs, and the time an upgrade may take isn't up, after which the process is another's that got its id. A command
    started then ran on files half replaced, or held one the installer was about to replace; and `y` stopped the
    command typed, so the next comes at once."""
    return pending["upgrading"] and time.time() - pending["since"] < RUN and _runs(pending["pid"])


def _stopped(path, pending):
    """The state, once the process of a pending update is gone. It left how the update went as it ended; if it left
    nothing, it was stopped before, as a restart of the machine stops it: said, and the update offered again."""
    state = read(path)
    if state.get("pending") != pending:
        return state
    console.say(
        f"the update to {pending['version']} stopped before it was done: its process is gone, as after a restart"
    )
    change(path, pending=None, asked=None)
    return {key: value for key, value in state.items() if key not in ("pending", "asked")}


def _newer(latest):
    new, now = _version(latest), _version(__version__)
    return bool(new and now and new > now)


def _version(text):
    """A release's version as numbers, (1, 12, 0); None for any other, such as a pre-release."""
    parts = str(text).split(".")
    return tuple(map(int, parts)) if len(parts) == 3 and all(p.isdigit() for p in parts) else None


def _ask(path, latest, upgrade):
    """Offer `latest`, once: a new major version is told of; one in this major updated to on yes, by the upgrade
    command if it can be run, else said."""
    change(path, asked=latest)
    if _version(latest)[0] > _version(__version__)[0]:
        console.say(f"sift {latest} is out, a new major version: read {CHANGELOG} before you update")
    elif not Path(upgrade[0]).is_absolute():  # uv or pipx isn't on the PATH
        console.say(f"sift {latest} is out (this is {__version__}): run `{console.command(*upgrade)}` to update")
    elif _yes(f"sift {latest} is out (this is {__version__}). Update now? [y/N] "):
        _schedule(path, latest, upgrade)


def _yes(question):
    print(question, end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in ("y", "yes")


def _schedule(path, latest, upgrade):
    """Start the process that updates sift once no sift of this installation runs, and stop this one. The update
    waits only once its process runs: one that can't be started leaves none waiting."""
    log = settings.current().home / "logs" / f"update-{datetime.now():%Y-%m-%d_%H%M%S}.log"
    held, since = [str(f) for f in held_files()], time.time()
    spec = {"version": latest, "command": upgrade, "since": since, "pid": os.getpid(), "held": held, "log": str(log)}
    updating = start(["finish", json.dumps(spec)])
    change(path, pending={"version": latest, "since": since, "pid": updating.pid, "upgrading": False})
    console.say(
        f"sift updates to {latest} as soon as no sift of this installation runs any more: this one stops now; stop "
        "any `sift serve` and `sift worker` too. The next command says how it went."
    )
    raise SystemExit(0)


def _outcome(result):
    """What the next command says of an update."""
    version, log = result["version"], result["log"]
    if not result["ok"]:
        return f"sift couldn't update to {version}: see {log}"
    if _version(__version__) < _version(version):
        return f"the update to {version} ran, but this is sift {__version__} still: see {log}"
    return f"sift updated to {version}; restart any `sift serve` and `sift worker`"


def upgrade_command(prefix=None, package=None, which=shutil.which):
    """The command that upgrades this sift as it was installed: by uv as a tool (its environment holds uv's receipt),
    by pipx, or by pip into its environment; its program as found on the PATH, or by name if it isn't. None for sift
    run from its source."""
    prefix, package = Path(prefix or sys.prefix).resolve(), Path(package or __file__).resolve().parent
    if not package.is_relative_to(prefix):
        return None
    if (prefix / "uv-receipt.toml").is_file():
        return [which("uv") or "uv", "tool", "upgrade", "sift-rag"]
    if "pipx" in prefix.parts and prefix.parent.name == "venvs":
        return [which("pipx") or "pipx", "upgrade", "sift-rag"]
    return [sys.executable, "-m", "pip", "install", "--upgrade", "sift-rag"]


def held_files():
    """The files a running sift of this installation holds open on Windows, which an upgrade replaces: its virtual
    environment's Python, and the launcher it was started by, sift.exe, whose name pip's and uv's launchers leave in
    sys.argv[0] without the ".exe"."""
    files = [Path(sys.executable)] if sys.prefix != sys.base_prefix else []
    started = sys.argv[0]
    launcher = Path(started if started.lower().endswith(".exe") else f"{started}.exe")
    return files + ([launcher] if launcher.is_file() else [])


def start(args, platform=sys.platform):
    """Run `main(args)` in a process that outlives this one, with nothing of its terminal: on the base Python, which no
    upgrade replaces, and sift's modules read from where this one reads them (none compiled, which it would hold).
    Returns the process, which no one need wait for."""
    root = str(Path(__file__).resolve().parents[1])
    code = "import sys; sys.path.insert(0, sys.argv[1]); from sift import update; update.main(sys.argv[2:])"
    apart = {"creationflags": HIDDEN} if platform == "win32" else {"start_new_session": True}
    python = getattr(sys, "_base_executable", None) or sys.executable
    devnull = subprocess.DEVNULL
    command = [python, "-I", "-B", "-c", code, root, *args]
    return subprocess.Popen(command, stdin=devnull, stdout=devnull, stderr=devnull, **apart)


def main(args):
    """What a process `start`ed does: "check" PATH, or "finish" SPEC."""
    if args[0] == "check":
        check(Path(args[1]))
    else:
        finish(json.loads(args[1]))


def check(path):
    """Ask PyPI for the latest version and keep it, with the time; when PyPI can't be reached, the time and why, so
    that no command starts another check before tomorrow."""
    import urllib.request  # here: commands that check nothing don't load it

    request = urllib.request.Request(PYPI, headers={"User-Agent": f"sift/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            latest = json.load(response)["info"]["version"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        change(path, checked=time.time(), failed=f"{type(error).__name__}: {error}")
        return
    change(path, checked=time.time(), latest=str(latest), failed=None)


def finish(spec, platform=sys.platform, kernel32=None):
    """Wait, at most WAIT, until the sift that asked has stopped, and on Windows every other of its installation too
    (`held` files free); then run the upgrade, logged, and keep how it went for the next command to say. Before it
    upgrades, the state says so, which stops the commands started from then on (see offer); a sift started just
    before runs, and is waited for too."""
    path, log = state_path(), Path(spec["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    deadline, waited = time.perf_counter() + WAIT, False
    while not waited:
        while _running(spec, platform, kernel32):
            if time.perf_counter() >= deadline:
                return _finished(path, spec, False, "a sift of this installation kept running for a week")
            time.sleep(POLL)
        _say_upgrading(path, spec, True)
        waited = not _running(spec, platform, kernel32)
        if not waited:
            _say_upgrading(path, spec, False)
    with open(log, "w", encoding="utf-8") as out:
        out.write(f"{console.command(*spec['command'])}\n")
        out.flush()
        try:
            code = subprocess.run(
                spec["command"], stdin=subprocess.DEVNULL, stdout=out, stderr=out, timeout=RUN
            ).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            code, how = None, f"{type(error).__name__}: {error}"
        else:
            how = f"exit code {code}"
    _finished(path, spec, code == 0, how)


def _say_upgrading(path, spec, now):
    """Keep in the state whether this process upgrades sift `now`, since this moment, or waits still, as it has since
    the update was agreed to."""
    since = time.time() if now else spec["since"]
    change(path, pending={"version": spec["version"], "since": since, "pid": os.getpid(), "upgrading": now})


def _finished(path, spec, ok, how):
    with open(spec["log"], "a", encoding="utf-8") as out:
        out.write(f"{how}\n")
    change(path, pending=None, result={"version": spec["version"], "ok": ok, "log": spec["log"]})


def _running(spec, platform, kernel32):
    return _runs(spec["pid"], platform, kernel32) or (platform == "win32" and any(map(_in_use, spec["held"])))


def _runs(pid, platform=sys.platform, kernel32=None):
    """Whether a process runs."""
    return _alive_on_windows(pid, kernel32) if platform == "win32" else _alive(pid)


def _alive(pid):
    """Whether a process runs, on POSIX: signal 0 only asks."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it runs, as another user
        pass
    return True


def _alive_on_windows(pid, kernel32=None):
    """Whether a process runs, on Windows, where os.kill would end it."""
    if kernel32 is None:
        import ctypes

        kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def _in_use(file):
    """Whether a program file runs, on Windows: then it can't be opened for writing (nothing is written)."""
    try:
        open(file, "r+b").close()
    except FileNotFoundError:
        return False
    except PermissionError:
        return True
    return False


def state_path():
    return settings.current().home / "update.json"


def _shared(action):
    """action() on the state's file, tried again for SHARING seconds while it is refused: Windows refuses to replace a
    file that another process has open, and to open one being replaced, as two sifts at once do to each other, each
    for well under a millisecond. What is refused for longer is so for another reason, and an error."""
    deadline = time.perf_counter() + SHARING
    while True:
        try:
            return action()
        except PermissionError:
            if time.perf_counter() >= deadline:
                raise
            time.sleep(SHARING / 100)


def read(path):
    """The state kept, of the entries in SHAPES that have their shape; {} if there is none or it was cut short."""
    try:
        state = json.loads(_shared(lambda: path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return {}
    except ValueError:  # written by hand, or a disk that filled: the next write makes it whole
        return {}
    return {k: v for k, v in state.items() if k in SHAPES and _fits(v, SHAPES[k])} if isinstance(state, dict) else {}


NUMBER = (int, float)
SHAPES = {  # what update.json holds, each entry of the type given, or a dict of such
    "checked": NUMBER,  # when PyPI was asked last, in seconds since 1970
    "latest": str,  # its latest version then
    "failed": str,  # why it couldn't be asked, if it couldn't
    "asked": str,  # the latest version offered, once
    # an update waiting, since then, for the sifts of its installation to stop, and the process that waits and then
    # updates; or, once it does, `upgrading` since then
    "pending": {"version": str, "since": NUMBER, "pid": int, "upgrading": bool},
    "result": {"version": str, "ok": bool, "log": str},  # how it went, for the next command to say
}


def _fits(value, shape):
    if isinstance(shape, dict):
        return isinstance(value, dict) and all(isinstance(value.get(k), kind) for k, kind in shape.items())
    return isinstance(value, shape)


def change(path, **changes):
    """Change the state kept, written whole, so that another process reads the old or the new; None removes a key.
    It is written beside the state first (update.json.PID.part), which goes if it can't take the state's place."""
    state = read(path) | changes
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps({k: v for k, v in state.items() if v is not None}), encoding="utf-8")
    try:
        _shared(lambda: os.replace(part, path))
    except BaseException:
        part.unlink(missing_ok=True)
        raise
