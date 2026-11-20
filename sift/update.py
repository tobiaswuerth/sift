"""Telling of a newer sift, and updating to it on a yes.

At most once a day, an interactive command starts a process of its own (`check`) that asks PyPI for the latest
version and keeps it in SIFT_HOME/update.json; no command waits for the network. The next interactive command (`offer`:
in a terminal, never `serve` or `worker`, nor with SIFT_UPDATE_CHECK=off or offline) asks once per version whether to
update. On yes it stops, and a process of its own (`finish`), on the base Python, waits until no sift of this
installation runs any more, upgrades it as it was installed (uv tool, pipx or pip), logged in
SIFT_HOME/logs/update-DATE.log, and leaves how it went for the next command to say. Waiting matters on Windows, where a
running sift holds its installation's files: uv, replacing them under a running `sift serve`, failed halfway. While
the installer runs, update.json says so, and any command of the installation that starts then stops with one line. An
update whose process is gone before it was done, as a restart of the machine ends it, is said and offered again. A
new major version is only told of, with its changelog, as it may need care; so is a newer sift that sift can't
install itself, with the command: in an environment without pip, or an installation the user may not write.

update.json is this module's alone, written whole; deleting it only makes the next command check again."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    NoReturn,
    TextIO,
    TypeAlias,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    get_type_hints,
    is_typeddict,
)

from . import __version__, console, settings

if TYPE_CHECKING:
    import ctypes

PYPI = "https://pypi.org/pypi/sift-rag/json"
CHANGELOG = "https://github.com/tobiaswuerth/sift/blob/main/CHANGELOG.md"
DAY = 24 * 3600  # seconds between checks
TIMEOUT = 10  # seconds a check waits for PyPI
WAIT = 7 * DAY  # seconds an update waits, at most, for the sifts of its installation to stop
POLL = 1.0  # seconds between looks whether they stopped
RUN = 3 * 3600  # seconds the upgrade may take: stopped, it leaves sift half upgraded, and a slow line takes long over
# a newer PyTorch (2.5 GB: half an hour at 10 Mbit/s)
SHARING = 1.0  # seconds the state's file is tried again while another sift reads or replaces it (see _shared)
QUIET = ("serve", "worker")  # servers, often started by scripts and services: they never ask
# Windows' CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP: a console of its own, hidden, which the programs it runs share,
# and no Ctrl+C. Detached from any console (DETACHED_PROCESS), each program it ran opened a window of its own.
HIDDEN = 0x08000000 | 0x00000200
SYNCHRONIZE, WAIT_TIMEOUT = 0x00100000, 0x00000102  # Windows: the right to wait for a process; "still running"
Json: TypeAlias = dict[str, "Json"] | list["Json"] | str | float | bool | None  # what a JSON file may hold
Done = TypeVar("Done")


class Pending(TypedDict):
    """An update waiting, since then, for the sifts of its installation to stop, and the process that waits and then
    updates; or, once it does, `upgrading` since then."""

    version: str
    since: float
    pid: int
    upgrading: bool


class Result(TypedDict):
    """How an update went, for the next command to say."""

    version: str
    ok: bool
    log: str


class State(TypedDict, total=False):
    """What update.json holds: of these entries, those that are of their types (see `read`)."""

    checked: float  # when PyPI was asked last, in seconds since 1970
    latest: str  # its latest version then
    failed: str  # why it couldn't be asked, if it couldn't
    asked: str  # the latest version offered, once
    pending: Pending
    result: Result


class Changes(TypedDict, total=False):
    """Entries of the state to `change`: as State has them, or None for one to remove."""

    checked: float | None
    latest: str | None
    failed: str | None
    asked: str | None
    pending: Pending | None
    result: Result | None


class Spec(TypedDict):
    """An update agreed to, as the process that makes it is told (see `finish`): the command that upgrades, the sift
    that asked (`pid`, `since`), and the files a running sift of the installation holds."""

    version: str
    command: list[str]
    since: float
    pid: int
    held: list[str]
    log: str


ENTRIES = get_type_hints(State)


def offer(command: str, streams: Sequence[TextIO] | None = None) -> None:
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
    if not all(stream is not None and stream.isatty() for stream in streams):  # None: pythonw has no streams
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
    latest = state.get("latest")
    if latest and _newer(latest) and state.get("asked") != latest:
        _ask(path, latest, upgrade)
    if time.time() - state.get("checked", 0) >= DAY:
        change(path, checked=time.time())  # first: commands started meanwhile start no check of their own
        start(["check", str(path)])


def _upgrading(pending: Pending) -> bool:
    """Whether the installer replaces sift's files right now: the pending update says so (since when), its process
    runs, and the time an upgrade may take isn't up, after which the process is another's that got its id. A command
    started then ran on files half replaced, or held one the installer was about to replace; and `y` stopped the
    command typed, so the next comes at once."""
    return pending["upgrading"] and time.time() - pending["since"] < RUN and _runs(pending["pid"])


def _stopped(path: Path, pending: Pending) -> State:
    """The state, once the process of a pending update is gone. It left how the update went as it ended; if it left
    nothing, it was stopped before, as a restart of the machine stops it: said, and the update offered again."""
    state = read(path)
    if state.get("pending") != pending:
        return state
    console.say(
        f"the update to {pending['version']} stopped before it was done: its process is gone, as after a restart"
    )
    change(path, pending=None, asked=None)
    return read(path)


def _newer(latest: str, major: bool = False) -> bool:
    """Whether a version is a release newer than this sift, with `major` of a later major version; never one that
    isn't a release's, nor for a sift that isn't one."""
    new, now = _version(latest), _version(__version__)
    return bool(new and now and (new[:1] > now[:1] if major else new > now))


def _version(text: str) -> tuple[int, ...] | None:
    """A release's version as numbers, (1, 12, 0); None for any other, such as a pre-release."""
    parts = text.split(".")
    return tuple(map(int, parts)) if len(parts) == 3 and all(p.isdigit() for p in parts) else None


def _ask(path: Path, latest: str, upgrade: list[str]) -> None:
    """Offer `latest`, once: a new major version is told of; one in this major updated to on yes, by the upgrade
    command if it can be run, else said, if there is one."""
    change(path, asked=latest)
    if _newer(latest, major=True):
        console.say(f"sift {latest} is out, a new major version: read {CHANGELOG} before you update")
    elif not upgrade:
        console.say(f"sift {latest} is out (this is {__version__}): update it with the installer it came by")
    elif not Path(upgrade[0]).is_absolute():  # uv or pipx isn't on the PATH, or the command is for the user to run
        console.say(f"sift {latest} is out (this is {__version__}): run `{console.command(*upgrade)}` to update")
    elif not _writable(sys.prefix):  # an administrator's: its files look held (see _in_use), and the wait never ends
        console.say(
            f"sift {latest} is out (this is {__version__}), but its installation isn't yours to write: update it as "
            f"who installed it, with `{console.command(*upgrade)}`"
        )
    elif _yes(f"sift {latest} is out (this is {__version__}). Update now? [y/N] "):
        _schedule(path, latest, upgrade)


def _writable(folder: str) -> bool:
    """Whether this user may write in a folder, as tried with a file made there and gone at once: what the system
    tells without trying leaves out who may write (on Windows, os.access looks at the read-only mark alone)."""
    try:
        with tempfile.TemporaryFile(dir=folder):
            return True
    except OSError:
        return False


def _yes(question: str) -> bool:
    print(question, end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in ("y", "yes")


def _schedule(path: Path, latest: str, upgrade: list[str]) -> NoReturn:
    """Start the process that updates sift once no sift of this installation runs, and stop this one. The update
    waits only once its process runs: one that can't be started leaves none waiting."""
    log = settings.current().home / "logs" / f"update-{datetime.now():%Y-%m-%d_%H%M%S}.log"
    held, since = [str(f) for f in held_files()], time.time()
    spec: Spec = {
        "version": latest,
        "command": upgrade,
        "since": since,
        "pid": os.getpid(),
        "held": held,
        "log": str(log),
    }
    updating = start(["finish", json.dumps(spec)])
    change(path, pending={"version": latest, "since": since, "pid": updating.pid, "upgrading": False})
    console.say(
        f"sift updates to {latest} as soon as no sift of this installation runs any more: this one stops now; stop "
        "any `sift serve` and `sift worker` too. The next command says how it went."
    )
    raise SystemExit(0)


def _outcome(result: Result) -> str:
    """What the next command says of an update."""
    version, log = result["version"], result["log"]
    if not result["ok"]:
        return f"sift couldn't update to {version}: see {log}"
    if _newer(version):
        return f"the update to {version} ran, but this is sift {__version__} still: see {log}"
    return f"sift updated to {version}; restart any `sift serve` and `sift worker`"


def upgrade_command(
    prefix: str | Path | None = None,
    package: str | Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    pip: bool | None = None,
) -> list[str] | None:
    """The command that upgrades this sift as it was installed: by uv as a tool (its environment holds uv's receipt),
    by pipx, or by pip into its environment; its program as found on the PATH, or by name if it isn't, and then only
    said. An environment without pip (`pip`: looked up, if not given) is uv's as a rule, and its command only said, by
    name: `uv pip` wants the environment chosen, and one `uvx` made, in uv's cache, is never upgraded, another is made
    for a newer sift. [] if nothing tells how it was installed: a newer sift is told of, no more. None for sift run
    from its source."""
    prefix, package = Path(prefix or sys.prefix).resolve(), Path(package or __file__).resolve().parent
    if not package.is_relative_to(prefix):
        return None
    if (prefix / "uv-receipt.toml").is_file():
        return [which("uv") or "uv", "tool", "upgrade", "sift-rag"]
    if "pipx" in prefix.parts and prefix.parent.name == "venvs":
        return [which("pipx") or "pipx", "upgrade", "sift-rag"]
    if pip is None:
        pip = importlib.util.find_spec("pip") is not None
    if pip:
        return [sys.executable, "-m", "pip", "install", "--upgrade", "sift-rag"]
    if not _made_by_uv(prefix):
        return []
    if prefix.parent.name.startswith("archive-v"):  # uv's cache
        return ["uvx", "sift-rag@latest"]
    return ["uv", "pip", "install", "--upgrade", "sift-rag"]


def _made_by_uv(prefix: Path) -> bool:
    """Whether uv made the virtual environment at `prefix`, as its pyvenv.cfg says ("uv = 0.13.9")."""
    try:
        lines = (prefix / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return False
    return any(line.partition("=")[0].strip() == "uv" for line in lines)


def held_files() -> list[Path]:
    """The files a running sift of this installation holds open on Windows, which an upgrade replaces: its virtual
    environment's Python, and the launcher it was started by, sift.exe, whose name pip's and uv's launchers leave in
    sys.argv[0] without the ".exe"."""
    files = [Path(sys.executable)] if sys.prefix != sys.base_prefix else []
    started = sys.argv[0]
    launcher = Path(started if started.lower().endswith(".exe") else f"{started}.exe")
    return files + ([launcher] if launcher.is_file() else [])


def start(args: list[str], platform: str = sys.platform) -> subprocess.Popen[bytes]:
    """Run `main(args)` in a process that outlives this one, with nothing of its terminal: on the base Python, which no
    upgrade replaces, and sift's modules read from where this one reads them (none compiled, which it would hold).
    It runs at the root of that Python's drive, not in the folder the command was typed in: it may wait a week, on
    Windows nobody could rename or delete that folder meanwhile, and an installer reads the settings of the project
    it runs in. Returns the process, which no one need wait for."""
    root = str(Path(__file__).resolve().parents[1])
    code = "import sys; sys.path.insert(0, sys.argv[1]); from sift import update; update.main(sys.argv[2:])"
    python = getattr(sys, "_base_executable", None) or sys.executable
    devnull = subprocess.DEVNULL
    command = [python, "-I", "-B", "-c", code, root, *args]
    folder = Path(python).anchor
    apart = partial(subprocess.Popen, command, stdin=devnull, stdout=devnull, stderr=devnull, cwd=folder)
    return apart(creationflags=HIDDEN) if platform == "win32" else apart(start_new_session=True)


def main(args: list[str]) -> None:
    """What a process `start`ed does: "check" PATH, or "finish" SPEC."""
    if args[0] == "check":
        check(Path(args[1]))
    else:
        finish(json.loads(args[1]))


def check(path: Path) -> None:
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


def finish(spec: Spec, platform: str = sys.platform, kernel32: "ctypes.CDLL | None" = None) -> None:
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


def _say_upgrading(path: Path, spec: Spec, now: bool) -> None:
    """Keep in the state whether this process upgrades sift `now`, since this moment, or waits still, as it has since
    the update was agreed to."""
    since = time.time() if now else spec["since"]
    change(path, pending={"version": spec["version"], "since": since, "pid": os.getpid(), "upgrading": now})


def _finished(path: Path, spec: Spec, ok: bool, how: str) -> None:
    with open(spec["log"], "a", encoding="utf-8") as out:
        out.write(f"{how}\n")
    change(path, pending=None, result={"version": spec["version"], "ok": ok, "log": spec["log"]})


def _running(spec: Spec, platform: str, kernel32: "ctypes.CDLL | None") -> bool:
    return _runs(spec["pid"], platform, kernel32) or (platform == "win32" and any(map(_in_use, spec["held"])))


def _runs(pid: int, platform: str = sys.platform, kernel32: "ctypes.CDLL | None" = None) -> bool:
    """Whether a process runs."""
    return _alive_on_windows(pid, kernel32) if platform == "win32" else _alive(pid)


def _alive(pid: int) -> bool:
    """Whether a process runs, on POSIX: signal 0 only asks."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it runs, as another user
        pass
    return True


def _alive_on_windows(pid: int, kernel32: "ctypes.CDLL | None" = None) -> bool:
    """Whether a process runs, on Windows, where os.kill would end it."""
    if kernel32 is None:
        import ctypes

        # cdll, which ctypes has on every system, so that the types are checked on each; on 64-bit Windows, the only
        # one sift's packages install on, it calls as windll does.
        kernel32 = ctypes.cdll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def _in_use(file: str) -> bool:
    """Whether a program file runs, on Windows: then it can't be opened for writing (nothing is written). Nor can a
    file this user may not write, which Python can't tell from it; such an installation is never updated (see _ask)."""
    try:
        open(file, "r+b").close()
    except FileNotFoundError:
        return False
    except PermissionError:
        return True
    return False


def state_path() -> Path:
    return settings.current().home / "update.json"


def _shared(action: Callable[[], Done]) -> Done:
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


def read(path: Path) -> State:
    """The state kept, of the entries State names that are of their types; {} if there is none or it was cut short."""
    try:
        state: Json = json.loads(_shared(lambda: path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return {}
    except ValueError:  # written by hand, or a disk that filled: the next write makes it whole
        return {}
    kept = {k: v for k, v in state.items() if k in ENTRIES and _fits(v, ENTRIES[k])} if isinstance(state, dict) else {}
    return cast(State, kept)  # as _fits has just made sure


def _fits(value: Json, kind: type) -> bool:
    """Whether a value read is of the type its entry is to be: a dict of entries, of each of theirs; a float, any
    number, as JSON writes a whole one."""
    if is_typeddict(kind):
        return isinstance(value, dict) and all(_fits(value.get(k), of) for k, of in get_type_hints(kind).items())
    return isinstance(value, (int, float) if kind is float else kind)


def change(path: Path, **changes: Unpack[Changes]) -> None:
    """Change the state kept, written whole, so that another process reads the old or the new; None removes a key.
    It is written beside the state first (update.json.PID.part), which goes if it can't take the state's place."""
    state = {**read(path), **changes}
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps({k: v for k, v in state.items() if v is not None}), encoding="utf-8")
    try:
        _shared(lambda: os.replace(part, path))
    except BaseException:
        part.unlink(missing_ok=True)
        raise
