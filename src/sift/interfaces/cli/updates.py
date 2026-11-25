import http.client
import importlib.util
import json
import re
import sys
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

PYPI = "https://pypi.org/pypi/sift-rag/json"
STATE = "update.json"
DAY = 24 * 3600.0
TIMEOUT = 3.0
DEADLINE = 5.0
MOST_BYTES = 32 << 20
PACKAGE = "sift-rag"
VERSION = re.compile(r"([0-9]+(?:\.[0-9]+)*)(?:(a|b|rc)([0-9]+))?")
PHASES = MappingProxyType({"a": 0, "b": 1, "rc": 2})
FINAL = (3, 0)
PLACES = 6
CONNECTIONS = MappingProxyType({"https": http.client.HTTPSConnection, "http": http.client.HTTPConnection})
NO_VERSION = "PyPI named no version"


@dataclass(frozen=True, slots=True)
class Known:
    checked: float = 0.0
    latest: str | None = None
    failed: str | None = None


def _ordered(version: str) -> tuple[tuple[int, ...], tuple[int, int]] | None:
    found = VERSION.fullmatch(version)
    if found is None:
        return None
    numbers = tuple(int(part) for part in found.group(1).split("."))
    phase = FINAL if found.group(2) is None else (PHASES[found.group(2)], int(found.group(3)))
    return numbers + (0,) * (PLACES - len(numbers)), phase


def newer(latest: str, installed: str) -> bool:
    new, now = _ordered(latest), _ordered(installed)
    return new is not None and now is not None and new > now


def from_source(prefix: Path, package: Path) -> bool:
    return not package.resolve().is_relative_to(prefix.resolve())


def _made_by_uv(prefix: Path) -> bool:
    try:
        lines = (prefix / "pyvenv.cfg").read_bytes().decode("utf-8", errors="replace").split("\n")
    except OSError:
        return False
    return any(line.partition("=")[0].strip() == "uv" for line in lines)


def upgrade_command(prefix: Path, *, pip: bool, python: str) -> str | None:
    if (prefix / "uv-receipt.toml").is_file():
        return f"uv tool upgrade {PACKAGE}"
    if "pipx" in prefix.parts and prefix.parent.name == "venvs":
        return f"pipx upgrade {PACKAGE}"
    if pip:
        return f"{python} -m pip install --upgrade {PACKAGE}"
    if not _made_by_uv(prefix):
        return None
    if prefix.parent.name.startswith("archive-v"):
        return f"uvx {PACKAGE}@latest"
    return f"uv pip install --upgrade {PACKAGE}"


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def known(path: Path) -> Known:
    try:
        read = cast("object", json.loads(path.read_bytes()))
    except (OSError, ValueError):
        return Known()
    if not isinstance(read, dict):
        return Known()
    held = cast("dict[str, object]", read)
    return Known(_number(held.get("checked")), _text(held.get("latest")), _text(held.get("failed")))


def _written(path: Path, state: Known) -> None:
    text = json.dumps({"checked": state.checked, "latest": state.latest, "failed": state.failed})
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f"{path.name}.", delete=False) as part:
        part.write(text.encode())
    try:
        Path(part.name).replace(path)
    except OSError:
        Path(part.name).unlink()
        raise


def _latest(body: bytes) -> str:
    read = cast("object", json.loads(body))
    info = cast("dict[str, object]", read).get("info") if isinstance(read, dict) else None
    version = cast("dict[str, object]", info).get("version") if isinstance(info, dict) else None
    if not isinstance(version, str):
        raise TypeError(NO_VERSION)
    return version


def fetched(url: str, installed: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in CONNECTIONS or not parts.hostname:
        raise ValueError(f"{url} is no web address")
    connection = CONNECTIONS[parts.scheme](parts.hostname, parts.port, timeout=TIMEOUT)
    try:
        connection.request("GET", parts.path or "/", headers={"User-Agent": f"sift/{installed}"})
        response = connection.getresponse()
        body = response.read(MOST_BYTES + 1)
    finally:
        connection.close()
    if response.status != http.client.OK or len(body) > MOST_BYTES:
        raise ValueError(f"PyPI answered {response.status} {response.reason}, {len(body):,} bytes")
    return _latest(body)


def _has_pip() -> bool:
    return importlib.util.find_spec("pip") is not None


@dataclass(frozen=True, slots=True)
class Checking:
    now: Callable[[], float] = time.time
    fetch: Callable[[str, str], str] = fetched
    url: str = PYPI
    prefix: Path = Path(sys.prefix)
    package: Path = Path(__file__)
    pip: Callable[[], bool] = _has_pip
    python: str = sys.executable
    deadline: float = DEADLINE


CHECKING = Checking()


@dataclass(slots=True)
class PypiAnswer:
    latest: str | None = None
    failed: str | None = None


def _ask(answered: PypiAnswer, checking: Checking, installed: str) -> None:
    try:
        answered.latest = checking.fetch(checking.url, installed)
    except (OSError, ValueError, TypeError) as error:
        answered.failed = f"{type(error).__name__}: {error}"


def _asked(checking: Checking, installed: str, before: Known) -> Known:
    answered = PypiAnswer()
    asking = threading.Thread(target=_ask, args=(answered, checking, installed), daemon=True)
    asking.start()
    asking.join(checking.deadline)
    if answered.latest is not None:
        return Known(checking.now(), answered.latest, None)
    failed = answered.failed or f"TimeoutError: PyPI did not answer within {checking.deadline:g} s"
    return Known(checking.now(), before.latest, failed)


def told(home: Path, installed: str, asked: Checking | None = None) -> str | None:
    checking = CHECKING if asked is None else asked
    if not home.is_dir() or from_source(checking.prefix, checking.package):
        return None
    path = home / STATE
    before = known(path)
    if checking.now() - before.checked < DAY:
        return None
    state = _asked(checking, installed, before)
    try:
        _written(path, state)
    except OSError as error:
        return f"what PyPI said could not be kept in {path} ({error}); the next command asks again"
    if state.latest is None or not newer(state.latest, installed):
        return None
    command = upgrade_command(checking.prefix, pip=checking.pip(), python=checking.python)
    return f"sift {state.latest} is out, this is {installed}" + (f": {command}" if command else "")
