from dataclasses import dataclass
from typing import Annotated
from urllib.parse import SplitResult, urlsplit

from sift.api.described import Bounds, Help, Length
from sift.api.errors import InvalidRequestError
from sift.api.requests import Request

PORT = 8765
WORKER_PORT = 8766
MOST_PORT = 65535
MOST_IDLE_SECONDS = 604_800
STDIO = (
    "answer MCP on standard input and output, for an agent that launches this command: through the home process, "
    "started if none runs, which then leaves after 15 minutes idle"
)
IDLE_EXIT = "stop after this many seconds with no request and no job (default: never)"
ALL_INTERFACES = "0.0.0.0"
HOST_LENGTH = Length(1, 253)
LENT_PORT = "the port to answer other machines on, 0 for any free one"
LENT_HOST = "the address of this machine to answer on, 0.0.0.0 for every one it has, 127.0.0.1 for itself alone"
EMBED_REMOTE_VARIABLE = "SIFT_EMBED_REMOTE"
NOT_AN_ADDRESS = f"{EMBED_REMOTE_VARIABLE} is no address a worker prints"
ADDRESS_REMEDY = (
    f"set {EMBED_REMOTE_VARIABLE} to the https://<key>@<machine>:<port> address the worker prints, or unset it"
)
SECURE = "https"
ROOT_PATHS = frozenset({"", "/"})


@dataclass(frozen=True, slots=True)
class Served:
    url: str


@dataclass(frozen=True, slots=True)
class OpenUi(Request):
    no_browser: Annotated[bool, Help("say the page's address alone, without opening it in the browser")] = False


@dataclass(frozen=True, slots=True)
class UiOpened:
    url: str
    served: bool


@dataclass(frozen=True, slots=True)
class ServeHome(Request):
    port: Annotated[
        int, Help("the port on 127.0.0.1 to answer on, 0 for any free one"), Bounds(0, MOST_PORT, grouped=False)
    ] = PORT
    stdio: Annotated[bool, Help(STDIO)] = False
    idle_exit: Annotated[int | None, Help(IDLE_EXIT), Bounds(1, MOST_IDLE_SECONDS)] = None


@dataclass(frozen=True, slots=True)
class ServeWorker(Request):
    port: Annotated[int, Help(LENT_PORT), Bounds(0, MOST_PORT, grouped=False)] = WORKER_PORT
    host: Annotated[str, Help(LENT_HOST), HOST_LENGTH] = ALL_INTERFACES


@dataclass(frozen=True, slots=True)
class WorkerAddress:
    host: str
    port: int
    token: str
    pin: str

    @property
    def address(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{SECURE}://{host}:{self.port}"


def _port(parts: SplitResult) -> int | None:
    try:
        found = parts.port
    except ValueError:
        return None
    return WORKER_PORT if found is None else found


def _bare(parts: SplitResult) -> bool:
    return parts.password is None and parts.path in ROOT_PATHS and not parts.query and not parts.fragment


def worker_address(written: str) -> WorkerAddress:
    try:
        parts = urlsplit(written)
    except ValueError:
        raise InvalidRequestError(NOT_AN_ADDRESS, ADDRESS_REMEDY, EMBED_REMOTE_VARIABLE) from None
    token, _, pin = (parts.username or "").partition(".")
    port = _port(parts)
    if parts.scheme != SECURE or not (parts.hostname and token and pin and port and _bare(parts)):
        raise InvalidRequestError(NOT_AN_ADDRESS, ADDRESS_REMEDY, EMBED_REMOTE_VARIABLE)
    return WorkerAddress(parts.hostname, port, token, pin)
