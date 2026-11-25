from dataclasses import dataclass
from typing import Annotated

from sift.api.described import Bounds, Help
from sift.api.requests import Request

PORT = 8765
MOST_PORT = 65535
MOST_IDLE_SECONDS = 604_800
STDIO = (
    "answer MCP on standard input and output, for an agent that launches this command: through the home process, "
    "started if none runs, which then leaves after 15 minutes idle"
)
IDLE_EXIT = "stop after this many seconds with no request and no job (default: never)"


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
