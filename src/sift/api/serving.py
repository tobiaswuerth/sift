from dataclasses import dataclass
from typing import Annotated

from sift.api.described import Bounds, Help
from sift.api.requests import Request

PORT = 8765
MOST_PORT = 65535


@dataclass(frozen=True, slots=True)
class Served:
    url: str


@dataclass(frozen=True, slots=True)
class ServeHome(Request):
    port: Annotated[
        int, Help(f"the port on 127.0.0.1 to answer on, 0 for any free one (default: {PORT})"), Bounds(0, MOST_PORT)
    ] = PORT
