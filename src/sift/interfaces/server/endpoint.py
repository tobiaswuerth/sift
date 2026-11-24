import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.wire import decoded
from sift.storage.database import PRIVATE_FILE
from sift.storage.store_files import WRITER_SUFFIX
from sift.storage.writer import Holder, held_by

ENDPOINT = "server.json"
SERVER = "server"
SERVING = "sift serve"
STARTING_SECONDS = 2.0
LOOK_EVERY_SECONDS = 0.05
STARTING_REMEDY = "ask again in a moment"


@dataclass(frozen=True, slots=True)
class Endpoint:
    port: int
    pid: int
    token: str
    version: str


def lock_of(home: Path) -> Path:
    return home / f"{SERVER}{WRITER_SUFFIX}"


def written(home: Path, endpoint: Endpoint) -> None:
    path = home / ENDPOINT
    part = path.with_name(f"{ENDPOINT}.{os.getpid()}.part")
    part.unlink(missing_ok=True)
    os.close(os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY, PRIVATE_FILE))
    part.write_bytes(json.dumps(asdict(endpoint)).encode())
    part.replace(path)


def _written_by(home: Path, holder: Holder) -> Endpoint | None:
    try:
        read = cast("object", json.loads((home / ENDPOINT).read_bytes()))
    except FileNotFoundError:
        return None
    endpoint = decoded(Endpoint, read)
    return endpoint if endpoint.pid == holder.pid else None


def found(home: Path, patience: float = STARTING_SECONDS) -> Endpoint | None:
    looked, looks = threading.Event(), max(1, round(patience / LOOK_EVERY_SECONDS))
    while (holder := held_by(lock_of(home))) is not None:
        if (endpoint := _written_by(home, holder)) is not None:
            return endpoint
        looks -= 1
        if not looks:
            said = f"a home process is starting (process {holder.pid})"
            raise StoreUnavailableError(said, STARTING_REMEDY, code=Problem.BUSY)
        looked.wait(LOOK_EVERY_SECONDS)
    return None
