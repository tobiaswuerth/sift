import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from sift.api.errors import InvalidRequestError, Problem, StoreUnavailableError
from sift.api.events import Noted
from sift.api.wire import decoded
from sift.settings import ActSetting, Settings, acting_on
from sift.storage.database import private_file, private_folder
from sift.storage.store_files import WRITER_SUFFIX
from sift.storage.writer import Holder, held_by

SERVERS = "servers"
SERVER = "server"
SERVING = "sift serve"
STARTING_SECONDS = 2.0
LOOK_EVERY_SECONDS = 0.05
STARTING = "a home process is starting, or serves another user (process {pid})"
STARTING_REMEDY = "ask again in a moment; a home process serves only the user who started it"
NO_OWN_FOLDER = "no folder of the user's own is known, to keep the home process's token in"
NO_OWN_FOLDER_REMEDY = "set LOCALAPPDATA on Windows, XDG_STATE_HOME elsewhere, to an absolute folder"
KEY_CHARACTERS = 32
UNSHARED = "the home process uses {settings}"
UNSHARED_REMEDY = "stop sift serve to use yours, or start it again with them"


@dataclass(frozen=True, slots=True)
class Endpoint:
    port: int
    pid: int
    token: str
    version: str
    acting_on: tuple[ActSetting, ...] = ()


def lock_of(home: Path) -> Path:
    return home / f"{SERVER}{WRITER_SUFFIX}"


def unshared(endpoint: Endpoint, settings: Settings) -> tuple[Noted, ...]:
    ours = acting_on(settings)
    theirs = [each for each in endpoint.acting_on if each not in ours]
    said = ", ".join(each.variable + (" unset" if each.value is None else f"={each.value}") for each in theirs)
    return (Noted(UNSHARED.format(settings=said), UNSHARED_REMEDY),) if theirs else ()


def endpoint_of(state: Path, home: Path) -> Path:
    key = hashlib.sha256(os.fsencode(os.path.normcase(home.resolve()))).hexdigest()[:KEY_CHARACTERS]
    return state / SERVERS / f"{key}.json"


def own_endpoint(settings: Settings) -> Path:
    if settings.state is None:
        raise InvalidRequestError(NO_OWN_FOLDER, NO_OWN_FOLDER_REMEDY)
    return endpoint_of(settings.state, settings.home)


def written(path: Path, endpoint: Endpoint) -> None:
    private_folder(path.parent)
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    part.unlink(missing_ok=True)
    private_file(part)
    part.write_bytes(json.dumps(asdict(endpoint)).encode())
    part.replace(path)


def _written_by(path: Path | None, holder: Holder) -> Endpoint | None:
    if path is None:
        return None
    try:
        read = cast("object", json.loads(path.read_bytes()))
    except FileNotFoundError:
        return None
    endpoint = decoded(Endpoint, read)
    return endpoint if endpoint.pid == holder.pid else None


def found(settings: Settings, patience: float = STARTING_SECONDS) -> Endpoint | None:
    listed = None if settings.state is None else endpoint_of(settings.state, settings.home)
    looked, looks = threading.Event(), max(1, round(patience / LOOK_EVERY_SECONDS))
    while (holder := held_by(lock_of(settings.home))) is not None:
        if (endpoint := _written_by(listed, holder)) is not None:
            return endpoint
        looks -= 1
        if not looks:
            raise StoreUnavailableError(STARTING.format(pid=holder.pid), STARTING_REMEDY, code=Problem.BUSY)
        looked.wait(LOOK_EVERY_SECONDS)
    return None
