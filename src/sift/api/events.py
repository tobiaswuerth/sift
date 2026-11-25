from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from sift.api.errors import Problem


class Phase(StrEnum):
    REMAKE = "remake"
    SCAN = "scan"
    READ = "read"
    MERGE = "merge"
    EMBED = "embed"
    LOAD = "load"
    ORDER = "order"
    COPY = "copy"
    TITLE = "title"
    MEASURE = "measure"
    CARRY = "carry"


@dataclass(frozen=True, slots=True)
class CollectionStarted:
    name: str
    folder: str


@dataclass(frozen=True, slots=True)
class PhaseStarted:
    phase: Phase
    total: int | None


@dataclass(frozen=True, slots=True)
class Advanced:
    done: int


@dataclass(frozen=True, slots=True)
class FilesKept:
    files: int


@dataclass(frozen=True, slots=True)
class PhaseDone:
    count: int


@dataclass(frozen=True, slots=True)
class Noted:
    text: str
    remedy: str | None = None
    code: Problem | None = None


@dataclass(frozen=True, slots=True)
class Finished:
    added: int
    changed: int
    removed: int
    unreadable: int


@dataclass(frozen=True, slots=True)
class Listening:
    url: str
    home: str


@dataclass(frozen=True, slots=True)
class Lending:
    machine: str
    port: int
    key: str


type Event = (
    CollectionStarted | PhaseStarted | Advanced | FilesKept | PhaseDone | Noted | Finished | Listening | Lending
)
type Reporter = Callable[[Event], None]
type EventKind = type[
    CollectionStarted | PhaseStarted | Advanced | FilesKept | PhaseDone | Noted | Finished | Listening | Lending
]

EVENT_KINDS: Mapping[str, EventKind] = MappingProxyType(
    {
        "CollectionStarted": CollectionStarted,
        "PhaseStarted": PhaseStarted,
        "Advanced": Advanced,
        "FilesKept": FilesKept,
        "PhaseDone": PhaseDone,
        "Noted": Noted,
        "Finished": Finished,
        "Listening": Listening,
        "Lending": Lending,
    }
)
EVENT_TAGS: Mapping[EventKind, str] = MappingProxyType({kind: tag for tag, kind in EVENT_KINDS.items()})
