from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class Phase(StrEnum):
    REMAKE = "remake"
    SCAN = "scan"
    READ = "read"
    MERGE = "merge"
    EMBED = "embed"
    LOAD = "load"


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
class PhaseDone:
    count: int


@dataclass(frozen=True, slots=True)
class Noted:
    text: str
    remedy: str | None = None


@dataclass(frozen=True, slots=True)
class Finished:
    added: int
    changed: int
    removed: int
    unreadable: int


type Event = CollectionStarted | PhaseStarted | Advanced | PhaseDone | Noted | Finished
type Reporter = Callable[[Event], None]
