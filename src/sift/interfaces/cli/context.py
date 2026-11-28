import argparse
import dataclasses
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from sift.api.events import Reporter
from sift.interfaces.cli.shown.outcome import Shown
from sift.interfaces.console.style import Style
from sift.settings import Settings

if TYPE_CHECKING:
    from sift.interfaces.server.endpoint import Endpoint


@dataclass(frozen=True, slots=True)
class Context:
    settings: Settings
    platform: str
    json: bool
    style: Style
    report: Reporter
    browse: Callable[[str], None]
    clock: Callable[[], float] = time.perf_counter
    server: "Endpoint | None" = None


class Answer(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]


@dataclass(frozen=True, slots=True)
class Command:
    run: Callable[[argparse.Namespace, Context], Shown]
    writes: bool = False


class Commands(Protocol):
    COMMANDS: Mapping[str, Command]
