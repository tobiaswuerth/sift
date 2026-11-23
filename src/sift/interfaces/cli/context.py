import argparse
import dataclasses
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol

from sift.api.events import Reporter
from sift.interfaces.cli.shown.outcome import Shown
from sift.interfaces.console.style import Style
from sift.settings import Settings


@dataclass(frozen=True, slots=True)
class Context:
    settings: Settings
    platform: str
    json: bool
    style: Style
    report: Reporter
    clock: Callable[[], float] = time.perf_counter


class Answer(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]


@dataclass(frozen=True, slots=True)
class Command:
    run: Callable[[argparse.Namespace, Context], Shown]


class Commands(Protocol):
    COMMANDS: Mapping[str, Command]
