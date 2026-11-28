from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from sift.api.jobs import ServedJob
from sift.api.wire import Json
from sift.interfaces.mcp.schemas import Schema


@dataclass(frozen=True, slots=True)
class Started:
    number: int
    answer: object | None


class ToolRunning(Protocol):
    def read(self, name: str, data: Json) -> object: ...

    def job(self, name: str, data: Json) -> Started: ...

    def jobs(self) -> tuple[ServedJob, ...]: ...


type Call = Callable[[ToolRunning, Mapping[str, object]], Json]


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    title: str
    description: str
    schema: Schema
    call: Call
    read_only: bool = field(kw_only=True)
