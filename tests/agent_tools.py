from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from sift.api.jobs import ServedJob
from sift.api.wire import Json
from sift.interfaces.acts.acting import Acting
from sift.interfaces.mcp.offered import TOOLS
from sift.interfaces.mcp.tools import Started, Tool
from sift.interfaces.server.handled import HANDLED, for_agents

type Object = dict[str, Json]


@dataclass(slots=True)
class Direct:
    acting: Acting
    asked: list[str] = field(default_factory=list[str])
    sent: list[Json] = field(default_factory=list[Json])
    finished: bool = True

    def read(self, name: str, data: Json) -> object:
        self.asked.append(name)
        self.sent.append(data)
        return for_agents(HANDLED)[name].run(self.acting, data)

    def job(self, name: str, data: Json) -> Started:
        self.asked.append(name)
        return Started(4, for_agents(HANDLED)[name].run(self.acting, data) if self.finished else None)

    def jobs(self) -> tuple[ServedJob, ...]:
        return (ServedJob(4, "reindex", running=True, seconds=2.5, watched=False, collection="open", done=1),)


def tool(name: str) -> Tool:
    return next(each for each in TOOLS if each.name == name)


def called(direct: Direct, name: str, arguments: Mapping[str, object]) -> Object:
    return cast("Object", tool(name).call(direct, arguments))


def ids(answer: Object) -> list[str]:
    hits = cast("list[Object]", answer["hits"])
    return [cast("str", hit["id"]) for hit in hits]
