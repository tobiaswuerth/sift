import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sift.api.wire import decoded, encoded
from sift.interfaces.acts import collections, indexing, links, searching
from sift.interfaces.acts.acting import Act, Acting


@dataclass(frozen=True, slots=True)
class Handled:
    run: Callable[[Acting, object], object]
    writes: bool
    kept: Callable[[object], str] | None = None
    alone: bool = False


def handled[R, A](
    act: Act[R, A], *, writes: bool = False, kept: bool = False, alone: bool = False
) -> tuple[str, Handled]:
    def run(acting: Acting, data: object) -> A:
        return act.run(acting, decoded(act.request, data))

    def key(data: object) -> str:
        return json.dumps([act.name, encoded(decoded(act.request, data))], sort_keys=True)

    return act.name, Handled(run, writes, key if kept else None, alone=alone)


HANDLED: Mapping[str, Handled] = MappingProxyType(
    dict(
        (
            handled(collections.ADD, writes=True),
            handled(collections.REMOVE, writes=True, alone=True),
            handled(collections.MOVE, writes=True, alone=True),
            handled(collections.ACTIVATE, writes=True),
            handled(collections.DEACTIVATE, writes=True),
            handled(indexing.INDEX, writes=True),
            handled(indexing.EMBED, writes=True),
            handled(collections.STATUS),
            handled(collections.GET),
            handled(searching.SEARCH, kept=True),
            handled(searching.VSEARCH, kept=True),
            handled(searching.QUERY, kept=True),
            handled(searching.MENTIONS, kept=True),
            handled(searching.SIMILAR, kept=True),
            handled(links.LINKS),
            handled(links.UNRESOLVED),
        )
    )
)
