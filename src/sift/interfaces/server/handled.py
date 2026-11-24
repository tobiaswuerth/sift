from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sift.api.wire import decoded
from sift.interfaces.acts import collections, indexing, links, searching
from sift.interfaces.acts.acting import Act, Acting


@dataclass(frozen=True, slots=True)
class Handled:
    run: Callable[[Acting, object], object]
    writes: bool


def handled[R, A](act: Act[R, A], *, writes: bool = False) -> tuple[str, Handled]:
    def run(acting: Acting, data: object) -> A:
        return act.run(acting, decoded(act.request, data))

    return act.name, Handled(run, writes)


HANDLED: Mapping[str, Handled] = MappingProxyType(
    dict(
        (
            handled(collections.ADD, writes=True),
            handled(collections.REMOVE, writes=True),
            handled(collections.MOVE, writes=True),
            handled(collections.ACTIVATE, writes=True),
            handled(collections.DEACTIVATE, writes=True),
            handled(indexing.INDEX, writes=True),
            handled(indexing.EMBED, writes=True),
            handled(collections.STATUS),
            handled(collections.GET),
            handled(searching.SEARCH),
            handled(searching.VSEARCH),
            handled(searching.QUERY),
            handled(searching.MENTIONS),
            handled(searching.SIMILAR),
            handled(links.LINKS),
            handled(links.UNRESOLVED),
        )
    )
)
