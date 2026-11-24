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


def handled[R, A](act: Act[R, A], *, writes: bool = False) -> Handled:
    def run(acting: Acting, data: object) -> A:
        return act.run(acting, decoded(act.request, data))

    return Handled(run, writes)


HANDLED: Mapping[str, Handled] = MappingProxyType(
    {
        "add": handled(collections.ADD, writes=True),
        "rm": handled(collections.REMOVE, writes=True),
        "move": handled(collections.MOVE, writes=True),
        "activate": handled(collections.ACTIVATE, writes=True),
        "deactivate": handled(collections.DEACTIVATE, writes=True),
        "index": handled(indexing.INDEX, writes=True),
        "embed": handled(indexing.EMBED, writes=True),
        "status": handled(collections.STATUS),
        "get": handled(collections.GET),
        "search": handled(searching.SEARCH),
        "vsearch": handled(searching.VSEARCH),
        "query": handled(searching.QUERY),
        "mentions": handled(searching.MENTIONS),
        "similar": handled(searching.SIMILAR),
        "links": handled(links.LINKS),
        "unresolved": handled(links.UNRESOLVED),
    }
)
