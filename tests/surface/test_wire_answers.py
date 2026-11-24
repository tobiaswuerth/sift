from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType

import pytest

from sift.api.events import (
    Advanced,
    CollectionStarted,
    Event,
    Finished,
    Listening,
    Noted,
    Phase,
    PhaseDone,
    PhaseStarted,
)
from sift.api.wire import Json, decoded, encoded
from sift.interfaces.acts import collections, indexing, links, searching
from sift.interfaces.acts.acting import Act, answer_kind
from sift.interfaces.server.client import event_of
from tests.answers import SAMPLES


def read_back[R, A](act: Act[R, A], data: Json) -> A:
    return decoded(answer_kind(act), data)


READ_BACK: Mapping[str, Callable[[Json], object]] = MappingProxyType(
    {
        "add": partial(read_back, collections.ADD),
        "add again": partial(read_back, collections.ADD),
        "rm": partial(read_back, collections.REMOVE),
        "move": partial(read_back, collections.MOVE),
        "activate": partial(read_back, collections.ACTIVATE),
        "deactivate": partial(read_back, collections.DEACTIVATE),
        "status": partial(read_back, collections.STATUS),
        "index": partial(read_back, indexing.INDEX),
        "index files": partial(read_back, indexing.INDEX),
        "embed": partial(read_back, indexing.EMBED),
        "get": partial(read_back, collections.GET),
        "search": partial(read_back, searching.SEARCH),
        "links": partial(read_back, links.LINKS),
        "unresolved": partial(read_back, links.UNRESOLVED),
    }
)
EVENTS: tuple[Event, ...] = (
    CollectionStarted("notes", "/data/notes"),
    PhaseStarted(Phase.READ, 3),
    PhaseStarted(Phase.SCAN, None),
    Advanced(2),
    PhaseDone(3),
    Noted("a.pdf: a scan", "turn OCR on"),
    Noted("said"),
    Finished(1, 2, 3, 4),
    Listening("http://127.0.0.1:8765", "/home/.sift"),
)


def test_every_answer_sample_is_read_back_by_its_act() -> None:
    assert set(READ_BACK) == set(SAMPLES)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_answer_reads_back_from_its_json_as_it_was(name: str) -> None:
    assert READ_BACK[name](encoded(SAMPLES[name])) == SAMPLES[name]


@pytest.mark.parametrize("event", EVENTS)
def test_every_event_reads_back_from_its_json_as_it_was(event: Event) -> None:
    assert event_of(type(event).__name__, encoded(event)) == event
