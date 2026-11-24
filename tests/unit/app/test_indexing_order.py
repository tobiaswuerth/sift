from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.events import Event, Phase, PhaseDone, PhaseStarted
from sift.api.indexing import IndexCollections
from sift.api.search import SearchKeywords
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.search import search
from sift.storage.catalog import known_collections
from sift.storage.database import reading, rows, run, writing
from sift.storage.store import connected
from tests.files import write
from tests.later import LATER
from tests.models import Hashed

WORDS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta")
QUERIES = ("alpha", "beta gamma", '"alpha beta"', "gamma OR delta", "alpha NOT beta", "eps*", "zeta eta alpha")
NOTES = 40


class Placed(NamedTuple):
    id: int
    place: int


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    for n in range(NOTES):
        said = " ".join(WORDS[(n * step + step) % len(WORDS)] for step in range(1 + n % 9))
        write(root / f"n{n:02}.md", f"# Note {n}\n\n{said}\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(root)))
    return home


def found(home: Path) -> list[list[tuple[str, str, float]]]:
    return [
        [(hit.id, hit.lines, hit.score) for hit in search(home, SearchKeywords(query, k=NOTES)).hits]
        for query in QUERIES
    ]


def placed(home: Path) -> list[Placed]:
    (one,) = known_collections(home)
    with connected(one.store, one.id) as connection, reading(connection) as transaction:
        return rows(transaction, Placed, "SELECT id, place FROM chunks ORDER BY place")


def indexed(home: Path) -> list[Event]:
    said: list[Event] = []
    index(home, IndexCollections(), said.append, readers=LATER)
    return said


def test_an_index_orders_its_store_by_topic_once_its_chunks_got_codes_and_finds_what_it_found(home: Path) -> None:
    assert PhaseStarted(Phase.ORDER, None) not in indexed(home)
    before, unordered = found(home), placed(home)
    assert all(each.place == each.id for each in unordered)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    said = indexed(home)
    start = said.index(PhaseStarted(Phase.ORDER, None))
    assert next(each for each in said[start:] if isinstance(each, PhaseDone)) == PhaseDone(len(unordered))
    assert PhaseStarted(Phase.REMAKE, len(unordered)) in said[start:]
    ordered = placed(home)
    assert [each.id for each in ordered] != sorted(each.id for each in ordered)
    assert min(each.place for each in ordered) > max(each.place for each in unordered)
    assert found(home) == before
    assert PhaseStarted(Phase.ORDER, None) not in indexed(home)
    assert placed(home) == ordered


def test_a_store_whose_keywords_are_stale_when_its_order_is_due_is_remade_once(home: Path) -> None:
    indexed(home)
    before = found(home)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    (one,) = known_collections(home)
    with connected(one.store, one.id) as connection, writing(connection) as transaction:
        run(transaction, "UPDATE meta SET value = 'older' WHERE key = 'keyword_fingerprint'")
    phases = [event.phase for event in indexed(home) if isinstance(event, PhaseStarted)]
    assert phases[:2] == [Phase.ORDER, Phase.REMAKE]
    assert phases.count(Phase.REMAKE) == 1
    assert found(home) == before
