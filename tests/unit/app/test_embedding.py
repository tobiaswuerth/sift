import contextlib
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import CollectionEmbedded, EmbedCollections, Embedded
from sift.api.errors import InvalidRequestError
from sift.api.events import Advanced, Event, Noted, Phase, PhaseDone, PhaseStarted
from sift.api.indexing import IndexCollections, Skipped
from sift.app.collections import add
from sift.app.embedding import BROKEN_REMEDY, Embedding, embed
from sift.app.indexing import index
from sift.storage.catalog import CATALOG
from sift.storage.vectors import EMBED_REMEDY
from tests.models import BROKEN, Hashed

LONG = "\n\n".join(["Pruning skips what cannot win. " * 20] * 8)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def counted(store: Path, sql: str) -> int:
    with contextlib.closing(sqlite3.connect(store)) as database:
        (found,) = cast("tuple[int]", database.execute(sql).fetchone())
        return found


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "pruning.md", f"# Block-Max Pruning\n\nRanges are scored best bound first.\n\n## More\n\n{LONG}\n")
    write(root / "halo.md", "# Halo\n\nDark matter halo models.\n")
    return root


@pytest.fixture
def home(tmp_path: Path, notes: Path) -> Path:
    home = tmp_path / "home"
    add(home, AddCollection(str(notes)))
    index(home, IndexCollections(), lambda _: None)
    return home


def store(home: Path, name: str = "notes") -> Path:
    return home / "collections" / f"{name}.sqlite"


def chunks(home: Path, name: str = "notes") -> int:
    return counted(store(home, name), "SELECT count(*) FROM chunks")


def test_every_chunk_is_embedded_once_with_its_progress(home: Path) -> None:
    said: list[Event] = []
    result = embed(home, EmbedCollections(), said.append, Hashed(), Embedding(chunks_a_write=2))
    total = chunks(home)
    assert result.collections == (CollectionEmbedded("notes", Embedded(total, 0)),)
    assert said[1] == PhaseStarted(Phase.EMBED, total)
    assert [event.done for event in said if isinstance(event, Advanced)] == [*range(2, total, 2), total]
    assert said[-1] == PhaseDone(total)
    assert counted(store(home), "SELECT count(*) FROM vectors") == total
    again: list[Event] = []
    assert embed(home, EmbedCollections(), again.append, Hashed()).collections[0].outcome == Embedded(0, 0)
    assert not [event for event in again if isinstance(event, PhaseStarted)]


def test_chunks_after_the_first_are_embedded_with_their_title_and_documents_as_documents(home: Path) -> None:
    models = Hashed()
    embed(home, EmbedCollections(), lambda _: None, models)
    texts = [text for asked, query in models.asked for text in asked if not query]
    assert "# Halo\n\nDark matter halo models." in texts
    pruning = [text for text in texts if "Pruning skips" in text]
    assert pruning[0].startswith("# Block-Max Pruning\n\nRanges are scored best bound first.")
    assert pruning[1].startswith("Block-Max Pruning\n\nPruning skips what cannot win.")
    assert all(not query for _, query in models.asked)


def test_a_chunk_the_model_gives_no_vector_for_is_counted_and_stays_to_embed(home: Path, notes: Path) -> None:
    write(notes / "odd.md", f"# Odd\n\nAn {BROKEN} passage.\n")
    index(home, IndexCollections(), lambda _: None)
    said: list[Event] = []
    outcome = embed(home, EmbedCollections(), said.append, Hashed()).collections[0].outcome
    assert outcome == Embedded(chunks(home) - 1, 1)
    assert said[-1] == Noted("1 chunk could not be embedded: the model gave no vector", BROKEN_REMEDY)
    assert embed(home, EmbedCollections(), lambda _: None, Hashed()).collections[0].outcome == Embedded(0, 1)


def test_a_collection_of_another_model_is_skipped_until_every_one_is_embedded_again(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "a.md", "# A\n\nAnother note.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    with contextlib.closing(sqlite3.connect(home / CATALOG)) as catalog, catalog:
        catalog.execute("UPDATE collections SET active = 0 WHERE name = 'more'")
    other = Hashed(name="other-model", width=32)
    (skipped,) = embed(home, EmbedCollections(), lambda _: None, other).collections
    assert other.asked == []
    assert skipped == CollectionEmbedded(
        "notes",
        Skipped(
            "the store holds vectors of hashed-words, not of other-model", EMBED_REMEDY.format(model="other-model")
        ),
    )
    with pytest.raises(InvalidRequestError, match="--reset embeds every collection again"):
        embed(home, EmbedCollections(("notes",), reset=True), lambda _: None, other)
    (redone,) = embed(home, EmbedCollections(reset=True), lambda _: None, other).collections
    assert redone == CollectionEmbedded("notes", Embedded(chunks(home), 0))
    assert counted(store(home, "more"), "SELECT count(*) FROM vectors") == 0
    assert embed(home, EmbedCollections(("more",)), lambda _: None, other).collections[0].outcome == Embedded(1, 0)


def test_a_collection_whose_store_is_gone_is_skipped_and_others_embedded(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "a.md", "# A\n\nAnother note.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    index(home, IndexCollections(), lambda _: None)
    store(home).unlink()
    found = {each.name: each.outcome for each in embed(home, EmbedCollections(), lambda _: None, Hashed()).collections}
    assert isinstance(found["notes"], Skipped)
    assert found["more"] == Embedded(1, 0)


def test_a_reset_says_which_store_it_could_not_clear(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "a.md", "# A\n\nAnother note.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    index(home, IndexCollections(), lambda _: None)
    with contextlib.closing(sqlite3.connect(home / CATALOG)) as catalog, catalog:
        catalog.execute("UPDATE collections SET active = 0 WHERE name = 'more'")
    store(home, "more").unlink()
    said: list[Event] = []
    embed(home, EmbedCollections(reset=True), said.append, Hashed())
    assert next(event.text for event in said if isinstance(event, Noted)).startswith(
        "its vectors could not be forgotten:"
    )
