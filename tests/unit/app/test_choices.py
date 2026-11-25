from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import pytest

from sift.api.collections import AddCollection, CollectionChanged, ShowStatus, TitleFrom
from sift.api.embedding import EmbedCollections
from sift.api.errors import InvalidRequestError, Problem
from sift.api.events import Event, Noted
from sift.api.indexing import IndexCollections, Skipped
from sift.app.collections import add, status
from sift.app.embedding import embed
from sift.app.filing import Limits
from sift.app.indexing import index
from sift.app.retitling import retitle
from sift.storage.catalog import known_collections
from sift.storage.database import reading, rows, run, writing
from sift.storage.store import connected
from tests.files import linked, write
from tests.homes import found_ids, indexed, indexed_home, quiet, stopped_when_advanced
from tests.models import Hashed


class Titled(NamedTuple):
    path: str
    key: str
    title: str


class Unembedded(NamedTuple):
    path: str
    n: int


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "rl.md", "# Reward shaping\n\nSpeeds up learning.\n")
    write(root / "Hans Müller.md", "## Summary\n\nKey account at Acme.\n")
    write(root / "robots" / "brush.md", "---\ntitle: Brush stroke planning\n---\n# Brush\n\nStrokes.\n")
    write(root / "plain.txt", "A plain text, its first line.\n\nMore of it.\n")
    write(root / "records.jsonl", '{"id": "r1", "text": "## Summary\\n\\nA record."}\n')
    return root


@pytest.fixture
def home(tmp_path: Path, notes: Path) -> Path:
    return indexed_home(tmp_path, notes)


def titles(home: Path) -> dict[str, str]:
    (one,) = known_collections(home)
    with connected(one.store, one.id) as connection, reading(connection) as transaction:
        found = rows(
            transaction,
            Titled,
            "SELECT s.path, d.key, d.title FROM documents d JOIN sources s ON s.id = d.source ORDER BY s.path, d.key",
        )
    return {f"{each.path}#{each.key}" if each.key else each.path: each.title for each in found}


def unembedded(home: Path) -> list[Unembedded]:
    (one,) = known_collections(home)
    with connected(one.store, one.id) as connection, reading(connection) as transaction:
        return rows(
            transaction,
            Unembedded,
            """SELECT s.path, count(*) AS n FROM chunks c JOIN documents d ON d.id = c.document
               JOIN sources s ON s.id = d.source WHERE c.id NOT IN (SELECT chunk FROM vectors)
               GROUP BY s.path ORDER BY s.path""",
        )


def test_adding_a_collection_again_changes_where_its_titles_come_from(home: Path, notes: Path) -> None:
    changed = add(home, AddCollection(str(notes), title_from=TitleFrom.NAME))
    assert changed == CollectionChanged(
        "notes", TitleFrom.NAME, follow_links=False, agents=True, retitled=2, to_embed=0
    )
    assert titles(home) == {
        "Hans Müller.md": "Hans Müller",
        "plain.txt": "plain",
        "records.jsonl#r1": "Summary",
        "rl.md": "rl",
        "robots/brush.md": "Brush stroke planning",
    }
    assert found_ids(home, "Müller") == ["notes/Hans Müller.md"]
    assert found_ids(home, "learning") == ["notes/rl.md"]
    back = add(home, AddCollection(str(notes), name="notes", title_from=TitleFrom.HEADING))
    assert isinstance(back, CollectionChanged)
    assert (back.retitled, titles(home)["Hans Müller.md"]) == (2, "Summary")
    assert found_ids(home, "Müller") == []


def test_retitled_notes_stay_retitled_when_indexed_again(home: Path, notes: Path) -> None:
    add(home, AddCollection(str(notes), title_from=TitleFrom.NAME))
    write(notes / "Hans Müller.md", "## Summary\n\nKey account, now at Initech.\n")
    indexed(home)
    assert (titles(home)["Hans Müller.md"], titles(home)["rl.md"]) == ("Hans Müller", "rl")


def test_a_retitle_stopped_midway_is_finished_by_the_next_index(home: Path) -> None:
    (one,) = known_collections(home)

    with pytest.raises(KeyboardInterrupt):
        retitle(
            home,
            one,
            replace(one.choices, title_from=TitleFrom.NAME),
            stopped_when_advanced,
            Limits(most_files_a_write=1),
        )
    assert known_collections(home)[0].choices.title_from is TitleFrom.HEADING
    indexed(home)
    assert known_collections(home)[0].choices.title_from is TitleFrom.NAME
    assert (titles(home)["Hans Müller.md"], titles(home)["rl.md"]) == ("Hans Müller", "rl")
    indexed(home)
    assert (titles(home)["Hans Müller.md"], titles(home)["rl.md"]) == ("Hans Müller", "rl")


@pytest.mark.parametrize("promised", ["{nope", "[]", '{"agents": 3}'])
def test_a_retitle_left_unfinished_in_words_sift_cannot_read_is_damage_of_the_store(home: Path, promised: str) -> None:
    (one,) = known_collections(home)
    with connected(one.store, one.id) as connection, writing(connection) as transaction:
        run(transaction, "INSERT INTO meta(key, value) VALUES ('retitling', ?)", (promised,))
    (found,) = index(home, IndexCollections(), quiet).collections
    assert isinstance(found.outcome, Skipped)
    assert (found.outcome.code, "is damaged: the retitling" in found.outcome.message) == (Problem.STORE_DAMAGED, True)


def test_retitling_drops_the_vectors_embedded_with_the_old_title_alone(home: Path, notes: Path) -> None:
    write(notes / "long.md", "## Summary\n\n" + "first part " * 300 + "\n\n## Details\n\n" + "second part " * 300)
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    changed = add(home, AddCollection(str(notes), title_from=TitleFrom.NAME))
    assert isinstance(changed, CollectionChanged)
    (left,) = unembedded(home)
    assert (changed.retitled, left.path, changed.to_embed) == (3, "long.md", left.n)


def test_adding_a_collection_again_follows_its_links_or_leaves_them_out_again(home: Path, notes: Path) -> None:
    add(home, AddCollection(str(notes), title_from=TitleFrom.NAME))
    followed = add(home, AddCollection(str(notes), follow_links=True))
    assert followed == CollectionChanged(
        "notes", TitleFrom.NAME, follow_links=True, agents=True, retitled=0, to_embed=0, index_again=True
    )
    add(home, AddCollection(str(notes), follow_links=False))
    (one,) = known_collections(home)
    assert (one.choices.follow_links, one.choices.title_from) == (False, TitleFrom.NAME)


def test_a_folder_of_links_alone_no_longer_followed_drops_what_was_read_through_them_saying_so(tmp_path: Path) -> None:
    write(tmp_path / "outside" / "secret.txt", "a private key\n")
    (tmp_path / "links").mkdir()
    linked(tmp_path / "links" / "notes.txt", str(tmp_path / "outside" / "secret.txt"))
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "links"), follow_links=True))
    indexed(home)
    assert found_ids(home, "private") == ["links/notes.txt"]
    add(home, AddCollection(str(tmp_path / "links"), follow_links=False))
    said = list[Event]()
    index(home, IndexCollections(), said.append)
    assert (found_ids(home, "private"), [each for each in said if isinstance(each, Noted)]) == (
        [],
        [Noted("1 file linked to outside the folder left out: notes.txt")],
    )


def test_a_collection_hidden_from_agents_says_so_and_is_still_searched(home: Path, notes: Path) -> None:
    hidden = add(home, AddCollection(str(notes), agents=False))
    assert isinstance(hidden, CollectionChanged)
    assert hidden.agents is False
    (shown,) = status(home, ShowStatus()).collections
    assert shown.agents is False
    assert found_ids(home, "learning") == ["notes/rl.md"]


NOTHING_ELSE: tuple[Callable[[Path], AddCollection], ...] = (
    lambda notes: AddCollection(str(notes)),
    lambda notes: AddCollection(str(notes), id_field="key"),
    lambda notes: AddCollection(str(notes), title_from=TitleFrom.NAME, name="other"),
    lambda notes: AddCollection(str(notes), title_from=TitleFrom.HEADING),
    lambda notes: AddCollection(str(notes), nested=True, follow_links=True),
)


@pytest.mark.parametrize("again", NOTHING_ELSE)
def test_adding_a_collection_again_with_nothing_to_change_or_anything_else_is_refused(
    home: Path, notes: Path, again: Callable[[Path], AddCollection]
) -> None:
    with pytest.raises(InvalidRequestError, match="is the collection 'notes' already"):
        add(home, again(notes))
    (one,) = known_collections(home)
    assert (one.choices.title_from, one.choices.follow_links) == (TitleFrom.HEADING, False)
