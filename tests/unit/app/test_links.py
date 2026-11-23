from contextlib import closing
from pathlib import Path

import pytest

from sift.api.collections import AddCollection
from sift.api.errors import InvalidRequestError, NotFoundError
from sift.api.indexing import IndexCollections
from sift.api.links import Direction, LinkIn, LinkingLine, LinkOut, ListUnresolved, ShowLinks, Target
from sift.app.collections import add
from sift.app.indexing import index
from sift.app.links import links, unresolved
from sift.app.notes import NOTE_REMEDY, Note
from sift.storage.catalog import known_collections
from sift.storage.database import reading
from sift.storage.store import open_store

LONG = "word " * 100


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def vault(tmp_path: Path, files: dict[str, str], name: str = "v") -> Path:
    home = tmp_path / "home"
    for path, text in files.items():
        write(tmp_path / name / path, text)
    add(home, AddCollection(str(tmp_path / name)))
    index(home, IndexCollections(), lambda _: None)
    return home


def leads(found: tuple[LinkOut, ...] | None) -> list[tuple[str, object]]:
    assert found is not None
    return [
        (
            each.target,
            (each.id, each.via) if each.id else ("ambiguous", list(each.ambiguous)) if each.ambiguous else "unresolved",
        )
        for each in found
    ]


def test_a_link_resolves_as_obsidian_resolves_it(tmp_path: Path) -> None:
    home = vault(
        tmp_path,
        {
            "People/Hans.md": "---\naliases: [Hansi, Johannes]\n---\n",
            "Archive/People/Hans.md": "",
            "Projects/Hans.md": "",
            "Projects/Plan.md": "[[Hans]] [[People/Hans]] [[Archive/People/Hans]] [[Johannes]] [[Plan]]",
            "Projects/Sub/Note.md": "[[Hans]] [[archive/people/hans.md]] [[Nope/Hans]] ![[notes.txt]] [[notes]]",
            "Other.md": "[[Hans]] [[Hansi]] [[Dup]]",
            "A/Dup.md": "",
            "B/Dup.md": "",
            "C/X.md": "---\naliases: Dup\n---\n",
            "notes.txt": "plain text",
        },
    )
    hans = ["v/Archive/People/Hans.md", "v/People/Hans.md", "v/Projects/Hans.md"]
    assert leads(links(home, ShowLinks("v/Projects/Plan.md")).out) == [
        ("Hans", ("v/Projects/Hans.md", "name")),
        ("People/Hans", ("v/People/Hans.md", "path")),
        ("Archive/People/Hans", ("v/Archive/People/Hans.md", "path")),
        ("Johannes", ("v/People/Hans.md", "alias")),
        ("Plan", ("v/Projects/Plan.md", "name")),
    ]
    assert leads(links(home, ShowLinks("v/Projects/Sub/Note.md")).out) == [
        ("Hans", ("ambiguous", hans)),
        ("archive/people/hans.md", ("v/Archive/People/Hans.md", "path")),
        ("Nope/Hans", "unresolved"),
        ("notes.txt", ("v/notes.txt", "name")),
        ("notes", "unresolved"),
    ]
    assert leads(links(home, ShowLinks("v/Other.md")).out)[2] == ("Dup", ("ambiguous", ["v/A/Dup.md", "v/B/Dup.md"]))


@pytest.fixture
def people(tmp_path: Path) -> Path:
    return vault(
        tmp_path,
        {
            "People/Hans Müller.md": '---\naliases: [Hans Mueller]\norg: "[[Acme AG]]"\n---\n# Hans Müller\n',
            "Orgs/Acme AG.md": "# Acme AG\n\nRun by [[Hans Müller]].\n",
            "Meetings/Offer.md": f"# Offer\n\nWith [[Hans Mueller]] about [[Contract]].\n\n{LONG}[[Acme AG]] {LONG}\n",
            "Meetings/Notes.md": "# Notes\n\n[[Offer]] and [[Offer]] again.\n",
        },
    )


def test_links_out_say_where_they_lead_and_links_in_show_their_lines(people: Path) -> None:
    found = links(people, ShowLinks("v/Meetings/Offer.md", Direction.BOTH))
    assert (found.id, found.title) == ("v/Meetings/Offer.md", "Offer")
    assert found.out == (
        LinkOut(3, "Hans Mueller", None, None, "alias", "v/People/Hans Müller.md", "Hans Müller", (), None),
        LinkOut(3, "Contract", None, None, "unresolved", None, None, (), None),
        LinkOut(5, "Acme AG", None, None, "name", "v/Orgs/Acme AG.md", "Acme AG", (), None),
    )
    assert found.into == (
        LinkIn("v/Meetings/Notes.md", "Notes", (LinkingLine(3, None, "[[Offer]] and [[Offer]] again."),), None),
    )
    (acme,) = [
        each for each in links(people, ShowLinks("v/Orgs/Acme AG.md", Direction.IN)).into or () if "Offer" in each.id
    ]
    (line,) = acme.lines
    assert line.text.startswith("\N{HORIZONTAL ELLIPSIS}")
    assert line.text.endswith("\N{HORIZONTAL ELLIPSIS}")
    assert "[[Acme AG]]" in line.text


def test_links_keep_to_a_field_go_two_steps_and_stop_at_k(people: Path) -> None:
    hans = links(people, ShowLinks("v/People/Hans Müller.md", Direction.BOTH, field="org"))
    assert [each.target for each in hans.out or ()] == ["Acme AG"]
    assert hans.into == ()
    further = links(people, ShowLinks("v/Meetings/Notes.md", depth=2)).out or ()
    assert [(each.target, each.through) for each in further] == [
        ("Offer", None),
        ("Offer", None),
        ("Hans Mueller", "v/Meetings/Offer.md"),
        ("Acme AG", "v/Meetings/Offer.md"),
    ]
    behind = links(people, ShowLinks("v/People/Hans Müller.md", Direction.IN, depth=2)).into or ()
    assert [(each.id, each.through) for each in behind] == [
        ("v/Meetings/Offer.md", None),
        ("v/Orgs/Acme AG.md", None),
        ("v/Meetings/Notes.md", "v/Meetings/Offer.md"),
    ]
    assert len(links(people, ShowLinks("v/Meetings/Offer.md", k=1)).out or ()) == 1


def test_a_note_is_named_by_its_id_or_as_links_name_it(people: Path, tmp_path: Path) -> None:
    assert links(people, ShowLinks("Hans Mueller")).id == "v/People/Hans Müller.md"
    assert links(people, ShowLinks("[[Acme AG]]")).id == "v/Orgs/Acme AG.md"
    with pytest.raises(NotFoundError, match="no note or document"):
        links(people, ShowLinks("Nobody"))
    write(tmp_path / "v" / "Other" / "Acme AG.md", "# Acme AG again\n")
    index(people, IndexCollections(), lambda _: None)
    with pytest.raises(InvalidRequestError, match="names several notes"):
        links(people, ShowLinks("Acme AG"))


@pytest.mark.parametrize(
    "request_, refusal",
    [(ShowLinks("v/Orgs/Acme AG.md", depth=3), "neither 1 nor 2"), (ShowLinks("x", k=0), "no result")],
)
def test_links_asked_otherwise_are_refused(people: Path, request_: ShowLinks, refusal: str) -> None:
    with pytest.raises(InvalidRequestError, match=refusal):
        links(people, request_)


def test_link_targets_no_note_answers_to_are_listed_most_linked_first(tmp_path: Path) -> None:
    home = vault(
        tmp_path,
        {
            "People/Hans.md": "",
            "a.md": "[[Other/Hans]] [x](Nope/Hans.md) [[Hans]] [[Missing]]",
            "b.md": "[[missing.md]] [[Missing]]",
            "sub/c.md": "[x](../Missing.md) [y](../People/Hans.md) [[./Gone]]",
            "sub/Hans.md": "",
        },
    )
    assert unresolved(home, ListUnresolved()).targets == (
        Target("Missing", 3, ("v/a.md", "v/b.md", "v/sub/c.md")),
        Target("./Gone", 1, ("v/sub/c.md",)),
        Target("Nope/Hans.md", 1, ("v/a.md",)),
        Target("Other/Hans", 1, ("v/a.md",)),
    )
    hans = links(home, ShowLinks("v/People/Hans.md", Direction.IN)).into or ()
    assert [each.id for each in hans] == ["v/sub/c.md"]
    assert len(unresolved(home, ListUnresolved(k=1)).targets) == 1


def test_links_stay_in_their_collection(tmp_path: Path) -> None:
    home = vault(tmp_path, {"a.md": "[[Shared]]"}, "first")
    vault(tmp_path, {"Shared.md": "# Shared\n"}, "second")
    assert leads(links(home, ShowLinks("first/a.md")).out) == [("Shared", "unresolved")]
    assert unresolved(home, ListUnresolved("first")).targets == (Target("Shared", 1, ("first/a.md",)),)


def test_a_note_named_as_a_link_writes_it_with_a_heading_or_shown_text_is_found(people: Path) -> None:
    assert links(people, ShowLinks("[[Acme AG|Acme]]")).id == "v/Orgs/Acme AG.md"
    assert links(people, ShowLinks("[[Acme AG#Team]]")).id == "v/Orgs/Acme AG.md"
    with pytest.raises(NotFoundError):
        links(people, ShowLinks("[[Acme AG"))


def test_a_name_many_notes_answer_to_names_a_few_of_them(tmp_path: Path) -> None:
    home = vault(tmp_path, {f"{n}/Same.md": "" for n in range(7)})
    with pytest.raises(InvalidRequestError, match=r"names several notes: v/0/Same.md, .*v/4/Same.md, \.\.\."):
        links(home, ShowLinks("Same"))


def test_a_damaged_store_is_passed_over_by_names_and_unresolved_saying_so(people: Path, tmp_path: Path) -> None:
    vault(tmp_path, {"x.md": "[[Nowhere]]"}, "w")
    store = people / "collections" / "w.sqlite"
    store.write_bytes(b"\xff" * 4096 + store.read_bytes()[4096:])
    named = links(people, ShowLinks("Acme AG"))
    assert (named.id, [each.collection for each in named.unsearched]) == ("v/Orgs/Acme AG.md", ["w"])
    assert links(people, ShowLinks("v/Orgs/Acme AG.md")).unsearched == ()
    found = unresolved(people, ListUnresolved())
    assert found.targets == (Target("Contract", 1, ("v/Meetings/Offer.md",)),)
    assert [each.collection for each in found.unsearched] == ["w"]
    with pytest.raises(NotFoundError, match=r"; not searched: w: the store .* is damaged") as refused:
        links(people, ShowLinks("Nowhere"))
    assert refused.value.remedy is not None
    assert refused.value.remedy.endswith(f"; {NOTE_REMEDY}")
    write(tmp_path / "v" / "Other" / "Acme AG.md", "# Acme AG again\n")
    index(people, IndexCollections(names=("v",)), lambda _: None)
    with pytest.raises(InvalidRequestError, match=r"Acme AG\.md; not searched: w: the store .* is damaged"):
        links(people, ShowLinks("Acme AG"))


def test_the_cut_at_k_and_a_field_hold_for_the_second_step_too(people: Path) -> None:
    further = links(people, ShowLinks("v/Meetings/Notes.md", depth=2, k=3)).out or ()
    assert [(each.target, each.through) for each in further] == [
        ("Offer", None),
        ("Offer", None),
        ("Hans Mueller", "v/Meetings/Offer.md"),
    ]
    org = links(people, ShowLinks("v/Orgs/Acme AG.md", Direction.IN, depth=2, field="org")).into or ()
    assert [(each.id, each.through) for each in org] == [("v/People/Hans Müller.md", None)]


@pytest.mark.parametrize("written", ["[[Acme AG]] b]]", "[[Acme AG]]]]", "[[Acme AG]] [[Offer]]"])
def test_a_name_holding_more_than_one_link_names_no_note(people: Path, written: str) -> None:
    with pytest.raises(NotFoundError, match="no note or document"):
        links(people, ShowLinks(written))


def test_a_note_found_by_name_is_read_again_at_its_path_alone(people: Path) -> None:
    (one,) = known_collections(people)
    with closing(open_store(one.store, one.id)) as connection, reading(connection) as transaction:
        assert Note(one, "Orgs/Acme AG.md").find(transaction) > 0
        with pytest.raises(NotFoundError, match=r"unknown document 'v/Acme AG\.md'"):
            Note(one, "Acme AG.md").find(transaction)


def test_every_store_a_name_was_not_looked_up_in_is_named(people: Path, tmp_path: Path) -> None:
    for name in ("w", "x"):
        vault(tmp_path, {"y.md": "# Y\n"}, name)
        store = people / "collections" / f"{name}.sqlite"
        store.write_bytes(b"\xff" * 4096 + store.read_bytes()[4096:])
    with pytest.raises(NotFoundError, match=r"not searched: w: .*; not searched: x: "):
        links(people, ShowLinks("Nowhere"))
    assert [each.collection for each in links(people, ShowLinks("Acme AG")).unsearched] == ["w", "x"]
