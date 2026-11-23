import dataclasses

from sift.api.collections import (
    CollectionAdded,
    CollectionStatus,
    Counted,
    Status,
    TitleFrom,
    Unavailable,
    UnreadableFile,
)
from sift.api.documents import Documents, DocumentText, Unfound
from sift.api.errors import NotFoundError, StoreUnavailableError
from sift.api.indexing import CollectionIndexed, Indexed, IndexResult
from sift.api.search import Hit, SearchResult
from sift.interfaces.cli.shown import Shown, added, documents, failure, hits, indexed, status
from sift.interfaces.console.style import PLAIN, Style

ADDED = CollectionAdded(
    "notes", "/data/notes", "/home/.sift/collections/notes.sqlite", TitleFrom.HEADING, vault=False, follow_links=False
)
NOTES = CollectionStatus("notes", "/data/notes", "/s/notes.sqlite", Counted(12_431, 76_738, 412_300_000, 0, ()))
STATUS = Status(
    "/home/.sift",
    (
        NOTES,
        CollectionStatus(
            "Papers-日本",
            "/data/papers",
            "/s/papers.sqlite",
            Unavailable("the store is missing: /s/papers.sqlite", "connect the drive it lies on"),
        ),
    ),
)
EMPTY = Shown((), ("empty  no collections yet", " next  sift add FOLDER"))


def test_an_added_collection_shows_where_its_store_lies_and_what_to_run_next() -> None:
    assert added(ADDED, "linux", PLAIN) == Shown(
        (),
        (
            "added  notes  /data/notes",
            "store  /home/.sift/collections/notes.sqlite",
            " next  sift index notes",
        ),
    )


def test_titles_from_file_names_are_said_with_the_vault_that_chose_them() -> None:
    by_name = dataclasses.replace(ADDED, title_from=TitleFrom.NAME)
    assert added(by_name, "linux", PLAIN).messages[2] == "title  from file names"
    vault = dataclasses.replace(by_name, vault=True)
    assert added(vault, "linux", PLAIN).messages[2] == "title  from file names, as in an Obsidian vault"
    assert "title" not in "".join(added(dataclasses.replace(ADDED, vault=True), "linux", PLAIN).messages)


def test_links_followed_out_of_the_folder_are_said() -> None:
    following = dataclasses.replace(ADDED, follow_links=True)
    assert added(following, "linux", PLAIN).messages[2] == "links  followed out of the folder"


def test_the_next_command_is_quoted_for_the_shell_there() -> None:
    named = dataclasses.replace(ADDED, name="Müller")
    assert added(named, "linux", PLAIN).messages[-1] == " next  sift index 'Müller'"
    assert added(named, "win32", PLAIN).messages[-1] == " next  sift index Müller"


def test_labels_take_their_colour_where_colour_is_on() -> None:
    painted = added(ADDED, "linux", Style(colour=True)).messages
    assert painted[0].startswith("\x1b[32madded\x1b[0m  notes")
    assert painted[-1].startswith(" \x1b[2mnext\x1b[0m  sift")


def test_status_is_a_table_whose_numbers_align_and_whose_problems_are_said_beside_it() -> None:
    assert status(STATUS, "linux", PLAIN) == Shown(
        (
            "collection   folder        documents  chunks    size",
            "notes        /data/notes      12,431  76,738  412 MB",
            "Papers-日本  /data/papers          -       -       -",
        ),
        (
            "error  Papers-日本: the store is missing: /s/papers.sqlite",
            "  fix  connect the drive it lies on",
        ),
    )


def test_status_counts_unreadable_files_and_lists_them_when_asked() -> None:
    failed = (UnreadableFile("huge.txt", "too large"), UnreadableFile("locked.md", "Permission denied"))
    counted = dataclasses.replace(NOTES, contents=Counted(1, 2, 3_000, 2, failed))
    assert status(Status("/h", (counted,)), "linux", PLAIN).results == (
        "collection  folder       documents  chunks    size  unreadable",
        "notes       /data/notes          1       2  3.0 kB           2",
        "",
        "collection  file       problem",
        "notes       huge.txt   too large",
        "notes       locked.md  Permission denied",
    )


def test_status_of_no_collection_says_how_to_add_one() -> None:
    assert status(Status("/home/.sift", ()), "linux", PLAIN) == EMPTY


def test_indexing_says_what_it_did_as_it_goes_and_nothing_after() -> None:
    result = IndexResult((CollectionIndexed("notes", Indexed(3, 3, 0, 0, 0)),))
    assert indexed(result, "linux", PLAIN) == Shown((), ())
    assert indexed(IndexResult(()), "linux", PLAIN) == EMPTY


def test_one_document_is_printed_as_its_text() -> None:
    one = Documents((DocumentText("notes/a.md", "A", "# A\n\ntext\n"),), ())
    assert documents(one, "linux", PLAIN) == Shown(("# A\n\ntext",), ())
    unended = Documents((DocumentText("data/r", "R", "a record"),), ())
    assert documents(unended, "linux", PLAIN).results == ("a record",)
    assert documents(Documents((DocumentText("notes/empty.md", "empty", ""),), ()), "linux", PLAIN).results == ()


def test_several_documents_are_headed_by_their_ids_and_those_not_found_said() -> None:
    found = (DocumentText("notes/a.md", "A", "# A\n"), DocumentText("notes/b.md", "B", "# B\n"))
    unfound = (Unfound("notes/x.md", "unknown document 'notes/x.md'", "take the id from a search result"),)
    assert documents(Documents(found, unfound), "linux", PLAIN) == Shown(
        ("==> notes/a.md <==", "# A", "", "==> notes/b.md <==", "# B"),
        ("error  notes/x.md: unknown document 'notes/x.md'", "  fix  take the id from a search result"),
    )
    assert documents(Documents((), unfound), "linux", PLAIN).results == ()


def test_an_id_that_would_steer_the_terminal_heads_its_document_as_an_escape() -> None:
    found = (DocumentText("notes/esc\x1b[2J.md", "x", ""), DocumentText("notes/b.md", "B", "# B\n"))
    assert documents(Documents(found, ()), "linux", PLAIN).results == (
        "==> notes/esc\\x1b[2J.md <==",
        "",
        "==> notes/b.md <==",
        "# B",
    )


def test_a_failure_says_what_failed_and_how_to_fix_it() -> None:
    assert failure(NotFoundError("no collection is named 'x'", "the collections are: a"), PLAIN) == Shown(
        (), ("error  no collection is named 'x'", "  fix  the collections are: a"), failed=True
    )
    assert failure(StoreUnavailableError("busy"), PLAIN) == Shown((), ("error  busy",), failed=True)
    assert failure(OSError(28, "No space left on device"), PLAIN) == Shown(
        (), ("error  [Errno 28] No space left on device",), failed=True
    )


def test_a_partial_search_says_better_matches_may_be_missing() -> None:
    found = SearchResult((Hit("notes/a.md", "A", "1-2", 1.5, None),), 1, (), partial=True)
    dot = "  \N{MIDDLE DOT}  "
    said = dot.join(("1 result", "keyword", "partial: better matches may be missing", "1 collection", "0.2 s"))
    assert hits(found, 0.25, "linux", PLAIN).messages == (said,)
    assert "partial" not in hits(dataclasses.replace(found, partial=False), 0.25, "linux", PLAIN).messages[0]


def test_a_hit_names_its_copies_in_other_collections_under_its_place() -> None:
    hit = Hit("notes/halo.md", "Halo", "1-3", 3.25, None, ("more/copy.md", "papers/halo.md"))
    shown = hits(SearchResult((hit,), 3, ()), 0.25, "linux", PLAIN, "hybrid")
    assert shown.results == (
        "1  3.25  Halo",
        "         notes/halo.md:1-3",
        "         also more/copy.md, papers/halo.md",
    )
    assert " hybrid " in shown.messages[-1]


def test_a_hit_shows_the_front_matter_asked_for_under_its_place() -> None:
    hit = Hit("notes/halo.md", "Halo", "1-3", 3.25, None, fields={"created": "2024-06-15", "tags": ["a", "é"], "n": 2})
    shown = hits(SearchResult((hit,), 1, ()), 0.25, "linux", PLAIN)
    assert shown.results == (
        "1  3.25  Halo",
        "         notes/halo.md:1-3",
        '         created: 2024-06-15  tags: ["a", "é"]  n: 2',
    )
    assert len(hits(SearchResult((dataclasses.replace(hit, fields={}),), 1, ()), 0.25, "linux", PLAIN).results) == 2


def test_a_document_shows_the_front_matter_asked_for_before_its_text() -> None:
    found = DocumentText("notes/a.md", "A", "# A\n", fields={"rank": 2, "tags": ["x"]})
    assert documents(Documents((found,), ()), "linux", PLAIN).results == ("rank: 2", 'tags: ["x"]', "", "# A")
