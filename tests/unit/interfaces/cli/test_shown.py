import dataclasses
import errno
import json

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
from sift.api.errors import InvalidRequestError, NotFoundError, StoreUnavailableError
from sift.api.indexing import CollectionIndexed, Indexed, IndexResult
from sift.api.links import LinkIn, LinkingLine, LinkOut, Links, Target, Unresolved
from sift.api.search import Hit, SearchResult, Unsearched
from sift.interfaces.cli.shown.collections import added, status
from sift.interfaces.cli.shown.documents import documents
from sift.interfaces.cli.shown.indexing import said_as_it_went
from sift.interfaces.cli.shown.links import links as shown_links
from sift.interfaces.cli.shown.links import unresolved as shown_unresolved
from sift.interfaces.cli.shown.outcome import FAILED, USAGE, Shown, failure
from sift.interfaces.cli.shown.searching import hits
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


def test_status_counts_links_where_a_collection_holds_any() -> None:
    linked = dataclasses.replace(NOTES, contents=Counted(1, 2, 3_000, 0, (), links=3_249, unresolved=7))
    assert status(Status("/h", (linked, STATUS.collections[1])), "linux", PLAIN).results == (
        "collection   folder        documents  chunks    size  links  unresolved",
        "notes        /data/notes           1       2  3.0 kB  3,249           7",
        "Papers-日本  /data/papers          -       -       -      -           -",
    )


def test_status_counts_links_where_none_leads_nowhere() -> None:
    linked = dataclasses.replace(NOTES, contents=Counted(1, 2, 3_000, 0, (), links=5, unresolved=0))
    assert status(Status("/h", (linked,)), "linux", PLAIN).results[0].split()[-2:] == ["links", "unresolved"]


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
    assert said_as_it_went(result, "linux", PLAIN) == Shown((), ())
    assert said_as_it_went(IndexResult(()), "linux", PLAIN) == EMPTY


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
        (), ("error  no collection is named 'x'", "  fix  the collections are: a"), FAILED
    )
    assert failure(StoreUnavailableError("busy", "wait"), PLAIN) == Shown((), ("error  busy", "  fix  wait"), FAILED)
    assert failure(OSError(errno.ENOSPC, "No space left on device"), PLAIN) == Shown(
        (), ("error  [Errno 28] No space left on device", "  fix  free space on the disk"), FAILED
    )
    assert (
        failure(OSError(errno.EIO, "Input/output error"), PLAIN).messages[1]
        == "  fix  check the path and its permissions"
    )
    assert failure(InvalidRequestError("-k 0 asks for no result", "ask for one"), PLAIN).code == USAGE


def test_a_failure_asked_for_in_json_is_json_too() -> None:
    found = failure(StoreUnavailableError("busy", "wait"), PLAIN, as_json=True)
    assert json.loads(found.results[0]) == {
        "error": {"kind": "store_unavailable", "message": "busy", "remedy": "wait", "field": None}
    }
    assert found.messages == ("error  busy", "  fix  wait")
    assert json.loads(failure(OSError(errno.ENOENT, "gone"), PLAIN, as_json=True).results[0])["error"]["kind"] == "os"


def test_a_partial_search_says_better_matches_may_be_missing() -> None:
    found = SearchResult((Hit("notes/a.md", "A", "1-2", 1.5, None),), 1, (), partial=True)
    dot = "  \N{MIDDLE DOT}  "
    said = dot.join(("1 result", "keyword", "partial: better matches may be missing", "1 collection", "0.2 s"))
    assert hits(found, 0.25, "linux", PLAIN).messages == (said,)
    assert "partial" not in hits(dataclasses.replace(found, partial=False), 0.25, "linux", PLAIN).messages[0]


def test_stores_ruled_out_by_the_links_are_said_apart_from_those_searched() -> None:
    dot = "  \N{MIDDLE DOT}  "
    found = SearchResult((), 1, (), ruled_out=2)
    said = dot.join(("no matches", "keyword", "1 collection, 2 ruled out by the links", "0.2 s"))
    assert hits(found, 0.25, "linux", PLAIN).messages == (said,)
    assert hits(dataclasses.replace(found, searched=0), 0.25, "linux", PLAIN).messages[0].startswith("no matches")


def test_a_hit_names_its_copies_in_other_collections_under_its_place() -> None:
    hit = Hit("notes/halo.md", "Halo", "1-3", 3.25, None, ("more/copy.md", "papers/halo.md"))
    shown = hits(SearchResult((hit,), 3, ()), 0.25, "linux", PLAIN, "hybrid")
    assert shown.results == (
        "1  3.25  Halo",
        "         notes/halo.md:1-3",
        "         also more/copy.md, papers/halo.md",
    )
    assert " hybrid " in shown.messages[-1]


def test_a_score_too_small_for_two_decimals_is_shown_in_full() -> None:
    found = SearchResult((Hit("n/a.md", "A", "1-1", 1.6e-06, None), Hit("n/b.md", "B", "1-1", 0.0, None)), 1, ())
    assert hits(found, 0.25, "linux", PLAIN).results[::2] == ("1  1.6e-06  A", "2     0.00  B")


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


def test_links_show_where_each_leads_and_the_lines_linking_in() -> None:
    found = Links(
        "v/Offer.md",
        "Offer",
        (
            LinkOut(3, "Hans", "org", "Team", "alias", "v/Hans.md", "Hans", (), None),
            LinkOut(4, "Dup", None, None, "ambiguous", None, None, ("v/A/Dup.md", "v/B/Dup.md"), None),
            LinkOut(5, "Gone", None, None, "unresolved", None, None, (), "v/Notes.md"),
        ),
        (LinkIn("v/Notes.md", "Notes", (LinkingLine(3, None, "[[Offer]] again"),), None),),
    )
    assert shown_links(found, "linux", PLAIN).results == (
        "Offer (v/Offer.md) links to:",
        "     3  org: Hans#Team  ->  v/Hans.md (alias)",
        "     4  Dup  ->  ambiguous: v/A/Dup.md, v/B/Dup.md",
        "     5  Gone  ->  unresolved  (through v/Notes.md)",
        "Linking to Offer (v/Offer.md):",
        "  v/Notes.md  Notes",
        "     3  [[Offer]] again",
    )
    alone = Links("v/A.md", "A", (), None)
    assert shown_links(alone, "linux", PLAIN).results == ("A (v/A.md) links to: nothing",)


def test_unresolved_targets_show_how_many_link_them_and_a_few_of_those() -> None:
    found = Unresolved((Target("Missing", 7, ("v/a.md", "v/b.md")), Target("Gone", 1, ("v/c.md",))))
    assert shown_unresolved(found, "linux", PLAIN).results == ("7  Missing  v/a.md, v/b.md, ...", "1  Gone     v/c.md")
    assert shown_unresolved(Unresolved(()), "linux", PLAIN).messages == ("nothing  every link leads to a note",)


def test_unresolved_targets_of_a_store_that_could_not_be_read_are_said_to_be_missing() -> None:
    found = Unresolved((Target("Gone", 1, ("v/c.md",)),), (Unsearched("w", "the store is damaged", "run sift index"),))
    assert shown_unresolved(found, "linux", PLAIN).messages[0] == "error  w: the store is damaged"
    alone = shown_unresolved(Unresolved((), found.unsearched), "linux", PLAIN)
    assert (alone.results, len(alone.messages)) == ((), 2)


def test_links_of_a_note_found_while_a_store_could_not_be_read_say_so() -> None:
    found = Links("v/A.md", "A", (), None, (Unsearched("w", "the store is damaged", "run sift index"),))
    shown_found = shown_links(found, "linux", PLAIN)
    assert shown_found.results == ("A (v/A.md) links to: nothing",)
    assert shown_found.messages[0] == "error  w: the store is damaged"
