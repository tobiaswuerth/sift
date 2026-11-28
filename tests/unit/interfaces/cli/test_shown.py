import dataclasses
import errno
import json

from sift.api.collections import (
    CollectionAdded,
    CollectionChanged,
    CollectionRemoved,
    CollectionsSwitched,
    CollectionStatus,
    Counted,
    Status,
    StoreMoved,
    StoresMoved,
    TitleFrom,
    Unavailable,
    UnreadableFile,
)
from sift.api.documents import Documents, DocumentText
from sift.api.errors import InvalidRequestError, NotFoundError, Problem, StoreUnavailableError
from sift.api.indexing import CollectionIndexed, Indexed, IndexResult
from sift.api.notices import Notice
from sift.interfaces.cli.shown.collections import added, moved, removed, status, switched
from sift.interfaces.cli.shown.documents import documents
from sift.interfaces.cli.shown.indexing import said_as_it_went
from sift.interfaces.cli.shown.outcome import FAILED, USAGE, Shown, failure
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
            Unavailable(
                Problem.STORE_MISSING, "the store is missing: /s/papers.sqlite", "connect the drive it lies on"
            ),
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
            "collection   folder        documents  passages    size",
            "notes        /data/notes      12,431    76,738  412 MB",
            "Papers-日本  /data/papers          -         -       -",
        ),
        (
            "error  Papers-日本: the store is missing: /s/papers.sqlite",
            "  fix  connect the drive it lies on",
        ),
    )


def test_status_counts_links_where_a_collection_holds_any() -> None:
    linked = dataclasses.replace(NOTES, contents=Counted(1, 2, 3_000, 0, (), links=3_249, unresolved=7))
    assert status(Status("/h", (linked, STATUS.collections[1])), "linux", PLAIN).results == (
        "collection   folder        documents  passages    size  links  unresolved",
        "notes        /data/notes           1         2  3.0 kB  3,249           7",
        "Papers-日本  /data/papers          -         -       -      -           -",
    )


def test_status_counts_links_where_none_leads_nowhere() -> None:
    linked = dataclasses.replace(NOTES, contents=Counted(1, 2, 3_000, 0, (), links=5, unresolved=0))
    assert status(Status("/h", (linked,)), "linux", PLAIN).results[0].split()[-2:] == ["links", "unresolved"]


def test_status_counts_unreadable_files_and_lists_them_when_asked() -> None:
    failed = (UnreadableFile("huge.txt", "too large"), UnreadableFile("locked.md", "Permission denied"))
    counted = dataclasses.replace(NOTES, contents=Counted(1, 2, 3_000, 2, failed))
    assert status(Status("/h", (counted,)), "linux", PLAIN).results == (
        "collection  folder       documents  passages    size  unreadable",
        "notes       /data/notes          1         2  3.0 kB           2",
        "",
        "collection  file       problem",
        "notes       huge.txt   too large",
        "notes       locked.md  Permission denied",
    )


def test_status_counts_what_is_embedded_by_which_model_and_says_what_searches_leave_out() -> None:
    embedded = dataclasses.replace(NOTES, contents=Counted(1, 2, 3_000, 0, (), embedded=2, model="google/x"))
    resting = dataclasses.replace(NOTES, name="old", contents=Counted(1, 2, 3_000, 0, ()), active=False)
    gone = dataclasses.replace(STATUS.collections[1], active=False)
    assert status(Status("/h", (embedded, resting, gone)), "linux", PLAIN).results == (
        "collection   folder        documents  passages  embedded    size  active",
        "notes        /data/notes           1         2         2  3.0 kB     yes",
        "old          /data/notes           1         2         0  3.0 kB      no",
        "Papers-日本  /data/papers          -         -         -       -      no",
        "",
        "model  google/x",
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
    unfound = (
        Notice(
            Problem.UNKNOWN_DOCUMENT,
            "no note or document 'notes/x.md'",
            "take the id from a search result",
            id="notes/x.md",
        ),
    )
    assert documents(Documents(found, unfound), "linux", PLAIN) == Shown(
        ("==> notes/a.md <==", "# A", "", "==> notes/b.md <==", "# B"),
        ("error  no note or document 'notes/x.md'", "  fix  take the id from a search result"),
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
        (), ("error  No space left on device", "  fix  free space on the disk"), FAILED
    )
    assert (
        failure(OSError(errno.EIO, "Input/output error"), PLAIN).messages[1]
        == "  fix  check the path and its permissions"
    )
    assert failure(InvalidRequestError("-k 0 asks for no result", "ask for one"), PLAIN).code == USAGE


def test_a_failure_of_a_field_or_a_part_of_one_names_its_flag() -> None:
    flags = {"filters": "--filter", "k": "-k"}
    for field in ("filters", "filters.values", "filters[2].values[0]"):
        refused = InvalidRequestError("a range takes one value", "give one", field)
        assert failure(refused, PLAIN, flags=flags).messages[0] == "error  --filter: a range takes one value"
    unknown = NotFoundError("no collection is named 'x'", "add it", "collections")
    assert failure(unknown, PLAIN, flags=flags).messages[0] == "error  no collection is named 'x'"


def test_a_failure_asked_for_in_json_is_json_too() -> None:
    found = failure(StoreUnavailableError("busy", "wait"), PLAIN, as_json=True)
    assert json.loads(found.results[0]) == {
        "error": {
            "kind": "store_unavailable",
            "code": "store_unavailable",
            "message": "busy",
            "remedy": "wait",
            "field": None,
        }
    }
    assert found.messages == ("error  busy", "  fix  wait")
    assert json.loads(failure(OSError(errno.ENOENT, "gone"), PLAIN, as_json=True).results[0])["error"]["kind"] == "os"


def test_a_collection_changed_says_its_choices_now_and_what_to_run_next() -> None:
    changed = CollectionChanged(
        "notes", TitleFrom.NAME, follow_links=True, agents=False, retitled=2, to_embed=1, index_again=True
    )
    assert added(changed, "linux", PLAIN).messages == (
        "title  notes: from file names: 2 titles changed",
        "links  notes: files linked from outside the folder read",
        "agent  notes: hidden from agents",
        "embed  1 passage to embed again, which carry their title",
        " next  sift index notes",
        " next  sift embed notes",
    )
    plain = CollectionChanged("notes", TitleFrom.HEADING, follow_links=False, agents=True, retitled=0, to_embed=0)
    assert added(plain, "linux", PLAIN).messages[-1] == "agent  notes: seen by agents"


def test_a_removal_says_what_its_store_held_and_names_the_files_left_in_use() -> None:
    gone = CollectionRemoved("diary", "/s/diary.sqlite", 1, None)
    said = " gone  diary: its store of 1 document deleted; its files stay where they are"
    assert removed(gone, "linux", PLAIN) == Shown((), (said,))
    missing = Notice(Problem.STORE_MISSING, "it is missing", "connect the drive it lies on")
    unread = CollectionRemoved("diary", "/s/diary.sqlite", None, missing, ("/s/diary.sqlite-wal",))
    assert removed(unread, "linux", PLAIN).messages == (
        " gone  diary: its store deleted; its files stay where they are",
        "store  could not be read: it is missing",
        " left  /s/diary.sqlite-wal: in use, not deleted",
        "  fix  the next sift index deletes it, once the program using it has stopped",
    )


def test_a_move_says_where_each_store_went_how_and_what_it_left() -> None:
    copied = StoreMoved("diary", "/s/diary.sqlite", "/t/diary.sqlite", copied=True, left=("/s/diary.sqlite",))
    linked = StoreMoved("notes", "/s/notes.sqlite", "/t/notes.sqlite", copied=False)
    assert moved(StoresMoved((copied, linked), ("papers",)), "linux", PLAIN).messages == (
        "moved  diary  /t/diary.sqlite, copied: it lay on another drive",
        " left  /s/diary.sqlite: in use, not deleted",
        "  fix  the next sift index deletes it, once the program using it has stopped",
        "moved  notes  /t/notes.sqlite",
        " kept  papers: there already",
    )


def test_switching_collections_says_which_changed_and_which_were_so_already() -> None:
    off = CollectionsSwitched(active=False, changed=("diary", "recipes"), already=("notes",))
    assert switched(off, "linux", PLAIN).messages == (
        "  off  diary, recipes: left out of searches, and of index and embed unless named",
        " kept  notes: inactive already",
    )
    on = CollectionsSwitched(active=True, changed=("diary",), already=())
    assert switched(on, "linux", PLAIN).messages == ("   on  diary: searched again",)


def test_a_document_shows_the_front_matter_asked_for_before_its_text() -> None:
    found = DocumentText("notes/a.md", "A", "# A\n", fields={"rank": 2, "tags": ["x"]})
    assert documents(Documents((found,), ()), "linux", PLAIN).results == ("rank: 2", 'tags: ["x"]', "", "# A")
