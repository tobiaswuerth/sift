import dataclasses

from sift.api.collections import CollectionAdded, CollectionStatus, Counted, Status, TitleFrom, Unavailable
from sift.api.errors import NotFoundError, StoreUnavailableError
from sift.interfaces.cli.shown import Shown, added, failure, status

ADDED = CollectionAdded("notes", "/data/notes", "/home/.sift/collections/notes.sqlite", TitleFrom.HEADING, vault=False)
STATUS = Status(
    "/home/.sift",
    (
        CollectionStatus("notes", "/data/notes", "/s/notes.sqlite", Counted(12_431, 76_738, 412_300_000)),
        CollectionStatus(
            "Papers-日本",
            "/data/papers",
            "/s/papers.sqlite",
            Unavailable("the store is missing: /s/papers.sqlite", "connect the drive it lies on"),
        ),
    ),
)


def test_an_added_collection_shows_where_its_store_lies_and_what_to_run_next() -> None:
    assert added(ADDED, "linux") == Shown(
        (),
        (
            "added  notes  /data/notes",
            "store  /home/.sift/collections/notes.sqlite",
            " next  sift index notes",
        ),
    )


def test_titles_from_file_names_are_said_with_the_vault_that_chose_them() -> None:
    by_name = dataclasses.replace(ADDED, title_from=TitleFrom.NAME)
    assert added(by_name, "linux").messages[2] == "title  from file names"
    vault = dataclasses.replace(by_name, vault=True)
    assert added(vault, "linux").messages[2] == "title  from file names, as in an Obsidian vault"
    assert "title" not in "".join(added(dataclasses.replace(ADDED, vault=True), "linux").messages)


def test_the_next_command_is_quoted_for_the_shell_there() -> None:
    named = dataclasses.replace(ADDED, name="Müller")
    assert added(named, "linux").messages[-1] == " next  sift index 'Müller'"
    assert added(named, "win32").messages[-1] == " next  sift index Müller"


def test_status_is_a_table_whose_numbers_align_and_whose_problems_are_said_beside_it() -> None:
    assert status(STATUS) == Shown(
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


def test_status_of_no_collection_says_how_to_add_one() -> None:
    assert status(Status("/home/.sift", ())) == Shown((), ("empty  no collections yet", " next  sift add FOLDER"))


def test_a_failure_says_what_failed_and_how_to_fix_it() -> None:
    assert failure(NotFoundError("no collection is named 'x'", "the collections are: a")) == Shown(
        (), ("error  no collection is named 'x'", "  fix  the collections are: a")
    )
    assert failure(StoreUnavailableError("busy")) == Shown((), ("error  busy",))
    assert failure(OSError(28, "No space left on device")) == Shown((), ("error  [Errno 28] No space left on device",))
