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
from sift.api.embedding import CollectionEmbedded, Embedded, EmbedResult
from sift.api.errors import Problem
from sift.api.indexing import CollectionIndexed, FileIndexed, Indexed, IndexResult, Skipped
from sift.api.links import LinkIn, LinkingLine, LinkOut, Links, Target, Unresolved
from sift.api.notices import Notice
from sift.api.search import Hit, SearchResult
from sift.interfaces.cli.context import Answer

UNSEARCHED = Notice(Problem.BUSY, "the store is busy", "wait, then ask again", "papers")
SAMPLES: dict[str, Answer] = {
    "add": CollectionAdded(
        "notes",
        "/data/notes",
        "/home/.sift/collections/notes.sqlite",
        TitleFrom.HEADING,
        vault=True,
        follow_links=False,
    ),
    "add again": CollectionChanged(
        "notes", TitleFrom.NAME, follow_links=True, agents=False, retitled=2, to_embed=1, index_again=True
    ),
    "rm": CollectionRemoved("diary", "/home/.sift/collections/diary.sqlite", 3, None, ("/s/diary.sqlite-wal",)),
    "rm unread": CollectionRemoved(
        "diary", "/home/.sift/collections/diary.sqlite", None, Notice(Problem.STORE_DAMAGED, "it is damaged", "move it")
    ),
    "move": StoresMoved(
        (StoreMoved("diary", "/s/diary.sqlite", "/t/diary.sqlite", copied=True, left=("/s/diary.sqlite-shm",)),),
        ("notes",),
    ),
    "activate": CollectionsSwitched(active=True, changed=("diary",), already=()),
    "deactivate": CollectionsSwitched(active=False, changed=("diary",), already=("notes",)),
    "status": Status(
        "/home/.sift",
        (
            CollectionStatus(
                "notes",
                "/data/notes",
                "/home/.sift/collections/notes.sqlite",
                Counted(3, 7, 4096, 1, (UnreadableFile("bad.pdf", "no text"),), 2, 1, 5, "a-model", ("year",)),
            ),
            CollectionStatus(
                "papers",
                "/data/papers",
                "/home/.sift/collections/papers.sqlite",
                Unavailable(Problem.STORE_MISSING, "gone", "add it"),
                active=False,
            ),
        ),
    ),
    "index": IndexResult(
        (
            CollectionIndexed("notes", Indexed(3, 1, 1, 0, 1)),
            CollectionIndexed("papers", Skipped(Problem.FOLDER_MISSING, "gone", "add it")),
        )
    ),
    "index files": IndexResult(
        (
            CollectionIndexed(
                "notes",
                Indexed(2, 0, 1, 0, 1),
                (
                    FileIndexed("notes/rl.md", "changed"),
                    FileIndexed("notes/scan.pdf", "unreadable", "no text"),
                    FileIndexed("notes/locked.md", "unreadable", "Access is denied", Problem.OS),
                ),
            ),
        )
    ),
    "embed": EmbedResult(
        (
            CollectionEmbedded("notes", Embedded(7, 0)),
            CollectionEmbedded("papers", Skipped(Problem.STORE_MISSING, "gone", "add it")),
        )
    ),
    "get": Documents(
        (DocumentText("notes/a.md", "A", "# A\n", {"year": 2024}),),
        (
            Notice(
                Problem.UNKNOWN_DOCUMENT,
                "unknown document 'notes/b.md'",
                "take the id from a search result",
                id="notes/b.md",
            ),
        ),
    ),
    "search": SearchResult(
        (Hit("notes/a.md", "notes", "a.md", "A", 1, 2, 1.5, "# A", ("copy/a.md",), {"year": 2024}),),
        1,
        (UNSEARCHED,),
        partial=True,
        ruled_out=1,
        left_out=(Notice(Problem.NOT_EMBEDDED, "1 document has no vectors: searched by keyword alone", None, "notes"),),
    ),
    "links": Links(
        "notes/a.md",
        "A",
        (LinkOut(3, "B", "org", "Intro", "name", "notes/b.md", "B", (), None),),
        (LinkIn("notes/c.md", "C", (LinkingLine(4, None, "see [[A]]"),), None),),
        (UNSEARCHED,),
    ),
    "unresolved": Unresolved((Target("Missing", 2, ("notes/a.md", "notes/c.md")),), (UNSEARCHED,)),
}
