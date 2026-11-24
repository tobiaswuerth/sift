from sift.api.collections import (
    CollectionAdded,
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
from sift.api.documents import Documents, DocumentText, Unfound
from sift.api.embedding import CollectionEmbedded, Embedded, EmbedResult
from sift.api.errors import Problem
from sift.api.indexing import CollectionIndexed, Indexed, IndexResult, Skipped
from sift.api.links import LinkIn, LinkingLine, LinkOut, Links, Target, Unresolved
from sift.api.search import Hit, SearchResult, Unsearched
from sift.interfaces.cli.context import Answer

UNSEARCHED = Unsearched("papers", "the store is busy", "wait, then ask again", Problem.STORE_UNAVAILABLE)
SAMPLES: dict[str, Answer] = {
    "add": CollectionAdded(
        "notes",
        "/data/notes",
        "/home/.sift/collections/notes.sqlite",
        TitleFrom.HEADING,
        vault=True,
        follow_links=False,
    ),
    "rm": CollectionRemoved("diary", "/home/.sift/collections/diary.sqlite", 3, None, ("/s/diary.sqlite-wal",)),
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
                Unavailable("gone", "add it"),
                active=False,
            ),
        ),
    ),
    "index": IndexResult(
        (CollectionIndexed("notes", Indexed(3, 1, 1, 0, 1)), CollectionIndexed("papers", Skipped("gone", "add it")))
    ),
    "embed": EmbedResult(
        (CollectionEmbedded("notes", Embedded(7, 0)), CollectionEmbedded("papers", Skipped("gone", "add it")))
    ),
    "get": Documents(
        (DocumentText("notes/a.md", "A", "# A\n", {"year": 2024}),),
        (
            Unfound(
                "notes/b.md",
                "unknown document 'notes/b.md'",
                "take the id from a search result",
                Problem.UNKNOWN_DOCUMENT,
            ),
        ),
    ),
    "search": SearchResult(
        (Hit("notes/a.md", "notes", "a.md", "A", 1, 2, 1.5, "# A", ("copy/a.md",), {"year": 2024}),),
        1,
        (UNSEARCHED,),
        partial=True,
        ruled_out=1,
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
