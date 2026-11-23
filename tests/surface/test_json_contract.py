import json
from pathlib import Path

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
from sift.api.embedding import CollectionEmbedded, Embedded, EmbedResult
from sift.api.indexing import CollectionIndexed, Indexed, IndexResult, Skipped
from sift.api.links import LinkIn, LinkingLine, LinkOut, Links, Target, Unresolved
from sift.api.search import Hit, SearchResult, Unsearched
from sift.interfaces.cli.commands.making import as_json
from sift.interfaces.cli.context import Answer

SNAPSHOT = Path(__file__).with_name("json_contract.json")
UNSEARCHED = Unsearched("papers", "the store is busy", "wait, then ask again")
SAMPLES: dict[str, Answer] = {
    "add": CollectionAdded(
        "notes",
        "/data/notes",
        "/home/.sift/collections/notes.sqlite",
        TitleFrom.HEADING,
        vault=True,
        follow_links=False,
    ),
    "status": Status(
        "/home/.sift",
        (
            CollectionStatus(
                "notes",
                "/data/notes",
                "/home/.sift/collections/notes.sqlite",
                Counted(3, 7, 4096, 1, (UnreadableFile("bad.pdf", "no text"),), 2, 1),
            ),
            CollectionStatus(
                "papers", "/data/papers", "/home/.sift/collections/papers.sqlite", Unavailable("gone", "add it")
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
        (Unfound("notes/b.md", "unknown document 'notes/b.md'", "take the id from a search result"),),
    ),
    "search": SearchResult(
        (Hit("notes/a.md", "A", "1-2", 1.5, "# A", ("copy/a.md",), {"year": 2024}),),
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


def test_the_json_of_every_answer_keeps_its_shape() -> None:
    found = {name: json.loads(as_json(answer)) for name, answer in SAMPLES.items()}
    assert found == json.loads(SNAPSHOT.read_text(encoding="utf-8"))
