import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

from sift.api.bench import Mode, ModeMeasured, SearchesMeasured
from sift.api.collections import (
    ActivateCollections,
    AddCollection,
    DeactivateCollections,
    MoveCollections,
    RemoveCollection,
    ShowStatus,
    TitleFrom,
)
from sift.api.documents import GetDocuments, WantedDocument
from sift.api.embedding import EmbedCollections
from sift.api.errors import CODE_KINDS, InvalidRequestError, NotFoundError, Problem, SiftError
from sift.api.events import (
    Advanced,
    CollectionStarted,
    Event,
    FilesKept,
    Finished,
    Lending,
    Listening,
    Noted,
    Phase,
    PhaseDone,
    PhaseStarted,
)
from sift.api.filters import Filter, Operator
from sift.api.indexing import IndexCollections
from sift.api.jobs import ListJobs
from sift.api.links import Direction, FindMentions, FindSimilar, ListUnresolved, ShowLinks
from sift.api.search import Options, SearchHybrid, SearchKeywords, SearchVectors
from sift.api.serving import Served


def searched[T: Options](request: T) -> T:
    return dataclasses.replace(
        request,
        k=5,
        collections=("notes",),
        path="2024/*",
        filters=(Filter("year", Operator.AT_LEAST, (2020,)),),
        fields=("url",),
        sort="-created",
        links_to=("Alice",),
        linked_from=("Bob",),
        all_passages=True,
        compact=True,
        adaptive=True,
        offset=5,
        budget=0.5,
    )


REQUEST_SAMPLES: Mapping[str, object] = MappingProxyType(
    {
        "add": AddCollection(
            "/data/notes",
            name="notes",
            store="/data/notes.sqlite",
            title_from=TitleFrom.NAME,
            nested=True,
            follow_links=True,
            agents=False,
            id_field="key",
            title_field="title",
            text_fields=("body",),
        ),
        "rm": RemoveCollection("diary"),
        "move": MoveCollections(("diary",), to="/data/stores"),
        "activate": ActivateCollections(("diary",)),
        "deactivate": DeactivateCollections(("diary",)),
        "index": IndexCollections(("notes",), files=("notes/a.md",)),
        "embed": EmbedCollections(("notes",), reset=True),
        "status": ShowStatus(("notes",), unreadable=True),
        "get": GetDocuments((WantedDocument("notes/a.md", 2, 4),), fields=("url",)),
        "search": searched(SearchKeywords("halo")),
        "vsearch": searched(SearchVectors("dark matter")),
        "query": dataclasses.replace(searched(SearchHybrid("dark matter")), candidates=60),
        "mentions": dataclasses.replace(searched(FindMentions("notes/Alice.md")), also=("emails",)),
        "similar": searched(FindSimilar("notes/a.md")),
        "links": ShowLinks("notes/a.md", direction=Direction.BOTH, depth=2, field="related", k=20),
        "unresolved": ListUnresolved(("notes",), k=20),
        "jobs": ListJobs(stop=3),
    }
)
ERROR_SAMPLES: Mapping[str, SiftError] = MappingProxyType(
    {
        "invalid_request of a field": InvalidRequestError("k is 0, outside 1 to 10,000", "give 1 to 10,000", "k"),
        "unknown_collection of a field": NotFoundError(
            "no collection is named 'x'", "the collections are: notes", "collections", code=Problem.UNKNOWN_COLLECTION
        ),
        **{code.value: CODE_KINDS[code](f"a refusal coded {code.value}", "what to do", code=code) for code in Problem},
    }
)
EVENT_SAMPLES: Mapping[str, Event] = MappingProxyType(
    {
        "collection started": CollectionStarted("notes", "/data/notes"),
        "phase started": PhaseStarted(Phase.READ, 3),
        "advanced": Advanced(2),
        "files kept": FilesKept(2),
        "phase done": PhaseDone(3),
        "noted": Noted("a note", "a fix"),
        "noted with a code": Noted("a folder can't be read", None, Problem.OS),
        "finished": Finished(1, 2, 3, 4),
        "listening": Listening("http://127.0.0.1:8765", "/home/.sift"),
        "lending": Lending("workstation", 8766, "t0ken.p1n"),
    }
)
SERVED = Served("http://127.0.0.1:8765")
MEASURED = SearchesMeasured(
    2, (ModeMeasured(Mode.KEYWORD, 0.5, 0.75, 1.0, 12.5), ModeMeasured(Mode.HYBRID, 1.0, 1.0, 1.0, 250.0))
)
